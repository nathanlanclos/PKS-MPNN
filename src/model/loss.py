"""
Loss functions for PKS-MPNN training.

Implements pLDDT-aware weighted loss following recommendations from
Gemini and GPT:
- pLDDT > 70: Full loss weight
- 50 < pLDDT <= 70: Mask from loss (in graph, not trained)
- pLDDT <= 50: Exclude from input graph
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


def compute_loss_masks(
    plddt: torch.Tensor,
    high_threshold: float = 70.0,
    low_threshold: float = 50.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute tiered masks based on pLDDT confidence.
    
    Args:
        plddt: Shape (B, L) or (L,) pLDDT scores (0-100)
        high_threshold: Threshold for loss contribution
        low_threshold: Threshold for graph inclusion
        
    Returns:
        loss_mask: Shape (B, L) or (L,) - residues contributing to loss
        input_mask: Shape (B, L) or (L,) - residues included in graph
    """
    loss_mask = (plddt >= high_threshold).float()
    input_mask = (plddt >= low_threshold).float()
    
    return loss_mask, input_mask


class PLDDTWeightedLoss(nn.Module):
    """
    Cross-entropy loss weighted by pLDDT confidence.
    
    Can operate in two modes:
    1. Binary masking: residues either contribute or don't
    2. Soft weighting: loss weighted proportionally to pLDDT
    """
    
    def __init__(
        self,
        use_soft_weighting: bool = False,
        high_threshold: float = 70.0,
        low_threshold: float = 50.0,
        label_smoothing: float = 0.0
    ):
        """
        Initialize the loss function.
        
        Args:
            use_soft_weighting: Use continuous pLDDT weights instead of binary
            high_threshold: pLDDT threshold for loss contribution
            low_threshold: pLDDT threshold for graph inclusion
            label_smoothing: Label smoothing factor
        """
        super().__init__()
        self.use_soft_weighting = use_soft_weighting
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.label_smoothing = label_smoothing
    
    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        plddt: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute weighted cross-entropy loss.
        
        Args:
            log_probs: Shape (B, L, 21) predicted log probabilities
            targets: Shape (B, L) target sequence indices
            mask: Shape (B, L) which positions are valid (not padding)
            plddt: Optional shape (B, L) confidence scores
            loss_mask: Optional pre-computed loss mask
            
        Returns:
            loss: Scalar loss value
            metrics: Dict with additional metrics
        """
        B, L, V = log_probs.shape
        
        # Compute per-residue cross-entropy
        # Reshape for cross_entropy: (B*L, V) and (B*L,)
        log_probs_flat = log_probs.reshape(-1, V)
        targets_flat = targets.reshape(-1)
        
        if self.label_smoothing > 0:
            # Manual label smoothing
            ce_loss = F.cross_entropy(
                log_probs_flat, targets_flat, reduction='none'
            )
        else:
            ce_loss = F.cross_entropy(
                log_probs_flat, targets_flat, reduction='none'
            )
        
        ce_loss = ce_loss.reshape(B, L)
        
        # Compute loss weights
        if loss_mask is not None:
            weights = loss_mask * mask
        elif plddt is not None:
            if self.use_soft_weighting:
                # Soft weighting: linear scaling from low to high threshold
                weights = torch.clamp(
                    (plddt - self.low_threshold) / (self.high_threshold - self.low_threshold),
                    0.0, 1.0
                )
                weights = weights * mask
            else:
                # Binary masking
                weights = (plddt >= self.high_threshold).float() * mask
        else:
            weights = mask
        
        # Weighted loss
        weighted_loss = ce_loss * weights
        total_weight = weights.sum() + 1e-8
        loss = weighted_loss.sum() / total_weight
        
        # Compute accuracy
        predictions = log_probs.argmax(dim=-1)
        correct = (predictions == targets).float() * weights
        accuracy = correct.sum() / total_weight
        
        # Metrics
        metrics = {
            'loss': loss.detach(),
            'accuracy': accuracy.detach(),
            'perplexity': torch.exp(loss).detach(),
            'n_residues': total_weight.detach(),
        }
        
        return loss, metrics


class SmoothCrossEntropyLoss(nn.Module):
    """
    Cross-entropy with label smoothing.
    
    This is the loss function used in the original ProteinMPNN.
    """
    
    def __init__(self, smoothing: float = 0.1):
        """
        Args:
            smoothing: Label smoothing factor (0 = no smoothing)
        """
        super().__init__()
        self.smoothing = smoothing
    
    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute smoothed cross-entropy loss.
        
        Args:
            log_probs: Shape (B, L, V) predicted log probabilities
            targets: Shape (B, L) target sequence indices
            mask: Shape (B, L) which positions are valid
            
        Returns:
            total_loss: Per-residue loss (masked)
            avg_loss: Average loss
        """
        V = log_probs.shape[-1]
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, V).float()
        
        # Apply label smoothing
        if self.smoothing > 0:
            targets_smooth = targets_one_hot * (1 - self.smoothing) + self.smoothing / V
        else:
            targets_smooth = targets_one_hot
        
        # Compute cross-entropy
        loss = -(targets_smooth * log_probs).sum(dim=-1)
        
        # Apply mask
        masked_loss = loss * mask
        avg_loss = masked_loss.sum() / (mask.sum() + 1e-8)
        
        return masked_loss, avg_loss


def compute_nll_loss(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute negative log-likelihood loss and accuracy.
    
    This matches the loss_nll function from original ProteinMPNN.
    
    Args:
        log_probs: Shape (B, L, V) predicted log probabilities
        targets: Shape (B, L) target sequence indices
        mask: Shape (B, L) which positions are valid
        
    Returns:
        loss: Per-residue NLL loss
        avg_loss: Average NLL loss
        correct: Per-residue correctness
    """
    # Cross-entropy (NLL) per residue
    criterion = nn.NLLLoss(reduction='none')
    
    B, L, V = log_probs.shape
    loss = criterion(log_probs.reshape(-1, V), targets.reshape(-1))
    loss = loss.reshape(B, L)
    
    # Average over valid residues
    avg_loss = (loss * mask).sum() / (mask.sum() + 1e-8)
    
    # Accuracy
    predictions = log_probs.argmax(dim=-1)
    correct = (predictions == targets).float()
    
    return loss, avg_loss, correct


def compute_perplexity(
    loss: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute perplexity from loss.
    
    Args:
        loss: Per-residue loss
        mask: Which residues are valid
        
    Returns:
        Perplexity value
    """
    avg_loss = (loss * mask).sum() / (mask.sum() + 1e-8)
    return torch.exp(avg_loss)
