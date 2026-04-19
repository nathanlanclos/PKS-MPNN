"""
Evaluation metrics for PKS-MPNN.

Includes:
- Perplexity (overall and per-domain)
- Sequence recovery (overall and per-domain)
- NLL loss
- Confidence-stratified metrics
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


def compute_perplexity(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute perplexity = exp(cross-entropy loss).
    
    Args:
        log_probs: Shape (B, L, V) predicted log probabilities
        targets: Shape (B, L) target sequence indices
        mask: Shape (B, L) which positions are valid
        
    Returns:
        Perplexity value
    """
    # NLL per residue
    B, L, V = log_probs.shape
    nll = -log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    
    # Average NLL
    avg_nll = (nll * mask).sum() / (mask.sum() + 1e-8)
    
    return torch.exp(avg_nll)


def compute_recovery(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute sequence recovery (accuracy).
    
    Args:
        log_probs: Shape (B, L, V) predicted log probabilities
        targets: Shape (B, L) target sequence indices
        mask: Shape (B, L) which positions are valid
        
    Returns:
        Recovery rate (0-1)
    """
    predictions = log_probs.argmax(dim=-1)
    correct = (predictions == targets).float()
    recovery = (correct * mask).sum() / (mask.sum() + 1e-8)
    return recovery


class PerDomainMetrics:
    """
    Compute metrics stratified by domain type.
    
    Allows analysis of model performance on different PKS domains
    (KS, AT, DH, ER, KR, ACP, linkers, etc.)
    """
    
    # Domain type indices (matching annotation_parser)
    DOMAIN_NAMES = {
        0: 'unknown',
        1: 'KS',
        2: 'AT',
        3: 'DH',
        4: 'ER',
        5: 'KR',
        6: 'ACP',
        7: 'oMT',
        8: 'C',
        9: 'A',
        10: 'PCP',
        11: 'E',
        12: 'linker',
    }
    
    def __init__(self):
        """Initialize metric accumulators."""
        self.reset()
    
    def reset(self):
        """Reset all accumulators."""
        self._nll_sum = defaultdict(float)
        self._correct_sum = defaultdict(float)
        self._count = defaultdict(float)
    
    def update(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        domain_labels: torch.Tensor
    ):
        """
        Update metrics with a batch.
        
        Args:
            log_probs: Shape (B, L, V) predicted log probabilities
            targets: Shape (B, L) target sequence indices
            mask: Shape (B, L) which positions are valid
            domain_labels: Shape (B, L) domain type indices
        """
        # Compute per-residue metrics
        B, L, V = log_probs.shape
        nll = -log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        predictions = log_probs.argmax(dim=-1)
        correct = (predictions == targets).float()
        
        # Aggregate by domain type
        for domain_idx in range(13):  # 0-12
            domain_mask = (domain_labels == domain_idx).float() * mask
            count = domain_mask.sum().item()
            
            if count > 0:
                self._nll_sum[domain_idx] += (nll * domain_mask).sum().item()
                self._correct_sum[domain_idx] += (correct * domain_mask).sum().item()
                self._count[domain_idx] += count
    
    def compute(self) -> Dict[str, Dict[str, float]]:
        """
        Compute final metrics for each domain type.
        
        Returns:
            Dict mapping domain name to metrics dict
        """
        results = {}
        
        for domain_idx, domain_name in self.DOMAIN_NAMES.items():
            count = self._count[domain_idx]
            
            if count > 0:
                avg_nll = self._nll_sum[domain_idx] / count
                perplexity = np.exp(avg_nll)
                recovery = self._correct_sum[domain_idx] / count
                
                results[domain_name] = {
                    'perplexity': perplexity,
                    'recovery': recovery,
                    'nll': avg_nll,
                    'n_residues': int(count),
                }
        
        return results


class ConfidenceStratifiedMetrics:
    """
    Compute metrics stratified by pLDDT confidence bins.
    
    This helps understand if the model performs differently on
    high vs low confidence regions.
    """
    
    def __init__(
        self,
        bins: List[float] = [0, 50, 70, 90, 100]
    ):
        """
        Initialize with confidence bins.
        
        Args:
            bins: Bin edges for pLDDT (default: very low, low, medium, high)
        """
        self.bins = bins
        self.bin_names = [
            f"pLDDT_{bins[i]}-{bins[i+1]}" 
            for i in range(len(bins) - 1)
        ]
        self.reset()
    
    def reset(self):
        """Reset accumulators."""
        self._nll_sum = defaultdict(float)
        self._correct_sum = defaultdict(float)
        self._count = defaultdict(float)
    
    def update(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        plddt: torch.Tensor
    ):
        """
        Update metrics with a batch.
        
        Args:
            log_probs: Shape (B, L, V) predicted log probabilities
            targets: Shape (B, L) target sequence indices
            mask: Shape (B, L) which positions are valid
            plddt: Shape (B, L) confidence scores (0-100)
        """
        # Compute per-residue metrics
        B, L, V = log_probs.shape
        nll = -log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        predictions = log_probs.argmax(dim=-1)
        correct = (predictions == targets).float()
        
        # Aggregate by confidence bin
        for i in range(len(self.bins) - 1):
            low, high = self.bins[i], self.bins[i + 1]
            bin_mask = ((plddt >= low) & (plddt < high)).float() * mask
            count = bin_mask.sum().item()
            
            if count > 0:
                bin_name = self.bin_names[i]
                self._nll_sum[bin_name] += (nll * bin_mask).sum().item()
                self._correct_sum[bin_name] += (correct * bin_mask).sum().item()
                self._count[bin_name] += count
    
    def compute(self) -> Dict[str, Dict[str, float]]:
        """Compute final metrics for each confidence bin."""
        results = {}
        
        for bin_name in self.bin_names:
            count = self._count[bin_name]
            
            if count > 0:
                avg_nll = self._nll_sum[bin_name] / count
                perplexity = np.exp(avg_nll)
                recovery = self._correct_sum[bin_name] / count
                
                results[bin_name] = {
                    'perplexity': perplexity,
                    'recovery': recovery,
                    'nll': avg_nll,
                    'n_residues': int(count),
                }
        
        return results


def log_metrics_to_wandb(
    metrics: Dict[str, float],
    per_domain: Optional[Dict[str, Dict[str, float]]] = None,
    per_confidence: Optional[Dict[str, Dict[str, float]]] = None,
    prefix: str = "train",
    step: Optional[int] = None
):
    """
    Log metrics to wandb.
    
    Args:
        metrics: Global metrics dict
        per_domain: Per-domain metrics
        per_confidence: Per-confidence-bin metrics
        prefix: Metric prefix (train/val/test)
        step: Training step
    """
    try:
        import wandb
        
        log_dict = {}
        
        # Global metrics
        for key, value in metrics.items():
            log_dict[f"{prefix}/{key}"] = value
        
        # Per-domain metrics
        if per_domain:
            for domain, domain_metrics in per_domain.items():
                for key, value in domain_metrics.items():
                    log_dict[f"{prefix}/domain_{domain}/{key}"] = value
        
        # Per-confidence metrics
        if per_confidence:
            for bin_name, bin_metrics in per_confidence.items():
                for key, value in bin_metrics.items():
                    log_dict[f"{prefix}/{bin_name}/{key}"] = value
        
        wandb.log(log_dict, step=step)
        
    except ImportError:
        pass  # wandb not installed
