"""
ProteinMPNN wrapper with fine-tuning support.

Wraps the original ProteinMPNN model to support:
- Loading pretrained weights
- Gradual unfreezing for fine-tuning
- pLDDT-aware inference
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def get_proteinmpnn_model_class():
    """
    Import ProteinMPNN model class from the cloned repository.
    
    Returns:
        ProteinMPNN model class
    """
    # Add ProteinMPNN to path
    proteinmpnn_path = Path(__file__).parent.parent.parent / "ProteinMPNN" / "training"
    if str(proteinmpnn_path) not in sys.path:
        sys.path.insert(0, str(proteinmpnn_path))
    
    try:
        from model_utils import ProteinMPNN
        return ProteinMPNN
    except ImportError:
        raise ImportError(
            "Could not import ProteinMPNN. Make sure ProteinMPNN is cloned in the repo root."
        )


class ProteinMPNNWrapper(nn.Module):
    """
    Wrapper around ProteinMPNN with fine-tuning utilities.
    
    Supports:
    - Loading pretrained weights from the original model
    - Gradual unfreezing (decoder only -> all -> with lower LR)
    - Forward pass compatible with our data format
    """
    
    def __init__(
        self,
        node_features: int = 128,
        edge_features: int = 128,
        hidden_dim: int = 128,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        k_neighbors: int = 48,
        dropout: float = 0.1,
        augment_eps: float = 0.0,
    ):
        """
        Initialize the model.
        
        Args:
            node_features: Node feature dimension
            edge_features: Edge feature dimension
            hidden_dim: Hidden layer dimension
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            k_neighbors: Number of neighbors for message passing
            dropout: Dropout rate
            augment_eps: Noise to add to coordinates during training
        """
        super().__init__()
        
        # Get the original ProteinMPNN class
        ProteinMPNN = get_proteinmpnn_model_class()
        
        # Initialize the model
        self.model = ProteinMPNN(
            node_features=node_features,
            edge_features=edge_features,
            hidden_dim=hidden_dim,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            k_neighbors=k_neighbors,
            dropout=dropout,
            augment_eps=augment_eps,
        )
        
        self.k_neighbors = k_neighbors
        self.hidden_dim = hidden_dim
    
    def load_pretrained(self, checkpoint_path: Path):
        """
        Load pretrained weights from ProteinMPNN checkpoint.
        
        Args:
            checkpoint_path: Path to .pt checkpoint file
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Load weights
        self.model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from {checkpoint_path}")
    
    def configure_unfreezing(self, phase: str) -> float:
        """
        Configure which parameters to train based on unfreezing phase.
        
        Args:
            phase: One of "decoder_only", "encoder_decoder", "full"
            
        Returns:
            Recommended learning rate multiplier
        """
        if phase == "decoder_only":
            # Freeze encoder, train decoder
            for name, param in self.model.named_parameters():
                if 'decoder' in name or 'W_out' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            return 1.0
            
        elif phase == "encoder_decoder":
            # Train everything
            for param in self.model.parameters():
                param.requires_grad = True
            return 0.5
            
        elif phase == "full":
            # Full training with lower LR
            for param in self.model.parameters():
                param.requires_grad = True
            return 0.1
        
        else:
            raise ValueError(f"Unknown phase: {phase}")
    
    def get_trainable_params(self) -> List[Dict]:
        """
        Get trainable parameters grouped by module.
        
        Returns:
            List of param groups for optimizer
        """
        encoder_params = []
        decoder_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'encoder' in name:
                    encoder_params.append(param)
                elif 'decoder' in name:
                    decoder_params.append(param)
                else:
                    other_params.append(param)
        
        return [
            {'params': encoder_params, 'name': 'encoder'},
            {'params': decoder_params, 'name': 'decoder'},
            {'params': other_params, 'name': 'other'},
        ]
    
    def forward(
        self,
        X: torch.Tensor,
        S: torch.Tensor,
        mask: torch.Tensor,
        chain_M: torch.Tensor,
        residue_idx: torch.Tensor,
        chain_encoding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through ProteinMPNN.
        
        Args:
            X: Shape (B, L, 4, 3) backbone coordinates [N, CA, C, O]
            S: Shape (B, L) sequence indices
            mask: Shape (B, L) which positions are valid
            chain_M: Shape (B, L) which residues are designable
            residue_idx: Shape (B, L) residue positions
            chain_encoding: Shape (B, L) chain assignments
            
        Returns:
            log_probs: Shape (B, L, 21) log probabilities over amino acids
        """
        # The original ProteinMPNN model expects chain_encoding_all
        # which includes information about all chains
        
        log_probs = self.model(
            X, S, mask, chain_M, residue_idx, chain_encoding
        )
        
        return log_probs
    
    def sample(
        self,
        X: torch.Tensor,
        mask: torch.Tensor,
        chain_M: torch.Tensor,
        residue_idx: torch.Tensor,
        chain_encoding: torch.Tensor,
        temperature: float = 0.1,
        n_samples: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample sequences from the model.
        
        Args:
            X: Shape (B, L, 4, 3) backbone coordinates
            mask: Shape (B, L) which positions are valid
            chain_M: Shape (B, L) which residues to design
            residue_idx: Shape (B, L) residue positions
            chain_encoding: Shape (B, L) chain assignments
            temperature: Sampling temperature
            n_samples: Number of sequences to sample
            
        Returns:
            sequences: Shape (B, n_samples, L) sampled sequence indices
            log_probs: Shape (B, n_samples) log probability of samples
        """
        self.eval()
        
        B, L = mask.shape
        device = X.device
        
        all_sequences = []
        all_log_probs = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                # Initialize with zeros
                S = torch.zeros(B, L, dtype=torch.long, device=device)
                sample_log_prob = torch.zeros(B, device=device)
                
                # Autoregressive sampling
                for i in range(L):
                    # Get log probs for current position
                    log_probs = self.forward(
                        X, S, mask, chain_M, residue_idx, chain_encoding
                    )
                    
                    # Sample at position i
                    probs = torch.softmax(log_probs[:, i, :] / temperature, dim=-1)
                    sampled = torch.multinomial(probs, 1).squeeze(-1)
                    
                    # Update sequence
                    S[:, i] = sampled
                    
                    # Accumulate log prob
                    sample_log_prob += log_probs[:, i, :].gather(1, sampled.unsqueeze(-1)).squeeze(-1) * mask[:, i]
                
                all_sequences.append(S)
                all_log_probs.append(sample_log_prob)
        
        sequences = torch.stack(all_sequences, dim=1)  # (B, n_samples, L)
        log_probs = torch.stack(all_log_probs, dim=1)  # (B, n_samples)
        
        return sequences, log_probs


def create_model(
    config: Dict,
    pretrained_path: Optional[Path] = None,
    device: str = "cuda"
) -> ProteinMPNNWrapper:
    """
    Create and optionally load pretrained ProteinMPNN model.
    
    Args:
        config: Model configuration dict
        pretrained_path: Optional path to pretrained weights
        device: Device to load model on
        
    Returns:
        Initialized model
    """
    model = ProteinMPNNWrapper(
        node_features=config.get('node_features', 128),
        edge_features=config.get('edge_features', 128),
        hidden_dim=config.get('hidden_dim', 128),
        num_encoder_layers=config.get('num_encoder_layers', 3),
        num_decoder_layers=config.get('num_decoder_layers', 3),
        k_neighbors=config.get('k_neighbors', 48),
        dropout=config.get('dropout', 0.1),
        augment_eps=config.get('augment_eps', 0.0),
    )
    
    if pretrained_path:
        model.load_pretrained(pretrained_path)
    
    model = model.to(device)
    
    return model
