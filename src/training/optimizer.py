"""
Optimizer and learning rate scheduler for PKS-MPNN.

Uses the Noam scheduler from the original Transformer paper, which
ProteinMPNN also uses by default.
"""

import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
from typing import Iterator, Optional


class NoamScheduler:
    """
    Noam learning rate scheduler from "Attention is All You Need".
    
    This wraps an optimizer and adjusts the learning rate according to:
    lr = factor * (d_model^-0.5) * min(step^-0.5, step * warmup^-1.5)
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int,
        warmup_steps: int = 4000,
        factor: float = 2.0,
        last_step: int = 0
    ):
        """
        Initialize the scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            d_model: Model hidden dimension
            warmup_steps: Number of warmup steps
            factor: Scaling factor
            last_step: Starting step (for resuming training)
        """
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self._step = last_step
        self._rate = 0
    
    @property
    def param_groups(self):
        """Return optimizer param groups."""
        return self.optimizer.param_groups
    
    def step(self):
        """Update learning rate and optimizer."""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
    
    def rate(self, step: Optional[int] = None) -> float:
        """Compute current learning rate."""
        if step is None:
            step = self._step
        
        return self.factor * (
            self.d_model ** (-0.5) *
            min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
        )
    
    def zero_grad(self):
        """Zero optimizer gradients."""
        self.optimizer.zero_grad()
    
    def state_dict(self):
        """Return scheduler state for checkpointing."""
        return {
            'step': self._step,
            'rate': self._rate,
            'optimizer_state_dict': self.optimizer.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        """Load scheduler state from checkpoint."""
        self._step = state_dict['step']
        self._rate = state_dict['rate']
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])


def get_optimizer(
    parameters: Iterator[torch.nn.Parameter],
    d_model: int,
    optimizer_type: str = "adam",
    lr: float = 0.0,
    betas: tuple = (0.9, 0.98),
    eps: float = 1e-9,
    weight_decay: float = 0.0,
    warmup_steps: int = 4000,
    use_noam: bool = True
) -> NoamScheduler:
    """
    Create optimizer with optional Noam scheduler.
    
    Args:
        parameters: Model parameters
        d_model: Model hidden dimension
        optimizer_type: "adam" or "adamw"
        lr: Base learning rate (0 for Noam-controlled)
        betas: Adam betas
        eps: Adam epsilon
        weight_decay: Weight decay for AdamW
        warmup_steps: Warmup steps for Noam
        use_noam: Whether to use Noam scheduler
        
    Returns:
        NoamScheduler wrapping the optimizer
    """
    if optimizer_type.lower() == "adam":
        optimizer = Adam(
            parameters,
            lr=lr,
            betas=betas,
            eps=eps
        )
    elif optimizer_type.lower() == "adamw":
        optimizer = AdamW(
            parameters,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    if use_noam:
        return NoamScheduler(
            optimizer,
            d_model=d_model,
            warmup_steps=warmup_steps
        )
    else:
        return optimizer


def get_finetune_optimizer(
    model: torch.nn.Module,
    base_lr: float = 1e-4,
    encoder_lr_mult: float = 0.1,
    weight_decay: float = 0.01
) -> torch.optim.Optimizer:
    """
    Create optimizer with different learning rates for encoder/decoder.
    
    For fine-tuning, we often want lower LR on the encoder (pretrained)
    and higher LR on the decoder.
    
    Args:
        model: ProteinMPNN model
        base_lr: Base learning rate for decoder
        encoder_lr_mult: Multiplier for encoder LR
        weight_decay: Weight decay
        
    Returns:
        Configured optimizer
    """
    encoder_params = []
    decoder_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if 'encoder' in name:
            encoder_params.append(param)
        elif 'decoder' in name:
            decoder_params.append(param)
        else:
            other_params.append(param)
    
    param_groups = []
    
    if encoder_params:
        param_groups.append({
            'params': encoder_params,
            'lr': base_lr * encoder_lr_mult,
            'name': 'encoder'
        })
    
    if decoder_params:
        param_groups.append({
            'params': decoder_params,
            'lr': base_lr,
            'name': 'decoder'
        })
    
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': base_lr,
            'name': 'other'
        })
    
    return AdamW(param_groups, weight_decay=weight_decay)
