"""
Model components for PKS-MPNN.

Wraps ProteinMPNN with:
- Weight unfreezing strategies for fine-tuning
- pLDDT-aware loss computation
- PKS-specific metrics
"""

from .protein_mpnn import ProteinMPNNWrapper
from .loss import PLDDTWeightedLoss, compute_loss_masks
from .metrics import compute_perplexity, compute_recovery, PerDomainMetrics
