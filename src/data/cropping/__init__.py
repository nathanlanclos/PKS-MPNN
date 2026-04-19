"""
Cropping strategies for PKS-MPNN experiments.

Three strategies supported:
1. DomainOnlyCropper: Extract individual domains without context
2. FullModuleCropper: Use complete structures with pLDDT masking
3. ContextAwareCropper: Smart cropping preserving K-nearest neighbor relationships
"""

from .domain_only import DomainOnlyCropper
from .full_module import FullModuleCropper
from .context_aware import ContextAwareCropper

__all__ = ["DomainOnlyCropper", "FullModuleCropper", "ContextAwareCropper"]
