"""
Data processing modules for PKS-MPNN.

Includes:
- CIF file parsing with pLDDT extraction
- Domain annotation parsing from CSV
- Cropping strategies (domain-only, full module, context-aware)
- Dataset classes for training
- Clustering and splitting utilities
"""

from .cif_parser import CIFParser, list_structure_files
from .annotation_parser import AnnotationParser
from .clustering import SequenceClusterer
from .splits import DatasetSplitter


def __getattr__(name: str):
    """Load torch-backed dataset classes only when accessed (avoids importing torch for annotation-only scripts)."""
    if name == "PKSDataset":
        from .dataset import PKSDataset
        return PKSDataset
    if name == "PKSBatchSampler":
        from .dataset import PKSBatchSampler
        return PKSBatchSampler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CIFParser",
    "list_structure_files",
    "AnnotationParser",
    "PKSDataset",
    "PKSBatchSampler",
    "SequenceClusterer",
    "DatasetSplitter",
]
