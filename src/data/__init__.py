"""
Data processing modules for PKS-MPNN.

Includes:
- CIF file parsing with pLDDT extraction
- Domain annotation parsing from CSV
- Cropping strategies (domain-only, full module, context-aware)
- Dataset classes for training
- Clustering and splitting utilities
"""

from .cif_parser import CIFParser
from .annotation_parser import AnnotationParser
from .dataset import PKSDataset, PKSBatchSampler
from .clustering import SequenceClusterer
from .splits import DatasetSplitter
