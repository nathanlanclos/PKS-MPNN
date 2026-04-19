"""
Full module cropping strategy.

Uses complete PKS module structures (including dimers) with pLDDT-aware
masking. This provides maximum structural context but may include noisy
regions from AlphaFold predictions.

Masking strategy (from Gemini/GPT recommendations):
- pLDDT > 70: Full loss weight (train on these residues)
- 50 < pLDDT <= 70: Include in graph, but mask from loss
- pLDDT <= 50: Exclude from input graph entirely (too unreliable)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..cif_parser import ParsedStructure
from ..annotation_parser import ModuleAnnotation


@dataclass
class FullModuleData:
    """Complete module ready for training."""
    name: str
    sequence: str
    coords: np.ndarray      # Shape: (L, 4, 3)
    plddt: np.ndarray       # Shape: (L,)
    chain_ids: np.ndarray   # Shape: (L,)
    loss_mask: np.ndarray   # Shape: (L,) - which residues to train on
    input_mask: np.ndarray  # Shape: (L,) - which residues to include in graph
    domain_labels: np.ndarray  # Shape: (L,) - domain type index for each residue
    
    @property
    def length(self) -> int:
        return len(self.sequence)
    
    @property
    def effective_length(self) -> int:
        """Number of residues included in input graph."""
        return int(self.input_mask.sum())
    
    @property
    def trainable_length(self) -> int:
        """Number of residues contributing to loss."""
        return int(self.loss_mask.sum())
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format for ProteinMPNN."""
        return {
            'name': self.name,
            'seq': self.sequence,
            'coords': {
                'N': self.coords[:, 0, :].tolist(),
                'CA': self.coords[:, 1, :].tolist(),
                'C': self.coords[:, 2, :].tolist(),
                'O': self.coords[:, 3, :].tolist(),
            },
            'plddt': self.plddt.tolist(),
            'chain_ids': self.chain_ids.tolist(),
            'loss_mask': self.loss_mask.tolist(),
            'input_mask': self.input_mask.tolist(),
            'domain_labels': self.domain_labels.tolist(),
        }
    
    def get_filtered_data(self) -> 'FullModuleData':
        """Return data with low-confidence residues removed."""
        mask = self.input_mask.astype(bool)
        
        return FullModuleData(
            name=self.name,
            sequence=''.join([self.sequence[i] for i in np.where(mask)[0]]),
            coords=self.coords[mask],
            plddt=self.plddt[mask],
            chain_ids=self.chain_ids[mask],
            loss_mask=self.loss_mask[mask],
            input_mask=np.ones(mask.sum(), dtype=np.float32),
            domain_labels=self.domain_labels[mask]
        )


class FullModuleCropper:
    """
    Process complete PKS modules with pLDDT-aware masking.
    
    This strategy uses the full structure but applies tiered masking
    based on AlphaFold confidence scores:
    
    - High confidence (pLDDT > 70): Train on these residues
    - Medium confidence (50-70): Include in structure but don't train
    - Low confidence (< 50): Exclude from structure entirely
    
    This prevents the model from learning incorrect sequence-structure
    relationships from unreliable regions.
    """
    
    def __init__(
        self,
        high_confidence_threshold: float = 70.0,
        low_confidence_threshold: float = 50.0,
        min_trainable_residues: int = 100,
        exclude_low_confidence: bool = True
    ):
        """
        Initialize the cropper.
        
        Args:
            high_confidence_threshold: pLDDT threshold for loss contribution
            low_confidence_threshold: pLDDT threshold for graph inclusion
            min_trainable_residues: Minimum residues with pLDDT > high threshold
            exclude_low_confidence: Whether to exclude very low confidence regions
        """
        self.high_confidence_threshold = high_confidence_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self.min_trainable_residues = min_trainable_residues
        self.exclude_low_confidence = exclude_low_confidence
    
    def process(
        self,
        structure: ParsedStructure,
        annotation: Optional[ModuleAnnotation] = None
    ) -> Optional[FullModuleData]:
        """
        Process a full module structure.
        
        Args:
            structure: Parsed structure with coordinates and pLDDT
            annotation: Optional domain annotations for labeling
            
        Returns:
            FullModuleData or None if insufficient high-confidence residues
        """
        plddt = structure.plddt
        
        # Create masks
        loss_mask = (plddt >= self.high_confidence_threshold).astype(np.float32)
        
        if self.exclude_low_confidence:
            input_mask = (plddt >= self.low_confidence_threshold).astype(np.float32)
        else:
            input_mask = np.ones(structure.length, dtype=np.float32)
        
        # Check if enough trainable residues
        if loss_mask.sum() < self.min_trainable_residues:
            return None
        
        # Create domain labels if annotation provided
        domain_labels = self._create_domain_labels(structure.length, annotation)
        
        return FullModuleData(
            name=structure.name,
            sequence=structure.sequence,
            coords=structure.coords,
            plddt=plddt,
            chain_ids=structure.chain_ids,
            loss_mask=loss_mask,
            input_mask=input_mask,
            domain_labels=domain_labels
        )
    
    def _create_domain_labels(
        self,
        length: int,
        annotation: Optional[ModuleAnnotation]
    ) -> np.ndarray:
        """Create domain type labels for each residue."""
        
        # Default label indices
        DOMAIN_TO_IDX = {
            'KS': 1, 'AT': 2, 'DH': 3, 'ER': 4, 'KR': 5, 'ACP': 6,
            'oMT': 7, 'C': 8, 'A': 9, 'PCP': 10, 'E': 11,
            'linker': 12, 'unknown': 0
        }
        
        labels = np.zeros(length, dtype=np.int32)  # 0 = unknown
        
        if annotation is None:
            return labels
        
        for domain in annotation.domains:
            indices = domain.get_residue_indices(zero_indexed=True)
            indices = indices[indices < length]  # Clip to structure length
            
            if domain.is_linker:
                labels[indices] = DOMAIN_TO_IDX['linker']
            elif domain.domain_type in DOMAIN_TO_IDX:
                labels[indices] = DOMAIN_TO_IDX[domain.domain_type]
        
        return labels
    
    def process_batch(
        self,
        structures: List[ParsedStructure],
        annotations: Optional[Dict[str, ModuleAnnotation]] = None
    ) -> List[FullModuleData]:
        """
        Process multiple structures.
        
        Args:
            structures: List of parsed structures
            annotations: Optional dict mapping structure names to annotations
            
        Returns:
            List of processed modules
        """
        processed = []
        
        for structure in structures:
            annotation = None
            if annotations:
                annotation = annotations.get(structure.name)
            
            result = self.process(structure, annotation)
            if result is not None:
                processed.append(result)
        
        return processed


def compute_plddt_statistics(data: FullModuleData) -> Dict:
    """Compute pLDDT statistics for a module."""
    return {
        'mean_plddt': float(data.plddt.mean()),
        'median_plddt': float(np.median(data.plddt)),
        'min_plddt': float(data.plddt.min()),
        'max_plddt': float(data.plddt.max()),
        'fraction_high_conf': float((data.plddt >= 70).mean()),
        'fraction_medium_conf': float(((data.plddt >= 50) & (data.plddt < 70)).mean()),
        'fraction_low_conf': float((data.plddt < 50).mean()),
        'total_residues': data.length,
        'trainable_residues': data.trainable_length,
        'input_residues': data.effective_length,
    }


def compute_domain_plddt_statistics(
    data: FullModuleData,
    annotation: ModuleAnnotation
) -> Dict[str, Dict]:
    """Compute pLDDT statistics per domain type."""
    stats = {}
    
    for domain in annotation.domains:
        indices = domain.get_residue_indices(zero_indexed=True)
        indices = indices[indices < data.length]
        
        if len(indices) == 0:
            continue
        
        domain_plddt = data.plddt[indices]
        
        stats[domain.domain_type] = {
            'mean_plddt': float(domain_plddt.mean()),
            'min_plddt': float(domain_plddt.min()),
            'max_plddt': float(domain_plddt.max()),
            'fraction_high_conf': float((domain_plddt >= 70).mean()),
            'length': len(indices),
            'is_linker': domain.is_linker,
        }
    
    return stats
