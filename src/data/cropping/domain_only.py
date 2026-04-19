"""
Domain-only cropping strategy.

Extracts individual PKS domains (KS, AT, DH, ER, KR, ACP, etc.) without
surrounding context. This tests whether local fold is sufficient for
sequence prediction.

Important: Linker regions (domains ending in 'L') are EXCLUDED from
domain-only training. They are only meaningful in context.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..cif_parser import ParsedStructure
from ..annotation_parser import ModuleAnnotation, DomainAnnotation, CORE_DOMAINS


@dataclass
class CroppedDomain:
    """A single cropped domain ready for training."""
    name: str
    domain_type: str
    sequence: str
    coords: np.ndarray  # Shape: (L, 4, 3)
    plddt: np.ndarray   # Shape: (L,)
    chain_ids: np.ndarray
    loss_mask: np.ndarray  # Shape: (L,) - which residues to train on
    original_indices: np.ndarray  # Indices in original structure
    
    @property
    def length(self) -> int:
        return len(self.sequence)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format for ProteinMPNN."""
        return {
            'name': self.name,
            'domain_type': self.domain_type,
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
        }


class DomainOnlyCropper:
    """
    Crop individual domains from PKS modules.
    
    This strategy extracts each core domain (KS, AT, DH, ER, KR, ACP, etc.)
    separately, without any context from neighboring domains. This is useful
    for understanding domain-specific sequence patterns.
    
    Linker regions (KSATL, ATDHL, etc.) are excluded as they are only
    meaningful in context of their flanking domains.
    """
    
    def __init__(
        self,
        plddt_threshold: float = 70.0,
        min_domain_length: int = 20,
        include_domains: Optional[List[str]] = None
    ):
        """
        Initialize the cropper.
        
        Args:
            plddt_threshold: Minimum pLDDT for residues to contribute to loss
            min_domain_length: Minimum domain length to include
            include_domains: Specific domains to include. If None, use all core domains.
        """
        self.plddt_threshold = plddt_threshold
        self.min_domain_length = min_domain_length
        self.include_domains = set(include_domains) if include_domains else CORE_DOMAINS
    
    def crop(
        self,
        structure: ParsedStructure,
        annotation: ModuleAnnotation
    ) -> List[CroppedDomain]:
        """
        Extract all individual domains from a structure.
        
        Args:
            structure: Parsed structure with coordinates and pLDDT
            annotation: Domain annotations for this structure
            
        Returns:
            List of CroppedDomain objects, one per core domain
        """
        cropped_domains = []
        
        for domain in annotation.domains:
            # Skip linkers and non-included domains
            if domain.is_linker:
                continue
            if domain.domain_type not in self.include_domains:
                continue
            if domain.length < self.min_domain_length:
                continue
            
            cropped = self._crop_single_domain(structure, annotation, domain)
            if cropped is not None:
                cropped_domains.append(cropped)
        
        return cropped_domains
    
    def crop_specific_domain(
        self,
        structure: ParsedStructure,
        annotation: ModuleAnnotation,
        domain_type: str
    ) -> Optional[CroppedDomain]:
        """
        Extract a specific domain type from a structure.
        
        Args:
            structure: Parsed structure
            annotation: Domain annotations
            domain_type: Domain type to extract (e.g., 'KS', 'AT')
            
        Returns:
            CroppedDomain or None if domain not found
        """
        domain = annotation.get_domain(domain_type)
        if domain is None:
            return None
        
        return self._crop_single_domain(structure, annotation, domain)
    
    def _crop_single_domain(
        self,
        structure: ParsedStructure,
        annotation: ModuleAnnotation,
        domain: DomainAnnotation
    ) -> Optional[CroppedDomain]:
        """Crop a single domain from the structure."""
        
        # Get residue indices for this domain (0-indexed)
        indices = domain.get_residue_indices(zero_indexed=True)
        
        # Validate indices are within structure bounds
        if indices.max() >= structure.length:
            print(f"Warning: Domain {domain.domain_type} indices exceed structure length")
            indices = indices[indices < structure.length]
        
        if len(indices) < self.min_domain_length:
            return None
        
        # Extract data for this domain
        sequence = ''.join([structure.sequence[i] for i in indices])
        coords = structure.coords[indices]
        plddt = structure.plddt[indices]
        chain_ids = structure.chain_ids[indices]
        
        # Create loss mask based on pLDDT
        loss_mask = (plddt >= self.plddt_threshold).astype(np.float32)
        
        # Don't train on domains with too few high-confidence residues
        if loss_mask.sum() < self.min_domain_length * 0.5:
            return None
        
        return CroppedDomain(
            name=f"{structure.name}_{domain.domain_type}",
            domain_type=domain.domain_type,
            sequence=sequence,
            coords=coords,
            plddt=plddt,
            chain_ids=chain_ids,
            loss_mask=loss_mask,
            original_indices=indices
        )
    
    def crop_batch(
        self,
        structures: List[ParsedStructure],
        annotations: Dict[str, ModuleAnnotation],
        domain_type: Optional[str] = None
    ) -> List[CroppedDomain]:
        """
        Crop domains from multiple structures.
        
        Args:
            structures: List of parsed structures
            annotations: Dict mapping structure names to annotations
            domain_type: If specified, only extract this domain type
            
        Returns:
            List of all cropped domains
        """
        all_cropped = []
        
        for structure in structures:
            # Find matching annotation
            annotation = annotations.get(structure.name)
            if annotation is None:
                continue
            
            if domain_type:
                cropped = self.crop_specific_domain(structure, annotation, domain_type)
                if cropped:
                    all_cropped.append(cropped)
            else:
                cropped = self.crop(structure, annotation)
                all_cropped.extend(cropped)
        
        return all_cropped


def get_domain_statistics(
    cropped_domains: List[CroppedDomain]
) -> Dict[str, Dict]:
    """
    Compute statistics for cropped domains.
    
    Args:
        cropped_domains: List of cropped domains
        
    Returns:
        Dict with statistics per domain type
    """
    from collections import defaultdict
    
    stats = defaultdict(lambda: {
        'count': 0,
        'lengths': [],
        'mean_plddt': [],
        'loss_fraction': []
    })
    
    for domain in cropped_domains:
        dtype = domain.domain_type
        stats[dtype]['count'] += 1
        stats[dtype]['lengths'].append(domain.length)
        stats[dtype]['mean_plddt'].append(domain.plddt.mean())
        stats[dtype]['loss_fraction'].append(domain.loss_mask.mean())
    
    # Compute summary statistics
    summary = {}
    for dtype, data in stats.items():
        summary[dtype] = {
            'count': data['count'],
            'mean_length': np.mean(data['lengths']),
            'std_length': np.std(data['lengths']),
            'mean_plddt': np.mean(data['mean_plddt']),
            'mean_loss_fraction': np.mean(data['loss_fraction'])
        }
    
    return summary
