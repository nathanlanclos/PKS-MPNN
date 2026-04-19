"""
Context-aware geometric cropping strategy.

Smart cropping that preserves ProteinMPNN's K=48 nearest neighbor relationships
while maximizing geometric diversity. This strategy:

1. Selects design residues based on domain type and pLDDT confidence
2. Builds a KNN graph (K=48 matching ProteinMPNN)
3. Expands selection to include all K-neighbors of design residues
4. Computes "neighborhood diversity" to prioritize informative context
5. Filters out low-confidence regions

The key insight is that ProteinMPNN predicts sequences based on local 3D
neighborhoods. By preserving these neighborhoods, we maintain the information
the model needs while avoiding noisy long-range context.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from scipy.spatial import cKDTree

from ..cif_parser import ParsedStructure, compute_ca_distances, build_knn_graph
from ..annotation_parser import ModuleAnnotation, CORE_DOMAINS


@dataclass
class ContextAwareCrop:
    """A context-aware cropped region ready for training."""
    name: str
    sequence: str
    coords: np.ndarray          # Shape: (L, 4, 3)
    plddt: np.ndarray           # Shape: (L,)
    chain_ids: np.ndarray       # Shape: (L,)
    loss_mask: np.ndarray       # Shape: (L,) - design residues
    context_mask: np.ndarray    # Shape: (L,) - context residues (not in loss)
    domain_labels: np.ndarray   # Shape: (L,)
    diversity_scores: np.ndarray  # Shape: (L,) - neighborhood diversity
    original_indices: np.ndarray  # Indices in original structure
    
    @property
    def length(self) -> int:
        return len(self.sequence)
    
    @property
    def design_length(self) -> int:
        """Number of residues in design region (contributing to loss)."""
        return int(self.loss_mask.sum())
    
    @property
    def context_length(self) -> int:
        """Number of context residues (not contributing to loss)."""
        return int(self.context_mask.sum())
    
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
            'context_mask': self.context_mask.tolist(),
            'domain_labels': self.domain_labels.tolist(),
            'diversity_scores': self.diversity_scores.tolist(),
        }


class ContextAwareCropper:
    """
    Smart cropping that preserves K-nearest neighbor relationships.
    
    This cropper ensures that design residues have access to their full
    local neighborhoods (K=48 by default, matching ProteinMPNN). It also
    prioritizes context residues with high "geometric diversity" - residues
    that see multiple domain types or structural contexts.
    
    Algorithm:
    1. Identify design residues (high-confidence core domains)
    2. Build KNN graph with K neighbors
    3. For each design residue, include all K neighbors as context
    4. Filter context by pLDDT (exclude very low confidence)
    5. Score context residues by neighborhood diversity
    6. Optionally prune low-diversity context to reduce size
    """
    
    def __init__(
        self,
        k_neighbors: int = 48,
        plddt_design_threshold: float = 70.0,
        plddt_context_threshold: float = 50.0,
        design_domains: Optional[List[str]] = None,
        include_linkers_as_context: bool = True,
        max_context_expansion: float = 2.0,
        diversity_threshold: float = 0.0
    ):
        """
        Initialize the cropper.
        
        Args:
            k_neighbors: Number of neighbors (matches ProteinMPNN default)
            plddt_design_threshold: Min pLDDT for design residues
            plddt_context_threshold: Min pLDDT for context residues
            design_domains: Domains to use as design regions. If None, all core domains.
            include_linkers_as_context: Include linker regions as context
            max_context_expansion: Max ratio of context to design residues
            diversity_threshold: Min diversity score for context inclusion
        """
        self.k_neighbors = k_neighbors
        self.plddt_design_threshold = plddt_design_threshold
        self.plddt_context_threshold = plddt_context_threshold
        self.design_domains = set(design_domains) if design_domains else CORE_DOMAINS
        self.include_linkers_as_context = include_linkers_as_context
        self.max_context_expansion = max_context_expansion
        self.diversity_threshold = diversity_threshold
    
    def crop(
        self,
        structure: ParsedStructure,
        annotation: ModuleAnnotation
    ) -> Optional[ContextAwareCrop]:
        """
        Create a context-aware crop from a structure.
        
        Args:
            structure: Parsed structure with coordinates and pLDDT
            annotation: Domain annotations
            
        Returns:
            ContextAwareCrop or None if insufficient design residues
        """
        # Step 1: Identify design residues
        design_mask = self._get_design_mask(structure, annotation)
        
        if design_mask.sum() < 20:  # Too few design residues
            return None
        
        # Step 2: Build KNN graph
        knn_indices = build_knn_graph(structure.coords, k=self.k_neighbors)
        
        # Step 3: Expand to include K-neighbors of design residues
        context_mask = self._expand_to_neighbors(design_mask, knn_indices)
        
        # Step 4: Filter by pLDDT
        context_mask = context_mask & (structure.plddt >= self.plddt_context_threshold)
        
        # Ensure design residues are always included
        context_mask = context_mask | design_mask
        
        # Step 5: Compute diversity scores
        diversity_scores = self._compute_diversity_scores(
            structure, annotation, context_mask, knn_indices
        )
        
        # Step 6: Apply diversity threshold
        if self.diversity_threshold > 0:
            # Keep design residues regardless of diversity
            diversity_mask = (diversity_scores >= self.diversity_threshold) | design_mask
            context_mask = context_mask & diversity_mask
        
        # Step 7: Apply max context expansion limit
        context_mask = self._limit_context_size(
            design_mask, context_mask, diversity_scores
        )
        
        # Extract the cropped region
        return self._extract_crop(
            structure, annotation, design_mask, context_mask, diversity_scores
        )
    
    def _get_design_mask(
        self,
        structure: ParsedStructure,
        annotation: ModuleAnnotation
    ) -> np.ndarray:
        """Get mask of design residues (high-confidence core domains)."""
        
        # Domain-based mask
        domain_mask = annotation.get_domain_mask(
            include_domains=list(self.design_domains),
            exclude_linkers=True
        )
        
        # Ensure mask matches structure length
        if len(domain_mask) != structure.length:
            # Annotation might be for monomer, structure might be dimer
            if len(domain_mask) * 2 == structure.length:
                # Dimer: duplicate mask for both chains
                domain_mask = np.tile(domain_mask, 2)
            else:
                # Length mismatch - use structure length
                new_mask = np.zeros(structure.length, dtype=bool)
                new_mask[:min(len(domain_mask), structure.length)] = \
                    domain_mask[:min(len(domain_mask), structure.length)]
                domain_mask = new_mask
        
        # pLDDT-based mask
        plddt_mask = structure.plddt >= self.plddt_design_threshold
        
        # Combine: must be both in design domain AND high confidence
        return domain_mask & plddt_mask
    
    def _expand_to_neighbors(
        self,
        design_mask: np.ndarray,
        knn_indices: np.ndarray
    ) -> np.ndarray:
        """Expand selection to include K-neighbors of design residues."""
        
        expanded_mask = design_mask.copy()
        design_indices = np.where(design_mask)[0]
        
        # Add all neighbors of design residues
        for idx in design_indices:
            neighbors = knn_indices[idx]
            expanded_mask[neighbors] = True
        
        return expanded_mask
    
    def _compute_diversity_scores(
        self,
        structure: ParsedStructure,
        annotation: ModuleAnnotation,
        mask: np.ndarray,
        knn_indices: np.ndarray
    ) -> np.ndarray:
        """
        Compute neighborhood diversity score for each residue.
        
        High diversity means the residue's neighborhood spans multiple
        domain types or has high geometric variance - these are informative
        residues, typically at domain interfaces.
        """
        n_residues = structure.length
        diversity = np.zeros(n_residues, dtype=np.float32)
        
        # Create domain label array
        domain_labels = self._get_domain_labels(n_residues, annotation)
        
        for i in range(n_residues):
            if not mask[i]:
                continue
            
            neighbors = knn_indices[i]
            
            # Domain diversity: number of unique domains in neighborhood
            neighbor_domains = domain_labels[neighbors]
            unique_domains = len(np.unique(neighbor_domains[neighbor_domains > 0]))
            domain_diversity = unique_domains / max(len(CORE_DOMAINS), 1)
            
            # Geometric diversity: variance of neighbor positions
            neighbor_coords = structure.coords[neighbors, 1, :]  # CA atoms
            center = neighbor_coords.mean(axis=0)
            distances = np.linalg.norm(neighbor_coords - center, axis=1)
            geometric_diversity = distances.std() / (distances.mean() + 1e-6)
            
            # pLDDT diversity: variance in confidence
            neighbor_plddt = structure.plddt[neighbors]
            plddt_diversity = neighbor_plddt.std() / 100.0
            
            # Combined score (weighted average)
            diversity[i] = (
                0.5 * domain_diversity +
                0.3 * geometric_diversity +
                0.2 * plddt_diversity
            )
        
        return diversity
    
    def _get_domain_labels(
        self,
        length: int,
        annotation: ModuleAnnotation
    ) -> np.ndarray:
        """Get domain type label for each residue."""
        DOMAIN_TO_IDX = {
            'KS': 1, 'AT': 2, 'DH': 3, 'ER': 4, 'KR': 5, 'ACP': 6,
            'oMT': 7, 'C': 8, 'A': 9, 'PCP': 10, 'E': 11,
        }
        
        labels = np.zeros(length, dtype=np.int32)
        
        for domain in annotation.domains:
            if domain.is_linker:
                continue
            
            indices = domain.get_residue_indices(zero_indexed=True)
            indices = indices[indices < length]
            
            if domain.domain_type in DOMAIN_TO_IDX:
                labels[indices] = DOMAIN_TO_IDX[domain.domain_type]
        
        # Handle dimer case
        if len(annotation.fragment_sequence) < length:
            # Assume symmetric dimer
            monomer_len = len(annotation.fragment_sequence)
            if monomer_len * 2 <= length:
                labels[monomer_len:monomer_len*2] = labels[:monomer_len]
        
        return labels
    
    def _limit_context_size(
        self,
        design_mask: np.ndarray,
        context_mask: np.ndarray,
        diversity_scores: np.ndarray
    ) -> np.ndarray:
        """Limit context size to max_context_expansion * design size."""
        
        n_design = design_mask.sum()
        max_context = int(n_design * self.max_context_expansion)
        
        # Get context-only indices (not design)
        context_only = context_mask & ~design_mask
        context_indices = np.where(context_only)[0]
        
        if len(context_indices) <= max_context:
            return context_mask
        
        # Keep top context residues by diversity score
        context_diversity = diversity_scores[context_indices]
        top_indices = context_indices[np.argsort(context_diversity)[-max_context:]]
        
        # Create new mask
        new_mask = design_mask.copy()
        new_mask[top_indices] = True
        
        return new_mask
    
    def _extract_crop(
        self,
        structure: ParsedStructure,
        annotation: ModuleAnnotation,
        design_mask: np.ndarray,
        context_mask: np.ndarray,
        diversity_scores: np.ndarray
    ) -> ContextAwareCrop:
        """Extract the final cropped region."""
        
        # Get indices to keep
        keep_indices = np.where(context_mask)[0]
        
        # Extract data
        sequence = ''.join([structure.sequence[i] for i in keep_indices])
        coords = structure.coords[keep_indices]
        plddt = structure.plddt[keep_indices]
        chain_ids = structure.chain_ids[keep_indices]
        
        # Map masks to cropped indices
        loss_mask = design_mask[keep_indices].astype(np.float32)
        context_only = (context_mask & ~design_mask)[keep_indices].astype(np.float32)
        
        # Domain labels for cropped region
        domain_labels = self._get_domain_labels(structure.length, annotation)
        domain_labels = domain_labels[keep_indices]
        
        # Diversity scores for cropped region
        cropped_diversity = diversity_scores[keep_indices]
        
        return ContextAwareCrop(
            name=f"{structure.name}_context",
            sequence=sequence,
            coords=coords,
            plddt=plddt,
            chain_ids=chain_ids,
            loss_mask=loss_mask,
            context_mask=context_only,
            domain_labels=domain_labels,
            diversity_scores=cropped_diversity,
            original_indices=keep_indices
        )
    
    def crop_batch(
        self,
        structures: List[ParsedStructure],
        annotations: Dict[str, ModuleAnnotation]
    ) -> List[ContextAwareCrop]:
        """
        Crop multiple structures.
        
        Args:
            structures: List of parsed structures
            annotations: Dict mapping structure names to annotations
            
        Returns:
            List of context-aware crops
        """
        crops = []
        
        for structure in structures:
            annotation = annotations.get(structure.name)
            if annotation is None:
                continue
            
            crop = self.crop(structure, annotation)
            if crop is not None:
                crops.append(crop)
        
        return crops


def analyze_crop_statistics(crops: List[ContextAwareCrop]) -> Dict:
    """Analyze statistics of context-aware crops."""
    
    stats = {
        'n_crops': len(crops),
        'total_residues': [],
        'design_residues': [],
        'context_residues': [],
        'mean_diversity': [],
        'mean_plddt_design': [],
        'mean_plddt_context': [],
    }
    
    for crop in crops:
        stats['total_residues'].append(crop.length)
        stats['design_residues'].append(crop.design_length)
        stats['context_residues'].append(crop.context_length)
        stats['mean_diversity'].append(crop.diversity_scores.mean())
        
        if crop.design_length > 0:
            design_plddt = crop.plddt[crop.loss_mask.astype(bool)]
            stats['mean_plddt_design'].append(design_plddt.mean())
        
        if crop.context_length > 0:
            context_plddt = crop.plddt[crop.context_mask.astype(bool)]
            stats['mean_plddt_context'].append(context_plddt.mean())
    
    return {
        'n_crops': stats['n_crops'],
        'mean_total_residues': np.mean(stats['total_residues']),
        'mean_design_residues': np.mean(stats['design_residues']),
        'mean_context_residues': np.mean(stats['context_residues']),
        'mean_diversity': np.mean(stats['mean_diversity']),
        'mean_plddt_design': np.mean(stats['mean_plddt_design']),
        'mean_plddt_context': np.mean(stats['mean_plddt_context']),
        'context_to_design_ratio': np.mean(stats['context_residues']) / max(np.mean(stats['design_residues']), 1),
    }
