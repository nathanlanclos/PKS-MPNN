"""
PyTorch Dataset classes for PKS-MPNN training.

Supports three experiment types:
1. DomainOnlyDataset: Individual domains without context
2. FullModuleDataset: Complete structures with pLDDT masking
3. ContextAwareDataset: Smart crops preserving K-neighbors
"""

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import random

from .cif_parser import CIFParser, ParsedStructure
from .annotation_parser import AnnotationParser, ModuleAnnotation
from .cropping.domain_only import DomainOnlyCropper, CroppedDomain
from .cropping.full_module import FullModuleCropper, FullModuleData
from .cropping.context_aware import ContextAwareCropper, ContextAwareCrop


# Amino acid encoding
AA_ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
AA_TO_IDX = {aa: idx for idx, aa in enumerate(AA_ALPHABET)}


def encode_sequence(sequence: str) -> torch.Tensor:
    """Convert amino acid sequence to tensor of indices."""
    return torch.tensor([AA_TO_IDX.get(aa, AA_TO_IDX['X']) for aa in sequence], dtype=torch.long)


def featurize_structure(
    coords: np.ndarray,
    sequence: str,
    chain_ids: np.ndarray,
    loss_mask: np.ndarray,
    plddt: Optional[np.ndarray] = None
) -> Dict[str, torch.Tensor]:
    """
    Convert structure data to ProteinMPNN-compatible format.
    
    Args:
        coords: Shape (L, 4, 3) backbone coordinates
        sequence: Amino acid sequence
        chain_ids: Shape (L,) chain assignment
        loss_mask: Shape (L,) which residues to train on
        plddt: Optional shape (L,) confidence scores
        
    Returns:
        Dict with tensors for model input
    """
    L = len(sequence)
    
    # Coordinates
    X = torch.tensor(coords, dtype=torch.float32)  # (L, 4, 3)
    
    # Sequence
    S = encode_sequence(sequence)  # (L,)
    
    # Chain encoding
    chain_M = torch.ones(L, dtype=torch.float32)  # All residues designable
    chain_encoding = torch.tensor(chain_ids, dtype=torch.long)  # (L,)
    
    # Residue indices (position in chain)
    residue_idx = torch.arange(L, dtype=torch.long)
    
    # Mask for loss computation
    mask_for_loss = torch.tensor(loss_mask, dtype=torch.float32)
    
    # General mask (which residues are present)
    mask = torch.ones(L, dtype=torch.float32)
    
    result = {
        'X': X,
        'S': S,
        'mask': mask,
        'chain_M': chain_M,
        'chain_encoding': chain_encoding,
        'residue_idx': residue_idx,
        'mask_for_loss': mask_for_loss,
    }
    
    if plddt is not None:
        result['plddt'] = torch.tensor(plddt, dtype=torch.float32)
    
    return result


class PKSDataset(Dataset):
    """
    Base dataset class for PKS-MPNN experiments.
    
    Handles loading structures, annotations, and applying cropping strategies.
    """
    
    def __init__(
        self,
        cif_dir: Path,
        annotation_csv: Path,
        split_ids: Optional[List[str]] = None,
        cropping_strategy: str = "full_module",
        cropper_kwargs: Optional[Dict] = None,
        cache_structures: bool = False,
        transform: Optional[callable] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            cif_dir: Directory containing CIF files
            annotation_csv: Path to annotations CSV
            split_ids: List of fragment IDs for this split (train/val/test)
            cropping_strategy: One of "domain_only", "full_module", "context_aware"
            cropper_kwargs: Arguments for the cropper
            cache_structures: Whether to cache parsed structures
            transform: Optional transform function
        """
        self.cif_dir = Path(cif_dir)
        self.annotation_parser = AnnotationParser(annotation_csv)
        self.split_ids = split_ids
        self.cropping_strategy = cropping_strategy
        self.cache_structures = cache_structures
        self.transform = transform
        
        # Initialize parser
        self.cif_parser = CIFParser()
        
        # Initialize cropper
        cropper_kwargs = cropper_kwargs or {}
        if cropping_strategy == "domain_only":
            self.cropper = DomainOnlyCropper(**cropper_kwargs)
        elif cropping_strategy == "full_module":
            self.cropper = FullModuleCropper(**cropper_kwargs)
        elif cropping_strategy == "context_aware":
            self.cropper = ContextAwareCropper(**cropper_kwargs)
        else:
            raise ValueError(f"Unknown cropping strategy: {cropping_strategy}")
        
        # Build index of available CIF files
        self._build_index()
        
        # Cache for parsed structures
        self._structure_cache: Dict[str, ParsedStructure] = {}
    
    def _build_index(self):
        """Build index of CIF files and their annotations."""
        self.samples = []
        
        # Find all CIF files
        cif_files = list(self.cif_dir.glob("*.cif"))
        
        for cif_path in cif_files:
            cif_name = cif_path.stem
            
            # Try to find matching annotation
            fragment_id = self._match_annotation(cif_name)
            if fragment_id is None:
                continue
            
            # Check if in split
            if self.split_ids is not None and fragment_id not in self.split_ids:
                continue
            
            self.samples.append({
                'cif_path': cif_path,
                'cif_name': cif_name,
                'fragment_id': fragment_id,
            })
        
        print(f"Found {len(self.samples)} samples for {self.cropping_strategy}")
    
    def _match_annotation(self, cif_name: str) -> Optional[str]:
        """Match CIF filename to annotation fragment_id."""
        import re
        
        # Direct match
        if cif_name in self.annotation_parser:
            return cif_name
        
        # Remove model suffix
        base_name = re.sub(r'_model_\d+$', '', cif_name)
        if base_name in self.annotation_parser:
            return base_name
        
        # Remove rank suffix
        base_name = re.sub(r'_rank_\d+$', '', cif_name)
        if base_name in self.annotation_parser:
            return base_name
        
        return None
    
    def _load_structure(self, sample: Dict) -> ParsedStructure:
        """Load and optionally cache a structure."""
        cif_path = sample['cif_path']
        
        if self.cache_structures and str(cif_path) in self._structure_cache:
            return self._structure_cache[str(cif_path)]
        
        structure = self.cif_parser.parse(cif_path)
        
        if self.cache_structures:
            self._structure_cache[str(cif_path)] = structure
        
        return structure
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load structure
        structure = self._load_structure(sample)
        
        # Get annotation
        annotation = self.annotation_parser[sample['fragment_id']]
        
        # Apply cropping strategy
        if self.cropping_strategy == "domain_only":
            # For domain_only, we get multiple domains per structure
            # Return a random domain
            domains = self.cropper.crop(structure, annotation)
            if not domains:
                # Fallback: return first domain if available
                return self._get_fallback_item(idx)
            cropped = random.choice(domains)
            
            features = featurize_structure(
                cropped.coords,
                cropped.sequence,
                cropped.chain_ids,
                cropped.loss_mask,
                cropped.plddt
            )
            features['name'] = cropped.name
            features['domain_type'] = cropped.domain_type
            
        elif self.cropping_strategy == "full_module":
            processed = self.cropper.process(structure, annotation)
            if processed is None:
                return self._get_fallback_item(idx)
            
            features = featurize_structure(
                processed.coords,
                processed.sequence,
                processed.chain_ids,
                processed.loss_mask,
                processed.plddt
            )
            features['name'] = processed.name
            features['input_mask'] = torch.tensor(processed.input_mask, dtype=torch.float32)
            features['domain_labels'] = torch.tensor(processed.domain_labels, dtype=torch.long)
            
        elif self.cropping_strategy == "context_aware":
            cropped = self.cropper.crop(structure, annotation)
            if cropped is None:
                return self._get_fallback_item(idx)
            
            features = featurize_structure(
                cropped.coords,
                cropped.sequence,
                cropped.chain_ids,
                cropped.loss_mask,
                cropped.plddt
            )
            features['name'] = cropped.name
            features['context_mask'] = torch.tensor(cropped.context_mask, dtype=torch.float32)
            features['diversity_scores'] = torch.tensor(cropped.diversity_scores, dtype=torch.float32)
        
        if self.transform:
            features = self.transform(features)
        
        return features
    
    def _get_fallback_item(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get fallback item if cropping fails."""
        # Try next item
        next_idx = (idx + 1) % len(self)
        if next_idx == idx:
            raise RuntimeError("All samples failed cropping")
        return self.__getitem__(next_idx)


class DomainOnlyDataset(PKSDataset):
    """Dataset for domain-only training."""
    
    def __init__(
        self,
        cif_dir: Path,
        annotation_csv: Path,
        domain_type: Optional[str] = None,
        split_ids: Optional[List[str]] = None,
        plddt_threshold: float = 70.0,
        **kwargs
    ):
        """
        Initialize domain-only dataset.
        
        Args:
            cif_dir: Directory containing CIF files
            annotation_csv: Path to annotations CSV
            domain_type: Specific domain type to train on (e.g., "KS", "AT")
            split_ids: List of fragment IDs for this split
            plddt_threshold: Minimum pLDDT for loss
        """
        include_domains = [domain_type] if domain_type else None
        
        super().__init__(
            cif_dir=cif_dir,
            annotation_csv=annotation_csv,
            split_ids=split_ids,
            cropping_strategy="domain_only",
            cropper_kwargs={
                'plddt_threshold': plddt_threshold,
                'include_domains': include_domains,
            },
            **kwargs
        )
        
        self.domain_type = domain_type


class FullModuleDataset(PKSDataset):
    """Dataset for full module training."""
    
    def __init__(
        self,
        cif_dir: Path,
        annotation_csv: Path,
        split_ids: Optional[List[str]] = None,
        high_confidence_threshold: float = 70.0,
        low_confidence_threshold: float = 50.0,
        **kwargs
    ):
        """
        Initialize full module dataset.
        
        Args:
            cif_dir: Directory containing CIF files
            annotation_csv: Path to annotations CSV
            split_ids: List of fragment IDs for this split
            high_confidence_threshold: pLDDT for loss
            low_confidence_threshold: pLDDT for graph inclusion
        """
        super().__init__(
            cif_dir=cif_dir,
            annotation_csv=annotation_csv,
            split_ids=split_ids,
            cropping_strategy="full_module",
            cropper_kwargs={
                'high_confidence_threshold': high_confidence_threshold,
                'low_confidence_threshold': low_confidence_threshold,
            },
            **kwargs
        )


class ContextAwareDataset(PKSDataset):
    """Dataset for context-aware training."""
    
    def __init__(
        self,
        cif_dir: Path,
        annotation_csv: Path,
        split_ids: Optional[List[str]] = None,
        k_neighbors: int = 48,
        design_domains: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize context-aware dataset.
        
        Args:
            cif_dir: Directory containing CIF files
            annotation_csv: Path to annotations CSV
            split_ids: List of fragment IDs for this split
            k_neighbors: Number of neighbors for context
            design_domains: Domains to use as design regions
        """
        super().__init__(
            cif_dir=cif_dir,
            annotation_csv=annotation_csv,
            split_ids=split_ids,
            cropping_strategy="context_aware",
            cropper_kwargs={
                'k_neighbors': k_neighbors,
                'design_domains': design_domains,
            },
            **kwargs
        )


class PKSBatchSampler(Sampler):
    """
    Batch sampler that groups sequences by length for efficient batching.
    
    ProteinMPNN uses variable-length sequences, so batching by similar lengths
    reduces padding waste.
    """
    
    def __init__(
        self,
        dataset: PKSDataset,
        batch_size: int = 10000,  # Max tokens per batch
        shuffle: bool = True,
        drop_last: bool = False
    ):
        """
        Initialize the sampler.
        
        Args:
            dataset: PKS dataset
            batch_size: Maximum tokens (residues) per batch
            shuffle: Shuffle batches
            drop_last: Drop incomplete final batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Estimate lengths (will be refined during iteration)
        self._build_batches()
    
    def _build_batches(self):
        """Group samples into batches by length."""
        # Get sequence lengths
        lengths = []
        for sample in self.dataset.samples:
            annotation = self.dataset.annotation_parser.get(sample['fragment_id'])
            if annotation:
                lengths.append(annotation.length)
            else:
                lengths.append(1000)  # Default estimate
        
        # Sort by length
        sorted_indices = np.argsort(lengths)
        
        # Group into batches
        self.batches = []
        current_batch = []
        current_tokens = 0
        
        for idx in sorted_indices:
            length = lengths[idx]
            
            if current_tokens + length > self.batch_size and current_batch:
                self.batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            
            current_batch.append(idx)
            current_tokens += length
        
        if current_batch and not self.drop_last:
            self.batches.append(current_batch)
    
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.batches)
        
        for batch in self.batches:
            yield batch
    
    def __len__(self):
        return len(self.batches)


def collate_pks_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for PKS batches.
    
    Handles variable-length sequences by padding.
    """
    # Find max length
    max_len = max(item['S'].shape[0] for item in batch)
    
    # Batch tensors
    batch_size = len(batch)
    
    # Initialize padded tensors
    X = torch.zeros(batch_size, max_len, 4, 3)
    S = torch.zeros(batch_size, max_len, dtype=torch.long)
    mask = torch.zeros(batch_size, max_len)
    chain_M = torch.zeros(batch_size, max_len)
    chain_encoding = torch.zeros(batch_size, max_len, dtype=torch.long)
    residue_idx = torch.zeros(batch_size, max_len, dtype=torch.long)
    mask_for_loss = torch.zeros(batch_size, max_len)
    lengths = []
    names = []
    
    for i, item in enumerate(batch):
        L = item['S'].shape[0]
        lengths.append(L)
        names.append(item.get('name', f'sample_{i}'))
        
        X[i, :L] = item['X']
        S[i, :L] = item['S']
        mask[i, :L] = item['mask']
        chain_M[i, :L] = item['chain_M']
        chain_encoding[i, :L] = item['chain_encoding']
        residue_idx[i, :L] = item['residue_idx']
        mask_for_loss[i, :L] = item['mask_for_loss']
    
    return {
        'X': X,
        'S': S,
        'mask': mask,
        'chain_M': chain_M,
        'chain_encoding': chain_encoding,
        'residue_idx': residue_idx,
        'mask_for_loss': mask_for_loss,
        'lengths': torch.tensor(lengths),
        'names': names,
    }
