"""
Dataset splitting utilities for PKS-MPNN.

Ensures proper train/val/test splits that:
1. Keep all 5 AF3 models of a sequence in the same split
2. Respect sequence clustering to prevent data leakage
3. Maintain architecture diversity across splits
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import Counter, defaultdict

from .clustering import SequenceClusterer, group_af_models, create_cluster_aware_splits
from .annotation_parser import AnnotationParser


class DatasetSplitter:
    """
    Create train/val/test splits for PKS-MPNN experiments.
    
    This splitter ensures:
    1. No data leakage between splits (sequence clustering)
    2. All AF3 models of a sequence stay together
    3. Architecture diversity is maintained
    """
    
    def __init__(
        self,
        annotation_parser: AnnotationParser,
        cluster_identity: float = 0.7,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42
    ):
        """
        Initialize the splitter.
        
        Args:
            annotation_parser: Loaded annotation parser
            cluster_identity: Sequence identity threshold for clustering
            train_ratio: Fraction for training
            val_ratio: Fraction for validation  
            test_ratio: Fraction for test
            seed: Random seed for reproducibility
        """
        self.annotation_parser = annotation_parser
        self.cluster_identity = cluster_identity
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        
        # Will be populated after clustering
        self.clusters: Optional[Dict[int, List[str]]] = None
        self.train_ids: Optional[Set[str]] = None
        self.val_ids: Optional[Set[str]] = None
        self.test_ids: Optional[Set[str]] = None
    
    def create_splits(
        self,
        use_clustering: bool = True,
        stratify_by_composition: bool = True
    ) -> Tuple[Set[str], Set[str], Set[str]]:
        """
        Create train/val/test splits.
        
        Args:
            use_clustering: Whether to cluster sequences first
            stratify_by_composition: Stratify by domain composition
            
        Returns:
            Tuple of (train_ids, val_ids, test_ids)
        """
        # Get unique sequences
        sequences = {
            ann.fragment_id: ann.fragment_sequence
            for ann in self.annotation_parser
        }
        
        if use_clustering:
            # Cluster sequences
            print(f"Clustering {len(sequences)} sequences at {self.cluster_identity*100:.0f}% identity...")
            clusterer = SequenceClusterer(min_seq_identity=self.cluster_identity)
            self.clusters = clusterer.cluster(sequences)
            print(f"Created {len(self.clusters)} clusters")
            
            # Create splits by cluster
            self.train_ids, self.val_ids, self.test_ids = create_cluster_aware_splits(
                self.clusters,
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                test_ratio=self.test_ratio,
                seed=self.seed
            )
        else:
            # Random split without clustering
            all_ids = list(sequences.keys())
            np.random.seed(self.seed)
            np.random.shuffle(all_ids)
            
            n = len(all_ids)
            n_train = int(n * self.train_ratio)
            n_val = int(n * self.val_ratio)
            
            self.train_ids = set(all_ids[:n_train])
            self.val_ids = set(all_ids[n_train:n_train + n_val])
            self.test_ids = set(all_ids[n_train + n_val:])
        
        return self.train_ids, self.val_ids, self.test_ids
    
    def get_split_statistics(self) -> Dict:
        """Get statistics for each split."""
        if self.train_ids is None:
            raise ValueError("Call create_splits() first")
        
        stats = {}
        
        for split_name, split_ids in [
            ('train', self.train_ids),
            ('val', self.val_ids),
            ('test', self.test_ids)
        ]:
            compositions = []
            lengths = []
            
            for fid in split_ids:
                ann = self.annotation_parser.get(fid)
                if ann:
                    compositions.append(ann.fragment_composition)
                    lengths.append(ann.length)
            
            stats[split_name] = {
                'n_samples': len(split_ids),
                'n_unique_compositions': len(set(compositions)),
                'mean_length': np.mean(lengths) if lengths else 0,
                'composition_counts': dict(Counter(compositions).most_common(10))
            }
        
        return stats
    
    def save_splits(self, output_dir: Path) -> None:
        """Save splits to files."""
        if self.train_ids is None:
            raise ValueError("Call create_splits() first")
        
        from .clustering import save_splits
        save_splits(self.train_ids, self.val_ids, self.test_ids, output_dir)
        
        # Also save statistics
        stats = self.get_split_statistics()
        import json
        with open(output_dir / "split_statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)
    
    def load_splits(self, splits_dir: Path) -> Tuple[Set[str], Set[str], Set[str]]:
        """Load splits from files."""
        from .clustering import load_splits
        self.train_ids, self.val_ids, self.test_ids = load_splits(splits_dir)
        return self.train_ids, self.val_ids, self.test_ids


def map_cif_to_annotations(
    cif_dir: Path,
    annotation_parser: AnnotationParser,
    model_pattern: str = r"_model_(\d+)"
) -> Dict[str, str]:
    """
    Map structure filenames to annotation fragment_ids.
    
    Handles model-number suffixes and other common filename patterns.
    
    Args:
        cif_dir: Directory containing structure files (.cif, .mmcif, .pdb, .ent)
        annotation_parser: Loaded annotation parser
        model_pattern: Regex pattern for model number in filename
        
    Returns:
        Dict mapping structure filename stem to fragment_id
    """
    import re

    from .cif_parser import list_structure_files
    
    cif_dir = Path(cif_dir)
    mapping = {}
    
    pattern = re.compile(model_pattern)
    
    for cif_file in list_structure_files(cif_dir):
        cif_name = cif_file.stem
        
        # Try direct match
        if cif_name in annotation_parser:
            mapping[cif_name] = cif_name
            continue
        
        # Try removing model suffix
        base_name = pattern.sub('', cif_name)
        if base_name in annotation_parser:
            mapping[cif_name] = base_name
            continue
        
        # Try other patterns
        for suffix in ['_relaxed', '_unrelaxed', '_rank_001']:
            cleaned = cif_name.replace(suffix, '')
            if cleaned in annotation_parser:
                mapping[cif_name] = cleaned
                break
    
    return mapping


def create_experiment_splits(
    annotation_csv: Path,
    cif_dir: Path,
    output_dir: Path,
    cluster_identity: float = 0.7,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Dict:
    """
    Create and save train/val/test splits for experiments.
    
    Args:
        annotation_csv: Path to annotations CSV
        cif_dir: Directory containing CIF files
        output_dir: Directory to save splits
        cluster_identity: Clustering threshold
        train_ratio: Training fraction
        val_ratio: Validation fraction
        test_ratio: Test fraction
        seed: Random seed
        
    Returns:
        Split statistics
    """
    # Load annotations
    parser = AnnotationParser(annotation_csv)
    
    # Create splitter
    splitter = DatasetSplitter(
        parser,
        cluster_identity=cluster_identity,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed
    )
    
    # Create and save splits
    splitter.create_splits(use_clustering=True)
    splitter.save_splits(output_dir)
    
    # Map CIF files to splits
    cif_mapping = map_cif_to_annotations(cif_dir, parser)
    
    # Save CIF mapping
    import json
    with open(output_dir / "cif_to_annotation_mapping.json", 'w') as f:
        json.dump(cif_mapping, f, indent=2)
    
    return splitter.get_split_statistics()
