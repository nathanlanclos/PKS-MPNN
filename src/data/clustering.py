"""
Sequence clustering using MMseqs2.

Clusters PKS module sequences to prevent data leakage between train/val/test
splits. All 5 AlphaFold3 models of a given sequence MUST stay in the same split.

Recommended: 70% sequence identity clustering for PKS modules.
"""

import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from collections import defaultdict


class SequenceClusterer:
    """
    Cluster protein sequences using MMseqs2.
    
    This ensures that similar sequences are grouped together for proper
    train/val/test splitting. All members of a cluster go to the same split.
    """
    
    def __init__(
        self,
        min_seq_identity: float = 0.7,
        coverage: float = 0.8,
        coverage_mode: int = 1,
        mmseqs_path: str = "mmseqs"
    ):
        """
        Initialize the clusterer.
        
        Args:
            min_seq_identity: Minimum sequence identity (0-1) for clustering
            coverage: Minimum alignment coverage (0-1)
            coverage_mode: 0=query, 1=target, 2=both
            mmseqs_path: Path to mmseqs2 executable
        """
        self.min_seq_identity = min_seq_identity
        self.coverage = coverage
        self.coverage_mode = coverage_mode
        self.mmseqs_path = mmseqs_path
        
        # Check if mmseqs2 is available
        self._check_mmseqs()
    
    def _check_mmseqs(self) -> None:
        """Verify mmseqs2 is available."""
        try:
            result = subprocess.run(
                [self.mmseqs_path, "version"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                raise RuntimeError("mmseqs2 returned non-zero exit code")
        except FileNotFoundError:
            raise RuntimeError(
                f"mmseqs2 not found at '{self.mmseqs_path}'. "
                "Install with: conda install -c conda-forge -c bioconda mmseqs2"
            )
    
    def cluster(
        self,
        sequences: Dict[str, str],
        work_dir: Optional[Path] = None
    ) -> Dict[int, List[str]]:
        """
        Cluster sequences and return cluster assignments.
        
        Args:
            sequences: Dict mapping sequence ID to amino acid sequence
            work_dir: Working directory for temp files (optional)
            
        Returns:
            Dict mapping cluster ID to list of sequence IDs in that cluster
        """
        # Create temp directory if not provided
        cleanup_work_dir = work_dir is None
        if work_dir is None:
            work_dir = Path(tempfile.mkdtemp(prefix="mmseqs_"))
        else:
            work_dir = Path(work_dir)
            work_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Write sequences to FASTA
            fasta_path = work_dir / "sequences.fasta"
            self._write_fasta(sequences, fasta_path)
            
            # Run MMseqs2 clustering
            db_path = work_dir / "seqdb"
            cluster_path = work_dir / "clusters"
            tmp_path = work_dir / "tmp"
            
            # Create sequence database
            subprocess.run([
                self.mmseqs_path, "createdb",
                str(fasta_path), str(db_path)
            ], check=True, capture_output=True)
            
            # Cluster sequences
            subprocess.run([
                self.mmseqs_path, "cluster",
                str(db_path), str(cluster_path), str(tmp_path),
                "--min-seq-id", str(self.min_seq_identity),
                "-c", str(self.coverage),
                "--cov-mode", str(self.coverage_mode),
            ], check=True, capture_output=True)
            
            # Convert to TSV
            tsv_path = work_dir / "clusters.tsv"
            subprocess.run([
                self.mmseqs_path, "createtsv",
                str(db_path), str(db_path), str(cluster_path), str(tsv_path)
            ], check=True, capture_output=True)
            
            # Parse clusters
            clusters = self._parse_clusters(tsv_path)
            
            return clusters
            
        finally:
            if cleanup_work_dir:
                shutil.rmtree(work_dir, ignore_errors=True)
    
    def _write_fasta(self, sequences: Dict[str, str], path: Path) -> None:
        """Write sequences to FASTA file."""
        with open(path, 'w') as f:
            for seq_id, sequence in sequences.items():
                f.write(f">{seq_id}\n")
                # Write in 80-character lines
                for i in range(0, len(sequence), 80):
                    f.write(f"{sequence[i:i+80]}\n")
    
    def _parse_clusters(self, tsv_path: Path) -> Dict[int, List[str]]:
        """Parse MMseqs2 TSV output into cluster assignments."""
        clusters = defaultdict(list)
        cluster_rep_to_id = {}
        
        with open(tsv_path) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    rep_id, member_id = parts[0], parts[1]
                    
                    # Assign cluster ID based on representative
                    if rep_id not in cluster_rep_to_id:
                        cluster_rep_to_id[rep_id] = len(cluster_rep_to_id)
                    
                    cluster_id = cluster_rep_to_id[rep_id]
                    clusters[cluster_id].append(member_id)
        
        return dict(clusters)
    
    def cluster_from_csv(
        self,
        csv_path: Path,
        id_column: str = "fragment_id",
        seq_column: str = "fragment_sequence"
    ) -> Dict[int, List[str]]:
        """
        Cluster sequences from a CSV file.
        
        Args:
            csv_path: Path to CSV file
            id_column: Column containing sequence IDs
            seq_column: Column containing sequences
            
        Returns:
            Dict mapping cluster ID to list of sequence IDs
        """
        df = pd.read_csv(csv_path)
        sequences = dict(zip(df[id_column], df[seq_column]))
        return self.cluster(sequences)


def group_af_models(fragment_ids: List[str]) -> Dict[str, List[str]]:
    """
    Group AlphaFold3 model predictions by their base sequence.
    
    AF3 generates 5 models per sequence. All 5 must stay in the same split.
    
    Assumes naming convention: {base_id}_model_{N}
    
    Args:
        fragment_ids: List of fragment/structure IDs
        
    Returns:
        Dict mapping base sequence ID to list of model IDs
    """
    import re
    
    groups = defaultdict(list)
    
    model_pattern = re.compile(r'(.+)_model_(\d+)$')
    
    for fid in fragment_ids:
        match = model_pattern.match(fid)
        if match:
            base_id = match.group(1)
            groups[base_id].append(fid)
        else:
            # No model suffix - treat as its own group
            groups[fid].append(fid)
    
    return dict(groups)


def create_cluster_aware_splits(
    clusters: Dict[int, List[str]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    stratify_by: Optional[Dict[str, str]] = None
) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Split data by clusters to prevent data leakage.
    
    All members of a cluster go to the same split. Optionally stratify
    by some property (e.g., domain composition).
    
    Args:
        clusters: Dict mapping cluster ID to list of sequence IDs
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for test
        seed: Random seed
        stratify_by: Optional dict mapping sequence ID to stratification group
        
    Returns:
        Tuple of (train_ids, val_ids, test_ids) as sets
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    np.random.seed(seed)
    
    cluster_ids = list(clusters.keys())
    np.random.shuffle(cluster_ids)
    
    n_clusters = len(cluster_ids)
    n_train = int(n_clusters * train_ratio)
    n_val = int(n_clusters * val_ratio)
    
    train_clusters = cluster_ids[:n_train]
    val_clusters = cluster_ids[n_train:n_train + n_val]
    test_clusters = cluster_ids[n_train + n_val:]
    
    # Collect sequence IDs for each split
    train_ids = set()
    for cid in train_clusters:
        train_ids.update(clusters[cid])
    
    val_ids = set()
    for cid in val_clusters:
        val_ids.update(clusters[cid])
    
    test_ids = set()
    for cid in test_clusters:
        test_ids.update(clusters[cid])
    
    return train_ids, val_ids, test_ids


def save_splits(
    train_ids: Set[str],
    val_ids: Set[str],
    test_ids: Set[str],
    output_dir: Path
) -> None:
    """
    Save train/val/test splits to files.
    
    Args:
        train_ids: Training set IDs
        val_ids: Validation set IDs
        test_ids: Test set IDs
        output_dir: Directory to save split files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "train.txt", 'w') as f:
        for sid in sorted(train_ids):
            f.write(f"{sid}\n")
    
    with open(output_dir / "val.txt", 'w') as f:
        for sid in sorted(val_ids):
            f.write(f"{sid}\n")
    
    with open(output_dir / "test.txt", 'w') as f:
        for sid in sorted(test_ids):
            f.write(f"{sid}\n")
    
    # Also save summary
    summary = {
        'n_train': len(train_ids),
        'n_val': len(val_ids),
        'n_test': len(test_ids),
        'train_fraction': len(train_ids) / (len(train_ids) + len(val_ids) + len(test_ids)),
        'val_fraction': len(val_ids) / (len(train_ids) + len(val_ids) + len(test_ids)),
        'test_fraction': len(test_ids) / (len(train_ids) + len(val_ids) + len(test_ids)),
    }
    
    import json
    with open(output_dir / "split_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved splits: {summary['n_train']} train, {summary['n_val']} val, {summary['n_test']} test")


def load_splits(splits_dir: Path) -> Tuple[Set[str], Set[str], Set[str]]:
    """Load train/val/test splits from files."""
    splits_dir = Path(splits_dir)
    
    def load_set(filename):
        path = splits_dir / filename
        if not path.exists():
            return set()
        with open(path) as f:
            return set(line.strip() for line in f if line.strip())
    
    return (
        load_set("train.txt"),
        load_set("val.txt"),
        load_set("test.txt")
    )
