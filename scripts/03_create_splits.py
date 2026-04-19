#!/usr/bin/env python3
"""
Create train/val/test splits for PKS-MPNN experiments.

Splits are based on sequence clusters to prevent data leakage.
All 5 AF3 models of a given sequence stay in the same split.

Usage:
    python scripts/03_create_splits.py \
        --annotation_csv fragments_for_prediction_COREONLY.csv \
        --clusters_json data/splits/clusters_70.json \
        --output_dir data/splits \
        --train_ratio 0.8 \
        --val_ratio 0.1 \
        --test_ratio 0.1
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.annotation_parser import AnnotationParser
from src.data.clustering import create_cluster_aware_splits, save_splits


def parse_args():
    parser = argparse.ArgumentParser(description="Create train/val/test splits")
    parser.add_argument(
        "--annotation_csv",
        type=Path,
        required=True,
        help="Path to annotations CSV"
    )
    parser.add_argument(
        "--clusters_json",
        type=Path,
        required=True,
        help="Path to clusters JSON from clustering step"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/splits"),
        help="Output directory"
    )
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate ratios
    total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total - 1.0) > 1e-6:
        print(f"Warning: Ratios sum to {total}, normalizing...")
        args.train_ratio /= total
        args.val_ratio /= total
        args.test_ratio /= total
    
    print("=" * 60)
    print("PKS-MPNN Dataset Splitting")
    print("=" * 60)
    
    # Load annotations
    print(f"\nLoading annotations from {args.annotation_csv}...")
    annotation_parser = AnnotationParser(args.annotation_csv)
    print(f"  Loaded {len(annotation_parser)} annotations")
    
    # Load clusters
    print(f"\nLoading clusters from {args.clusters_json}...")
    with open(args.clusters_json) as f:
        clusters = json.load(f)
    # Convert string keys to int
    clusters = {int(k): v for k, v in clusters.items()}
    print(f"  Loaded {len(clusters)} clusters")
    
    # Create splits
    print(f"\nCreating splits with ratio {args.train_ratio}/{args.val_ratio}/{args.test_ratio}...")
    train_ids, val_ids, test_ids = create_cluster_aware_splits(
        clusters,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    print(f"  Train: {len(train_ids)} sequences")
    print(f"  Val:   {len(val_ids)} sequences")
    print(f"  Test:  {len(test_ids)} sequences")
    
    # Analyze splits by domain composition
    print("\nAnalyzing domain composition by split...")
    from collections import Counter
    
    for split_name, split_ids in [
        ('Train', train_ids),
        ('Val', val_ids),
        ('Test', test_ids)
    ]:
        compositions = []
        for fid in split_ids:
            ann = annotation_parser.get(fid)
            if ann:
                compositions.append(ann.fragment_composition)
        
        counts = Counter(compositions)
        print(f"\n  {split_name} - Top 5 compositions:")
        for comp, count in counts.most_common(5):
            print(f"    {comp}: {count}")
    
    # Save splits
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_splits(train_ids, val_ids, test_ids, args.output_dir)
    
    # Save additional metadata
    metadata = {
        'train_ratio': args.train_ratio,
        'val_ratio': args.val_ratio,
        'test_ratio': args.test_ratio,
        'seed': args.seed,
        'n_clusters': len(clusters),
        'n_train': len(train_ids),
        'n_val': len(val_ids),
        'n_test': len(test_ids),
    }
    
    with open(args.output_dir / "split_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSaved splits to {args.output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
