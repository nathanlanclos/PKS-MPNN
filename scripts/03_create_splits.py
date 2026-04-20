#!/usr/bin/env python3
"""
Create train/val/test splits for PKS-MPNN experiments.

**Clusters** (from ``02_cluster_sequences.py``) are assigned wholly to
train, val, or test, so similar sequences never leak across splits.
**fragment_id** appears once per split file; all structure files that map to
that ID (e.g. five models) are trained together in that split.

Usage:
    python scripts/03_create_splits.py \
        --annotation_csv fragments_for_prediction_COREONLY.csv \
        --clusters_json data/splits/clusters_70.json \
        --output_dir data/splits \
        --train_ratio 0.8 \
        --val_ratio 0.1 \
        --test_ratio 0.1

    # Report how many PDB/CIF files fall in each split (optional)
    python scripts/03_create_splits.py ... --cif_dir data/raw
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.annotation_parser import AnnotationParser
from src.data.clustering import (
    create_cluster_aware_splits,
    save_splits,
    count_structure_files_per_fragment,
    split_structure_file_totals,
)


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
    parser.add_argument(
        "--cif_dir",
        type=Path,
        default=None,
        help="If set, count structure files per split and write split_structure_stats.json",
    )
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
        'clusters_json': str(args.clusters_json.resolve()),
        'n_clusters': len(clusters),
        'n_train': len(train_ids),
        'n_val': len(val_ids),
        'n_test': len(test_ids),
    }
    
    file_counts = None
    if args.cif_dir is not None:
        print(f"\nCounting structure files under {args.cif_dir}...")
        file_counts = count_structure_files_per_fragment(args.cif_dir, args.annotation_csv)
        stats = split_structure_file_totals(train_ids, val_ids, test_ids, file_counts)
        metadata["structure_files"] = stats
        metadata["total_structure_files"] = (
            stats["train_structure_files"]
            + stats["val_structure_files"]
            + stats["test_structure_files"]
        )
        print(
            f"  Structure files — train: {stats['train_structure_files']}, "
            f"val: {stats['val_structure_files']}, test: {stats['test_structure_files']}"
        )
        with open(args.output_dir / "split_structure_stats.json", 'w') as f:
            json.dump(
                {
                    "per_split_totals": stats,
                    "note": "Per fragment_id, training loads all matching files; counts sum models.",
                },
                f,
                indent=2,
            )

    with open(args.output_dir / "split_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSaved splits to {args.output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
