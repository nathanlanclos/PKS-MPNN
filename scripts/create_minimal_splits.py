#!/usr/bin/env python3
"""
Create minimal train/val splits for smoke testing.

Creates train.txt and val.txt with only fragment IDs that have matching
structure files (.cif, .mmcif, .pdb, .ent) in the cif_dir. Use this to quickly
test the training pipeline without running the full clustering/splitting pipeline.

Usage:
    # Create splits from structures in data/raw (5 train, 2 val)
    python scripts/create_minimal_splits.py \
        --cif_dir data/raw \
        --annotation_csv fragments_for_prediction_COREONLY.csv \
        --output_dir data/splits \
        --n_train 5 \
        --n_val 2

    # Use with training (domain-only smoke test):
    python scripts/04_train_domain_only.py --config configs/exp_domain_only.yaml \
        --num_epochs 2 --domain_type KS --wandb_mode disabled
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.annotation_parser import AnnotationParser, match_cif_to_annotation
from src.data.cif_parser import list_structure_files


def parse_args():
    parser = argparse.ArgumentParser(description="Create minimal splits for smoke testing")
    parser.add_argument("--cif_dir", type=Path, required=True,
                       help="Directory containing structure files (.cif, .mmcif, .pdb, .ent)")
    parser.add_argument("--annotation_csv", type=Path, required=True,
                       help="Path to annotations CSV")
    parser.add_argument("--output_dir", type=Path, default=Path("data/splits"),
                       help="Output directory for split files")
    parser.add_argument("--n_train", type=int, default=5,
                       help="Number of structures for training")
    parser.add_argument("--n_val", type=int, default=2,
                       help="Number of structures for validation")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load annotations
    print(f"Loading annotations from {args.annotation_csv}...")
    annotation_parser = AnnotationParser(args.annotation_csv)
    
    # Find structure files and match to annotations
    cif_files = list_structure_files(args.cif_dir)
    if not cif_files:
        print(f"\nERROR: No structure files (.cif, .mmcif, .pdb, .ent) found in {args.cif_dir}")
        print("Please link or copy structure files first, e.g.:")
        print("  ln -s /path/to/your/structures data/raw")
        sys.exit(1)
    
    print(f"Found {len(cif_files)} structure files")
    
    matched = []
    for cif_path in cif_files:
        fragment_id = match_cif_to_annotation(cif_path.name, annotation_parser)
        if fragment_id:
            matched.append(fragment_id)
    
    # Deduplicate (multiple models can map to same fragment_id)
    matched_ids = list(dict.fromkeys(matched))
    print(f"Matched {len(matched_ids)} unique fragment IDs to annotations")
    
    if len(matched_ids) < args.n_train + args.n_val:
        print(f"\nWARNING: Only {len(matched_ids)} matched structures, but requested "
              f"{args.n_train} train + {args.n_val} val.")
        print("Reducing split sizes.")
        args.n_train = min(args.n_train, max(1, len(matched_ids) - 1))
        args.n_val = min(args.n_val, len(matched_ids) - args.n_train)
    
    # Shuffle and split
    import random
    random.seed(args.seed)
    random.shuffle(matched_ids)
    
    train_ids = matched_ids[:args.n_train]
    val_ids = matched_ids[args.n_train:args.n_train + args.n_val]
    
    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(args.output_dir / "train.txt", 'w') as f:
        for fid in train_ids:
            f.write(f"{fid}\n")
    
    with open(args.output_dir / "val.txt", 'w') as f:
        for fid in val_ids:
            f.write(f"{fid}\n")
    
    # Create empty test.txt (required by load_splits)
    with open(args.output_dir / "test.txt", 'w') as f:
        pass
    
    print(f"\nSaved minimal splits to {args.output_dir}")
    print(f"  Train: {len(train_ids)} structures")
    print(f"  Val:   {len(val_ids)} structures")
    print(f"\nRun smoke test with:")
    print(f"  python scripts/04_train_domain_only.py --config configs/exp_domain_only.yaml \\")
    print(f"    --num_epochs 2 --domain_type KS --wandb_mode disabled")


if __name__ == "__main__":
    main()
