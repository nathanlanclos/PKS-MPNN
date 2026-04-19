#!/usr/bin/env python3
"""
Prepare PKS-MPNN training data.

This script:
1. Parses CIF files from AlphaFold3 predictions
2. Extracts pLDDT confidence scores
3. Matches structures to domain annotations
4. Saves processed data for training

Usage:
    python scripts/01_prepare_data.py \
        --cif_dir /mnt/d/PKS_Modeling/extracted_cif_files \
        --annotation_csv fragments_for_prediction_COREONLY.csv \
        --output_dir data/processed
"""

import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cif_parser import CIFParser
from src.data.annotation_parser import AnnotationParser, match_cif_to_annotation


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare PKS-MPNN training data")
    parser.add_argument(
        "--cif_dir",
        type=Path,
        required=True,
        help="Directory containing CIF files"
    )
    parser.add_argument(
        "--annotation_csv",
        type=Path,
        required=True,
        help="Path to annotations CSV"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=1,
        help="Number of parallel workers"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("PKS-MPNN Data Preparation")
    print("=" * 60)
    
    # Load annotations
    print(f"\nLoading annotations from {args.annotation_csv}...")
    annotation_parser = AnnotationParser(args.annotation_csv)
    print(f"  Loaded {len(annotation_parser)} annotations")
    
    # Find CIF files
    print(f"\nFinding CIF files in {args.cif_dir}...")
    cif_files = list(args.cif_dir.glob("*.cif"))
    print(f"  Found {len(cif_files)} CIF files")
    
    # Parse CIF files and match to annotations
    print("\nParsing CIF files and matching to annotations...")
    cif_parser = CIFParser()
    
    matched = []
    unmatched = []
    parse_errors = []
    
    plddt_stats = {
        'all': [],
        'per_structure': []
    }
    
    for cif_path in tqdm(cif_files, desc="Processing"):
        try:
            # Parse structure
            structure = cif_parser.parse(cif_path)
            
            # Try to match to annotation
            fragment_id = match_cif_to_annotation(
                cif_path.name, 
                annotation_parser
            )
            
            if fragment_id:
                matched.append({
                    'cif_path': str(cif_path),
                    'cif_name': cif_path.stem,
                    'fragment_id': fragment_id,
                    'length': structure.length,
                    'mean_plddt': float(structure.plddt.mean()),
                    'min_plddt': float(structure.plddt.min()),
                    'max_plddt': float(structure.plddt.max()),
                    'is_dimer': structure.is_dimer,
                    'sequence': structure.sequence,
                })
                
                plddt_stats['all'].extend(structure.plddt.tolist())
                plddt_stats['per_structure'].append(structure.plddt.mean())
            else:
                unmatched.append(cif_path.stem)
                
        except Exception as e:
            parse_errors.append({
                'file': cif_path.stem,
                'error': str(e)
            })
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Total CIF files:      {len(cif_files)}")
    print(f"  Successfully matched: {len(matched)}")
    print(f"  Unmatched:            {len(unmatched)}")
    print(f"  Parse errors:         {len(parse_errors)}")
    
    if plddt_stats['all']:
        all_plddt = np.array(plddt_stats['all'])
        print(f"\npLDDT Statistics (all residues):")
        print(f"  Mean:   {all_plddt.mean():.1f}")
        print(f"  Median: {np.median(all_plddt):.1f}")
        print(f"  Std:    {all_plddt.std():.1f}")
        print(f"  Min:    {all_plddt.min():.1f}")
        print(f"  Max:    {all_plddt.max():.1f}")
        print(f"\n  >90 (very high): {(all_plddt > 90).mean()*100:.1f}%")
        print(f"  70-90 (confident): {((all_plddt >= 70) & (all_plddt <= 90)).mean()*100:.1f}%")
        print(f"  50-70 (low):       {((all_plddt >= 50) & (all_plddt < 70)).mean()*100:.1f}%")
        print(f"  <50 (very low):    {(all_plddt < 50).mean()*100:.1f}%")
    
    # Save results
    print(f"\nSaving results to {args.output_dir}...")
    
    # Save matched data
    with open(args.output_dir / "matched_structures.json", 'w') as f:
        json.dump(matched, f, indent=2)
    
    # Save unmatched for debugging
    with open(args.output_dir / "unmatched_structures.txt", 'w') as f:
        for name in unmatched:
            f.write(f"{name}\n")
    
    # Save parse errors
    if parse_errors:
        with open(args.output_dir / "parse_errors.json", 'w') as f:
            json.dump(parse_errors, f, indent=2)
    
    # Save pLDDT statistics
    with open(args.output_dir / "plddt_statistics.json", 'w') as f:
        json.dump({
            'mean': float(np.mean(plddt_stats['per_structure'])),
            'std': float(np.std(plddt_stats['per_structure'])),
            'min': float(np.min(plddt_stats['per_structure'])),
            'max': float(np.max(plddt_stats['per_structure'])),
            'n_structures': len(matched),
            'n_residues': len(plddt_stats['all']),
        }, f, indent=2)
    
    print("\nDone!")
    print(f"  Matched structures: {args.output_dir / 'matched_structures.json'}")
    print(f"  pLDDT statistics:   {args.output_dir / 'plddt_statistics.json'}")


if __name__ == "__main__":
    main()
