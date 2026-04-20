#!/usr/bin/env python3
"""
Cluster PKS module sequences using MMseqs2.

Clusters are used by ``03_create_splits.py`` so **similar sequences never
appear in different splits** (no train/test leakage). Splits are by
**fragment_id**; every structure file for that ID (e.g. five AF models) maps
to the same ID, so all models stay in one split automatically.

Optional ``--cif_dir`` restricts clustering to fragment IDs that have at least
one structure file on disk (recommended before full training).

Usage:
    python scripts/02_cluster_sequences.py \
        --annotation_csv fragments_for_prediction_COREONLY.csv \
        --output_dir data/splits \
        --min_seq_identity 0.7

    # Only fragments with files under data/raw (aligns splits to your PDB/CIF set)
    python scripts/02_cluster_sequences.py \
        --annotation_csv fragments_for_prediction_COREONLY.csv \
        --output_dir data/splits \
        --cif_dir data/raw \
        --min_seq_identity 0.7
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.annotation_parser import AnnotationParser
from src.data.clustering import SequenceClusterer, fragment_ids_with_structures


def parse_args():
    parser = argparse.ArgumentParser(description="Cluster PKS sequences")
    parser.add_argument(
        "--annotation_csv",
        type=Path,
        required=True,
        help="Path to annotations CSV"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/splits"),
        help="Output directory for cluster assignments"
    )
    parser.add_argument(
        "--min_seq_identity",
        type=float,
        default=0.7,
        help="Minimum sequence identity for clustering (0-1)"
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=0.8,
        help="Minimum alignment coverage (0-1)"
    )
    parser.add_argument(
        "--cif_dir",
        type=Path,
        default=None,
        help="If set, only cluster fragment_ids that have ≥1 structure file here (same as training data dir)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("PKS-MPNN Sequence Clustering")
    print("=" * 60)
    
    # Load annotations
    print(f"\nLoading annotations from {args.annotation_csv}...")
    parser = AnnotationParser(args.annotation_csv)
    print(f"  Loaded {len(parser)} annotations")
    
    # Extract sequences (optionally restrict to fragments with structure files)
    sequences = {
        ann.fragment_id: ann.fragment_sequence
        for ann in parser
    }
    n_ann = len(sequences)
    if args.cif_dir is not None:
        with_files = fragment_ids_with_structures(args.cif_dir, args.annotation_csv)
        before = len(sequences)
        sequences = {fid: sequences[fid] for fid in sequences if fid in with_files}
        print(f"  Annotation entries: {n_ann}")
        print(f"  With structure files in {args.cif_dir}: {len(with_files)} fragment IDs")
        print(f"  Sequences used for clustering (intersection): {len(sequences)} "
              f"(dropped {before - len(sequences)} without files)")
        if not sequences:
            print("ERROR: No sequences left after --cif_dir filter.")
            sys.exit(1)
    else:
        print(f"  Extracted {len(sequences)} sequences (not filtered by --cif_dir)")
    
    # Cluster sequences
    print(f"\nClustering at {args.min_seq_identity*100:.0f}% sequence identity...")
    clusterer = SequenceClusterer(
        min_seq_identity=args.min_seq_identity,
        coverage=args.coverage
    )
    
    clusters = clusterer.cluster(
        sequences,
        work_dir=args.output_dir / "mmseqs_tmp"
    )
    
    print(f"  Created {len(clusters)} clusters")
    
    # Analyze cluster sizes
    cluster_sizes = [len(members) for members in clusters.values()]
    print(f"\nCluster size statistics:")
    print(f"  Min:    {min(cluster_sizes)}")
    print(f"  Max:    {max(cluster_sizes)}")
    print(f"  Mean:   {sum(cluster_sizes) / len(cluster_sizes):.1f}")
    print(f"  Median: {sorted(cluster_sizes)[len(cluster_sizes)//2]}")
    
    # Count singleton clusters
    n_singletons = sum(1 for s in cluster_sizes if s == 1)
    print(f"  Singletons: {n_singletons} ({n_singletons/len(clusters)*100:.1f}%)")
    
    # Save clusters
    output_file = args.output_dir / f"clusters_{int(args.min_seq_identity*100)}.json"
    with open(output_file, 'w') as f:
        json.dump(clusters, f, indent=2)
    
    print(f"\nSaved clusters to {output_file}")
    
    # Also save a mapping from sequence ID to cluster ID
    seq_to_cluster = {}
    for cluster_id, members in clusters.items():
        for member in members:
            seq_to_cluster[member] = cluster_id
    
    mapping_file = args.output_dir / f"seq_to_cluster_{int(args.min_seq_identity*100)}.json"
    with open(mapping_file, 'w') as f:
        json.dump(seq_to_cluster, f, indent=2)
    
    print(f"Saved sequence-to-cluster mapping to {mapping_file}")

    meta = {
        "min_seq_identity": args.min_seq_identity,
        "coverage": args.coverage,
        "n_sequences_clustered": len(sequences),
        "n_clusters": len(clusters),
        "cif_dir_filter": str(args.cif_dir) if args.cif_dir else None,
    }
    meta_path = args.output_dir / f"clustering_metadata_{int(args.min_seq_identity*100)}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {meta_path}")
    print("\nNext: python scripts/03_create_splits.py --clusters_json", output_file, "...")
    print("\nDone!")


if __name__ == "__main__":
    main()
