#!/usr/bin/env python3
"""
Scan MMseqs2 clustering at several sequence-identity thresholds.

Use this on a login node or cluster job to pick ``--min_seq_identity`` for
``02_cluster_sequences.py``. Lower identity → fewer, larger clusters (more
aggressive deduplication for split assignment).

Example:
    python scripts/scan_mmseqs_identity_thresholds.py \\
        --annotation_csv fragments_for_prediction_COREONLY.csv \\
        --cif_dir data/raw

Requires ``mmseqs`` on PATH (``conda install -c bioconda mmseqs2``).
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.annotation_parser import AnnotationParser
from src.data.clustering import SequenceClusterer, fragment_ids_with_structures


def parse_args():
    p = argparse.ArgumentParser(description="Scan MMseqs clustering vs identity threshold")
    p.add_argument("--annotation_csv", type=Path, required=True)
    p.add_argument(
        "--cif_dir",
        type=Path,
        default=None,
        help="Only consider fragment_ids with structure files (recommended)",
    )
    p.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
        help="Identity thresholds to try (0-1)",
    )
    p.add_argument("--coverage", type=float, default=0.8)
    p.add_argument("--output_json", type=Path, default=None, help="Save table as JSON")
    return p.parse_args()


def main():
    args = parse_args()
    ap = AnnotationParser(args.annotation_csv)
    sequences = {ann.fragment_id: ann.fragment_sequence for ann in ap}

    if args.cif_dir is not None:
        have = fragment_ids_with_structures(args.cif_dir, args.annotation_csv)
        sequences = {k: v for k, v in sequences.items() if k in have}
        print(f"Sequences after --cif_dir filter: {len(sequences)}")
    else:
        print(f"All annotation sequences: {len(sequences)} (no --cif_dir filter)")

    if not sequences:
        print("ERROR: no sequences to cluster.")
        sys.exit(1)

    rows = []

    print("\n" + "=" * 72)
    print(f"{'identity':>10} {'clusters':>10} {'singletons':>12} {'mean size':>12} {'max size':>10}")
    print("=" * 72)

    for thr in sorted(args.thresholds):
        if not 0 < thr <= 1:
            print(f"Skip invalid threshold {thr}")
            continue
        clusterer = SequenceClusterer(min_seq_identity=thr, coverage=args.coverage)
        tmp = Path(tempfile.mkdtemp(prefix="mmseqs_scan_"))
        try:
            clusters = clusterer.cluster(sequences, work_dir=tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        sizes = [len(m) for m in clusters.values()]
        n_clust = len(clusters)
        n_single = sum(1 for s in sizes if s == 1)
        mean_sz = sum(sizes) / len(sizes) if sizes else 0
        max_sz = max(sizes) if sizes else 0
        print(f"{thr:10.2f} {n_clust:10d} {n_single:12d} {mean_sz:12.2f} {max_sz:10d}")
        rows.append(
            {
                "min_seq_identity": thr,
                "n_clusters": n_clust,
                "n_singleton_clusters": n_single,
                "mean_cluster_size": mean_sz,
                "max_cluster_size": max_sz,
                "n_sequences": len(sequences),
            }
        )

    print("=" * 72)
    print(
        "\nInterpretation: higher identity → more clusters (stricter grouping). "
        "For PKS modules, 0.7 is a common starting point; tune if test leakage is a concern."
    )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
