#!/bin/bash
#SBATCH --job-name=pks_mmseqs_split
#SBATCH --partition=savio3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --account=fc_pkss
#SBATCH --output=logs/mmseqs_split_%j.out
#SBATCH --error=logs/mmseqs_split_%j.err

# =============================================================================
# MMseqs2 clustering + train/val/test splits (structure-aware)
#
# 1. Clusters sequences at MIN_SEQ_IDENTITY (only fragments with files in CIF_DIR).
# 2. Assigns whole clusters to splits (no sequence leakage).
# 3. Split files list fragment_id once; all models map to same ID → same split.
#
# Edit paths and account/partition for your site. Requires: mmseqs2, conda env.
#
# Usage:
#   mkdir -p logs
#   sbatch slurm/cluster_mmseqs_split.sbatch
# =============================================================================

set -euo pipefail
echo "Date: $(date)  Job: ${SLURM_JOB_ID:-local}  Host: $(hostname)"

module load anaconda3 2>/dev/null || true
source activate pks-mpnn

# --- edit for your deployment ---
REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"

ANNOTATION_CSV="${ANNOTATION_CSV:-fragments_for_prediction_COREONLY.csv}"
CIF_DIR="${CIF_DIR:-data/raw}"
SPLITS_DIR="${SPLITS_DIR:-data/splits_full}"
MIN_SEQ_IDENTITY="${MIN_SEQ_IDENTITY:-0.7}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
VAL_RATIO="${VAL_RATIO:-0.1}"
TEST_RATIO="${TEST_RATIO:-0.1}"
SEED="${SEED:-42}"

echo "ANNOTATION_CSV=$ANNOTATION_CSV"
echo "CIF_DIR=$CIF_DIR"
echo "SPLITS_DIR=$SPLITS_DIR"
echo "MIN_SEQ_IDENTITY=$MIN_SEQ_IDENTITY"

mmseqs version

python scripts/02_cluster_sequences.py \
  --annotation_csv "$ANNOTATION_CSV" \
  --output_dir "$SPLITS_DIR" \
  --cif_dir "$CIF_DIR" \
  --min_seq_identity "$MIN_SEQ_IDENTITY"

ID_TAG=$(python -c "print(int(float('${MIN_SEQ_IDENTITY}') * 100))")
CLUSTER_JSON="${SPLITS_DIR}/clusters_${ID_TAG}.json"
echo "Using clusters file: $CLUSTER_JSON"

python scripts/03_create_splits.py \
  --annotation_csv "$ANNOTATION_CSV" \
  --clusters_json "$CLUSTER_JSON" \
  --output_dir "$SPLITS_DIR" \
  --train_ratio "$TRAIN_RATIO" \
  --val_ratio "$VAL_RATIO" \
  --test_ratio "$TEST_RATIO" \
  --seed "$SEED" \
  --cif_dir "$CIF_DIR"

echo "Done. Splits: $SPLITS_DIR/train.txt val.txt test.txt"
