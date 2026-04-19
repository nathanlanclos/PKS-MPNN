#!/bin/bash
#SBATCH --job-name=pks_prep
#SBATCH --partition=savio3_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --account=fc_pkss
#SBATCH --gres=gpu:A40:1
#SBATCH --qos=a40_gpu3_normal
#SBATCH --output=logs/prepare_%j.out
#SBATCH --error=logs/prepare_%j.err

# =============================================================================
# PKS-MPNN Data Preparation
#
# Parses CIF files, clusters sequences, and creates train/val/test splits.
#
# Prerequisites:
#   - CIF files linked to data/raw/
#   - Annotation CSV in repo root
#
# Usage:
#   sbatch slurm/prepare_data.sbatch
# =============================================================================

echo "=========================================="
echo "PKS-MPNN Data Preparation"
echo "=========================================="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Load modules
module load anaconda3
source activate pks-mpnn

# Navigate to project directory
cd /global/scratch/users/$USER/PKS-MPNN

# Set paths - MODIFY THESE FOR YOUR SETUP
CIF_DIR="data/raw"
ANNOTATION_CSV="fragments_for_prediction_COREONLY.csv"
OUTPUT_DIR="data/processed"
SPLITS_DIR="data/splits"

# Step 1: Parse CIF files and extract pLDDT
echo ""
echo "Step 1: Parsing CIF files..."
python scripts/01_prepare_data.py \
    --cif_dir $CIF_DIR \
    --annotation_csv $ANNOTATION_CSV \
    --output_dir $OUTPUT_DIR

# Step 2: Cluster sequences (requires mmseqs2)
echo ""
echo "Step 2: Clustering sequences..."
# Load mmseqs2 if available as module
module load mmseqs2 2>/dev/null || echo "mmseqs2 not a module, assuming in PATH"

python scripts/02_cluster_sequences.py \
    --annotation_csv $ANNOTATION_CSV \
    --output_dir $SPLITS_DIR \
    --min_seq_identity 0.7

# Step 3: Create train/val/test splits
echo ""
echo "Step 3: Creating splits..."
python scripts/03_create_splits.py \
    --annotation_csv $ANNOTATION_CSV \
    --clusters_json $SPLITS_DIR/clusters_70.json \
    --output_dir $SPLITS_DIR \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --seed 42

echo ""
echo "=========================================="
echo "Data preparation complete!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  - $OUTPUT_DIR/matched_structures.json"
echo "  - $OUTPUT_DIR/plddt_statistics.json"
echo "  - $SPLITS_DIR/train.txt"
echo "  - $SPLITS_DIR/val.txt"
echo "  - $SPLITS_DIR/test.txt"
