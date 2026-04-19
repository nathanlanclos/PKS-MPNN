#!/bin/bash
#SBATCH --job-name=pks_eval
#SBATCH --partition=savio3_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --account=fc_pkss
#SBATCH --gres=gpu:A40:1
#SBATCH --qos=a40_gpu3_normal
#SBATCH --output=logs/evaluate_%j.out
#SBATCH --error=logs/evaluate_%j.err

# =============================================================================
# PKS-MPNN Evaluation
#
# Evaluate trained model on test set.
#
# Usage:
#   CHECKPOINT=outputs/full_module/checkpoints/best_model.pt \
#   CONFIG=configs/exp_full_module.yaml \
#   sbatch slurm/evaluate.sbatch
# =============================================================================

echo "=========================================="
echo "PKS-MPNN Evaluation"
echo "=========================================="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Validate required environment variables
if [ -z "$CHECKPOINT" ]; then
    echo "Error: CHECKPOINT environment variable not set"
    echo "Usage: CHECKPOINT=path/to/checkpoint.pt sbatch slurm/evaluate.sbatch"
    exit 1
fi

if [ -z "$CONFIG" ]; then
    echo "Error: CONFIG environment variable not set"
    echo "Usage: CONFIG=configs/exp_xxx.yaml sbatch slurm/evaluate.sbatch"
    exit 1
fi

echo "Checkpoint: $CHECKPOINT"
echo "Config: $CONFIG"

# Load modules
module load anaconda3
source activate pks-mpnn

# Navigate to project directory
cd /global/scratch/users/$USER/PKS-MPNN

# Set paths
CIF_DIR="data/raw"
ANNOTATION_CSV="fragments_for_prediction_COREONLY.csv"
SPLITS_DIR="data/splits"
OUTPUT_DIR="outputs/evaluation"

# Run evaluation
python scripts/07_evaluate.py \
    --checkpoint $CHECKPOINT \
    --config $CONFIG \
    --cif_dir $CIF_DIR \
    --annotation_csv $ANNOTATION_CSV \
    --splits_dir $SPLITS_DIR \
    --output_dir $OUTPUT_DIR \
    --split test

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "=========================================="
echo "Results saved to: $OUTPUT_DIR"
