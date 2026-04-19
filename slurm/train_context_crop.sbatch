#!/bin/bash
#SBATCH --job-name=pks_context
#SBATCH --partition=savio3_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --account=fc_pkss
#SBATCH --gres=gpu:A40:1
#SBATCH --qos=a40_gpu3_normal
#SBATCH --output=logs/context_crop_%j.out
#SBATCH --error=logs/context_crop_%j.err

# =============================================================================
# PKS-MPNN Context-Aware Cropping Training
#
# Train with smart K=48 neighbor-preserving cropping strategy.
#
# Usage:
#   sbatch slurm/train_context_crop.sbatch
# =============================================================================

echo "=========================================="
echo "PKS-MPNN Training: Context-Aware Crop"
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
SPLITS_DIR="data/splits"

# Run training
python scripts/06_train_context_crop.py \
    --config configs/exp_context_crop.yaml \
    --cif_dir $CIF_DIR \
    --annotation_csv $ANNOTATION_CSV \
    --splits_dir $SPLITS_DIR \
    --wandb_project PKS-MPNN \
    --wandb_mode online

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
