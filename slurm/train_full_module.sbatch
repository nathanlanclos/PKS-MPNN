#!/bin/bash
#SBATCH --job-name=pks_full
#SBATCH --partition=savio3_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --account=fc_pkss
#SBATCH --gres=gpu:A40:1
#SBATCH --qos=a40_gpu3_normal
#SBATCH --output=logs/full_module_%j.out
#SBATCH --error=logs/full_module_%j.err

# =============================================================================
# PKS-MPNN Full Module Training
#
# Train on complete PKS module structures with pLDDT-aware masking.
#
# Usage:
#   sbatch slurm/train_full_module.sbatch
# =============================================================================

echo "=========================================="
echo "PKS-MPNN Training: Full Module"
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
python scripts/05_train_full_module.py \
    --config configs/exp_full_module.yaml \
    --cif_dir $CIF_DIR \
    --annotation_csv $ANNOTATION_CSV \
    --splits_dir $SPLITS_DIR \
    --wandb_project PKS-MPNN \
    --wandb_mode online

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
