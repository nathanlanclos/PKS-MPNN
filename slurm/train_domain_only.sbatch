#!/bin/bash
#SBATCH --job-name=pks_domain
#SBATCH --partition=savio3_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --account=fc_pkss
#SBATCH --gres=gpu:A40:1
#SBATCH --qos=a40_gpu3_normal
#SBATCH --output=logs/domain_only_%j.out
#SBATCH --error=logs/domain_only_%j.err

# =============================================================================
# PKS-MPNN Domain-Only Training
#
# Train on individual PKS domains without context.
#
# Usage:
#   # Train on all domains:
#   sbatch slurm/train_domain_only.sbatch
#
#   # Train on specific domain (e.g., KS):
#   DOMAIN_TYPE=KS sbatch slurm/train_domain_only.sbatch
# =============================================================================

echo "=========================================="
echo "PKS-MPNN Training: Domain-Only"
echo "=========================================="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Domain type: ${DOMAIN_TYPE:-all}"

# Load modules
module load anaconda3
source activate pks-mpnn

# Navigate to project directory
cd /global/scratch/users/$USER/PKS-MPNN

# Set paths - MODIFY THESE FOR YOUR SETUP
CIF_DIR="data/raw"
ANNOTATION_CSV="fragments_for_prediction_COREONLY.csv"
SPLITS_DIR="data/splits"

# Construct command
CMD="python scripts/04_train_domain_only.py \
    --config configs/exp_domain_only.yaml \
    --cif_dir $CIF_DIR \
    --annotation_csv $ANNOTATION_CSV \
    --splits_dir $SPLITS_DIR \
    --wandb_project PKS-MPNN \
    --wandb_mode online"

# Add domain type if specified
if [ ! -z "$DOMAIN_TYPE" ]; then
    CMD="$CMD --domain_type $DOMAIN_TYPE"
fi

# Add number of epochs if specified
if [ ! -z "$NUM_EPOCHS" ]; then
    CMD="$CMD --num_epochs $NUM_EPOCHS"
fi

# Run training
echo ""
echo "Running: $CMD"
echo ""
eval $CMD

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
