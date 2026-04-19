#!/bin/bash
#SBATCH --job-name=pks_setup
#SBATCH --partition=savio3_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --account=fc_pkss
#SBATCH --gres=gpu:A40:1
#SBATCH --qos=a40_gpu3_normal
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err

# =============================================================================
# PKS-MPNN Environment Setup for Berkeley Savio Cluster
# 
# Run once to set up the conda environment and clone ProteinMPNN.
#
# Usage:
#   mkdir -p logs
#   sbatch slurm/setup_environment.sbatch
# =============================================================================

echo "=========================================="
echo "PKS-MPNN Environment Setup"
echo "=========================================="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Load required modules
module load anaconda3

# Navigate to scratch directory
cd /global/scratch/users/$USER

# Clone the repository if needed
if [ ! -d "PKS-MPNN" ]; then
    echo "Cloning PKS-MPNN repository..."
    git clone https://github.com/your-org/PKS-MPNN.git
fi

cd PKS-MPNN

# Create conda environment
echo "Creating conda environment..."
conda env create -f environment.yml -n pks-mpnn || {
    echo "Environment exists, updating..."
    conda env update -f environment.yml -n pks-mpnn
}

# Activate environment
source activate pks-mpnn

# Install the package in development mode
echo "Installing PKS-MPNN package..."
pip install -e .

# Clone ProteinMPNN if not already present
if [ ! -d "ProteinMPNN" ]; then
    echo "Cloning ProteinMPNN..."
    git clone https://github.com/dauparas/ProteinMPNN.git
fi

# Create necessary directories
mkdir -p data/raw data/processed data/splits data/annotations
mkdir -p outputs logs

# Test GPU availability
echo ""
echo "Testing GPU availability..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Test imports
echo ""
echo "Testing imports..."
python -c "from src.data.cif_parser import CIFParser; from src.model.protein_mpnn import ProteinMPNNWrapper; print('All imports successful!')"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Create symlink to CIF files:"
echo "     ln -s /path/to/your/cif_files data/raw"
echo ""
echo "  2. Copy annotation CSV:"
echo "     cp /path/to/fragments_for_prediction_COREONLY.csv ."
echo ""
echo "  3. Prepare data:"
echo "     sbatch slurm/prepare_data.sbatch"
