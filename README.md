# PKS-MPNN

Fine-tuning ProteinMPNN for Polyketide Synthase (PKS) sequence design.

## Overview

PKS-MPNN adapts ProteinMPNN to predict amino acid sequences for PKS module structures predicted by AlphaFold3. The project implements three experimental strategies to understand how structural context affects sequence prediction:

1. **Domain-Only**: Train on individual domains (KS, AT, DH, ER, KR, ACP) without neighboring context
2. **Full Module**: Train on complete module structures with pLDDT-aware masking
3. **Context-Aware Cropping**: Smart cropping that preserves K=48 nearest neighbor relationships while maximizing geometric diversity

## Key Features

- **pLDDT-aware training**: Tiered masking based on AlphaFold confidence scores
  - pLDDT > 70: Full loss contribution
  - 50 < pLDDT ≤ 70: Include in graph, mask from loss
  - pLDDT ≤ 50: Exclude from structure entirely
- **wandb integration**: Real-time tracking of perplexity, NLL, recovery, and per-domain metrics
- **Berkeley Savio cluster support**: SLURM scripts for A40 GPUs
- **Modular experiment design**: YAML configs for easy experiment management

## Installation

### Local Development

```bash
# Clone the repository
git clone https://github.com/your-org/PKS-MPNN.git
cd PKS-MPNN

# Create conda environment
conda env create -f environment.yml
conda activate pks-mpnn

# Install package in development mode
pip install -e .

# Clone ProteinMPNN (if not already present)
git clone https://github.com/dauparas/ProteinMPNN.git
```

### Berkeley Savio Cluster

```bash
# Submit environment setup job
mkdir -p logs
sbatch slurm/setup_environment.sbatch
```

## Weights & Biases (wandb) Setup

To connect training runs to your wandb project:

1. **Login** (one-time):
   ```bash
   wandb login
   ```

2. **Set your project and entity** in config or via CLI:
   - Edit `configs/base.yaml` and set `wandb.entity` to your username or team name
   - Or pass at runtime:
     ```bash
     python scripts/04_train_domain_only.py --config configs/exp_domain_only.yaml \
       --wandb_project YOUR_PROJECT --wandb_entity YOUR_USERNAME_OR_TEAM
     ```

3. **View runs** at [wandb.ai](https://wandb.ai) under your project.

To disable wandb: use `--wandb_mode disabled` or set `wandb.mode: disabled` in your config.

## Data Preparation

### 1. Link your CIF files

```bash
# Create symlink to your CIF files
ln -s /mnt/d/PKS_Modeling/extracted_cif_files data/raw
```

### 2. Copy annotation CSV

```bash
# Copy the annotations file to repo root
cp /path/to/fragments_for_prediction_COREONLY.csv .
```

### 3. Run data preparation pipeline

```bash
# Parse CIF files and extract pLDDT
python scripts/01_prepare_data.py \
    --cif_dir data/raw \
    --annotation_csv fragments_for_prediction_COREONLY.csv \
    --output_dir data/processed

# Cluster sequences at 70% identity
python scripts/02_cluster_sequences.py \
    --annotation_csv fragments_for_prediction_COREONLY.csv \
    --output_dir data/splits \
    --min_seq_identity 0.7

# Create train/val/test splits
python scripts/03_create_splits.py \
    --annotation_csv fragments_for_prediction_COREONLY.csv \
    --clusters_json data/splits/clusters_70.json \
    --output_dir data/splits
```

## Quick Smoke Test

To verify the training pipeline with minimal data (no clustering required):

```bash
# 1. Ensure CIF files are in data/raw (symlink from your source)
ln -s /path/to/your/cif/files data/raw

# 2. Create minimal splits (5 train, 2 val structures)
python scripts/create_minimal_splits.py \
    --cif_dir data/raw \
    --annotation_csv fragments_for_prediction_COREONLY.csv \
    --output_dir data/splits \
    --n_train 5 --n_val 2

# 3. Run a 2-epoch smoke test (wandb disabled)
python scripts/04_train_domain_only.py --config configs/exp_domain_only.yaml \
    --num_epochs 2 --domain_type KS --wandb_mode disabled

# Or with wandb to verify logging:
python scripts/04_train_domain_only.py --config configs/exp_domain_only.yaml \
    --num_epochs 2 --domain_type KS \
    --wandb_project PKS-MPNN --wandb_entity YOUR_USERNAME
```

If you already have full splits, use `--max_train_samples` and `--max_val_samples` to limit dataset size for fast tests.

## Training

### Experiment 1: Domain-Only

Train on individual domains without context:

```bash
# Train on all domains
python scripts/04_train_domain_only.py --config configs/exp_domain_only.yaml

# Train on specific domain (e.g., KS)
python scripts/04_train_domain_only.py --config configs/exp_domain_only.yaml --domain_type KS
```

### Experiment 2: Full Module

Train on complete structures with pLDDT masking:

```bash
python scripts/05_train_full_module.py --config configs/exp_full_module.yaml
```

### Experiment 3: Context-Aware Cropping

Train with K=48 neighbor-preserving crops:

```bash
python scripts/06_train_context_crop.py --config configs/exp_context_crop.yaml
```

### SLURM Submission (Savio)

```bash
# Domain-only (all domains)
sbatch slurm/train_domain_only.sbatch

# Domain-only (specific domain)
DOMAIN_TYPE=KS sbatch slurm/train_domain_only.sbatch

# Full module
sbatch slurm/train_full_module.sbatch

# Context-aware
sbatch slurm/train_context_crop.sbatch
```

## Evaluation

```bash
# Evaluate on test set
python scripts/07_evaluate.py \
    --checkpoint outputs/full_module/checkpoints/best_model.pt \
    --config configs/exp_full_module.yaml \
    --output_dir outputs/evaluation

# SLURM submission
CHECKPOINT=outputs/full_module/checkpoints/best_model.pt \
CONFIG=configs/exp_full_module.yaml \
sbatch slurm/evaluate.sbatch
```

## Visualization Notebooks

Explore the dataset and training results:

1. `notebooks/01_dataset_exploration.ipynb` - Domain composition, sequence lengths
2. `notebooks/02_plddt_analysis.ipynb` - pLDDT distributions by domain type
3. `notebooks/03_clustering_viz.ipynb` - Cluster sizes, split analysis
4. `notebooks/04_cropping_viz.ipynb` - Compare cropping strategies
5. `notebooks/05_training_analysis.ipynb` - Training curves, experiment comparison

## Configuration

Experiments are configured via YAML files in `configs/`:

```yaml
# configs/exp_context_crop.yaml
experiment:
  name: context_crop
  type: context_aware

cropping:
  strategy: context_aware
  k_neighbors: 48
  plddt_design_threshold: 70.0
  plddt_context_threshold: 50.0

training:
  num_epochs: 100
  batch_size: 8000
  backbone_noise: 0.2

wandb:
  project: PKS-MPNN
  mode: online
```

## Project Structure

```
PKS-MPNN/
├── configs/                    # Experiment configurations
│   ├── base.yaml              # Shared settings
│   ├── exp_domain_only.yaml   
│   ├── exp_full_module.yaml   
│   └── exp_context_crop.yaml  
├── data/
│   ├── raw/                   # Symlink to CIF files
│   ├── processed/             # Parsed structures
│   └── splits/                # Train/val/test splits
├── src/
│   ├── data/                  # Data processing
│   │   ├── cif_parser.py      # CIF parsing
│   │   ├── annotation_parser.py
│   │   ├── cropping/          # Three cropping strategies
│   │   ├── dataset.py         # PyTorch datasets
│   │   └── clustering.py      # MMseqs2 wrapper
│   ├── model/                 # Model components
│   │   ├── protein_mpnn.py    # Model wrapper
│   │   ├── loss.py            # pLDDT-weighted loss
│   │   └── metrics.py         # Evaluation metrics
│   └── training/              # Training infrastructure
│       ├── trainer.py         # Training loop
│       └── optimizer.py       # Noam scheduler
├── scripts/                   # Training/evaluation scripts
├── notebooks/                 # Visualization notebooks
├── slurm/                     # SLURM job scripts
├── ProteinMPNN/              # Cloned ProteinMPNN repo
├── environment.yml           # Conda environment
└── setup.py                  # Package setup
```

## wandb Metrics

The following metrics are logged to wandb:

**Global metrics:**
- `train/loss`, `train/perplexity`, `train/accuracy`
- `val/loss`, `val/perplexity`, `val/accuracy`

**Per-domain metrics:**
- `train/domain_{KS,AT,DH,ER,KR,ACP,...}/perplexity`
- `train/domain_{...}/recovery`

**Confidence-stratified metrics:**
- `train/pLDDT_70-90/recovery`
- `train/pLDDT_50-70/recovery`

## Citation

If you use this code, please cite:

```bibtex
@article{dauparas2022robust,
  title={Robust deep learning-based protein sequence design using ProteinMPNN},
  author={Dauparas, Justas and others},
  journal={Science},
  year={2022}
}
```

## License

MIT License - see LICENSE for details.
