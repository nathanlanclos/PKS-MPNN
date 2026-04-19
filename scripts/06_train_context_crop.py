#!/usr/bin/env python3
"""
Train PKS-MPNN with context-aware cropping strategy.

Smart cropping that preserves ProteinMPNN's K=48 nearest neighbor relationships
while maximizing geometric diversity at interfaces.

Usage:
    python scripts/06_train_context_crop.py --config configs/exp_context_crop.yaml
"""

import argparse
import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.data.dataset import ContextAwareDataset, PKSBatchSampler, collate_pks_batch
from src.data.clustering import load_splits
from src.model.protein_mpnn import create_model
from src.training.trainer import PKSTrainer
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Train PKS-MPNN (context-aware)")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/exp_context_crop.yaml"),
        help="Path to config file"
    )
    parser.add_argument("--cif_dir", type=Path, default=None)
    parser.add_argument("--annotation_csv", type=Path, default=None)
    parser.add_argument("--splits_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--k_neighbors", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override with command line args
    if args.cif_dir:
        config.setdefault('data', {})['cif_dir'] = str(args.cif_dir)
    if args.annotation_csv:
        config.setdefault('data', {})['annotation_csv'] = str(args.annotation_csv)
    if args.splits_dir:
        config.setdefault('data', {})['splits_dir'] = str(args.splits_dir)
    if args.output_dir:
        config['output_dir'] = str(args.output_dir)
    if args.wandb_project:
        config.setdefault('wandb', {})['project'] = args.wandb_project
    if args.wandb_entity:
        config.setdefault('wandb', {})['entity'] = args.wandb_entity
    if args.wandb_mode:
        config.setdefault('wandb', {})['mode'] = args.wandb_mode
    if args.num_epochs:
        config.setdefault('training', {})['num_epochs'] = args.num_epochs
    if args.k_neighbors:
        config.setdefault('cropping', {})['k_neighbors'] = args.k_neighbors
    
    # Set up paths
    repo_root = Path(__file__).parent.parent
    cif_dir = Path(config.get('data', {}).get('cif_dir', 'data/raw'))
    annotation_csv = Path(config.get('data', {}).get('annotation_csv'))
    splits_dir = Path(config.get('data', {}).get('splits_dir', 'data/splits'))
    output_dir = Path(config.get('output_dir', 'outputs')) / "context_crop"
    
    print("=" * 60)
    print("PKS-MPNN Training: Context-Aware Cropping")
    print("=" * 60)
    print(f"\nCIF directory: {cif_dir}")
    print(f"Annotations: {annotation_csv}")
    print(f"Output: {output_dir}")
    
    cropping_config = config.get('cropping', {})
    print(f"\nCropping configuration:")
    print(f"  K neighbors: {cropping_config.get('k_neighbors', 48)}")
    print(f"  Design pLDDT threshold: {cropping_config.get('plddt_design_threshold', 70.0)}")
    print(f"  Context pLDDT threshold: {cropping_config.get('plddt_context_threshold', 50.0)}")
    
    # Load splits
    print("\nLoading train/val/test splits...")
    train_ids, val_ids, test_ids = load_splits(splits_dir)
    print(f"  Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    
    # Create datasets
    print("\nCreating datasets...")
    
    train_dataset = ContextAwareDataset(
        cif_dir=cif_dir,
        annotation_csv=annotation_csv,
        split_ids=list(train_ids),
        k_neighbors=cropping_config.get('k_neighbors', 48),
        design_domains=cropping_config.get('design_domains'),
    )
    
    val_dataset = ContextAwareDataset(
        cif_dir=cif_dir,
        annotation_csv=annotation_csv,
        split_ids=list(val_ids),
        k_neighbors=cropping_config.get('k_neighbors', 48),
        design_domains=cropping_config.get('design_domains'),
    )
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    
    # Create data loaders
    batch_size = config.get('training', {}).get('batch_size', 8000)
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=PKSBatchSampler(train_dataset, batch_size=batch_size, shuffle=True),
        collate_fn=collate_pks_batch,
        num_workers=4,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=PKSBatchSampler(val_dataset, batch_size=batch_size, shuffle=False),
        collate_fn=collate_pks_batch,
        num_workers=4,
        pin_memory=True,
    )
    
    # Create model
    print("\nCreating model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    
    model_config = config.get('model', {})
    pretrained_path = config.get('finetune', {}).get('pretrained_weights')
    if pretrained_path:
        pretrained_path = repo_root / pretrained_path
    
    model = create_model(model_config, pretrained_path, device)
    
    # Configure unfreezing
    unfreeze_phase = config.get('finetune', {}).get('unfreeze_phase', 'decoder_only')
    model.configure_unfreezing(unfreeze_phase)
    print(f"  Unfreeze phase: {unfreeze_phase}")
    
    # Create trainer
    wandb_config = config.get('wandb', {})
    wandb_project = wandb_config.get('project') if wandb_config.get('mode') != 'disabled' else None
    wandb_run_name = wandb_config.get('run_name', 'context_crop')
    wandb_entity = wandb_config.get('entity')
    
    trainer = PKSTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=output_dir,
        device=device,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_entity=wandb_entity,
    )
    
    # Resume from checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Train
    num_epochs = config.get('training', {}).get('num_epochs', 100)
    save_every = config.get('training', {}).get('save_every', 10)
    
    print(f"\nStarting training for {num_epochs} epochs...")
    trainer.train(num_epochs=num_epochs, save_every=save_every)


if __name__ == "__main__":
    main()
