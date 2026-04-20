#!/usr/bin/env python3
"""
Train PKS-MPNN with full module strategy.

Trains on complete PKS module structures with pLDDT-aware masking:
- pLDDT > 70: Full loss weight
- 50 < pLDDT <= 70: In graph, but masked from loss
- pLDDT <= 50: Excluded from graph

Usage:
    python scripts/05_train_full_module.py --config configs/exp_full_module.yaml
"""

import argparse
import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.data.dataset import FullModuleDataset, PKSBatchSampler, collate_pks_batch
from src.data.clustering import load_splits
from src.model.protein_mpnn import create_model
from src.training.trainer import PKSTrainer
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Train PKS-MPNN (full module)")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/exp_full_module.yaml"),
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
    
    # Set up paths
    repo_root = Path(__file__).parent.parent
    cif_dir = Path(config.get('data', {}).get('cif_dir', 'data/raw'))
    annotation_csv = Path(config.get('data', {}).get('annotation_csv'))
    splits_dir = Path(config.get('data', {}).get('splits_dir', 'data/splits'))
    output_dir = Path(config.get('output_dir', 'outputs')) / "full_module"
    
    print("=" * 60)
    print("PKS-MPNN Training: Full Module")
    print("=" * 60)
    print(f"\nCIF directory: {cif_dir}")
    print(f"Annotations: {annotation_csv}")
    print(f"Output: {output_dir}")
    
    # Load splits
    print("\nLoading train/val/test splits...")
    train_ids, val_ids, test_ids = load_splits(splits_dir)
    print(f"  Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    
    # Create datasets
    print("\nCreating datasets...")
    cropping_config = config.get('cropping', {})
    
    train_dataset = FullModuleDataset(
        cif_dir=cif_dir,
        annotation_csv=annotation_csv,
        split_ids=list(train_ids),
        high_confidence_threshold=cropping_config.get('high_confidence_threshold', 70.0),
        low_confidence_threshold=cropping_config.get('low_confidence_threshold', 50.0),
        min_trainable_residues=cropping_config.get('min_trainable_residues', 100),
        exclude_low_confidence=cropping_config.get('exclude_low_confidence', True),
    )
    
    val_dataset = FullModuleDataset(
        cif_dir=cif_dir,
        annotation_csv=annotation_csv,
        split_ids=list(val_ids),
        high_confidence_threshold=cropping_config.get('high_confidence_threshold', 70.0),
        low_confidence_threshold=cropping_config.get('low_confidence_threshold', 50.0),
        min_trainable_residues=cropping_config.get('min_trainable_residues', 100),
        exclude_low_confidence=cropping_config.get('exclude_low_confidence', True),
    )
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    
    # Create data loaders
    batch_size = config.get('training', {}).get('batch_size', 6000)
    
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
    wandb_run_name = wandb_config.get('run_name', 'full_module')
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
