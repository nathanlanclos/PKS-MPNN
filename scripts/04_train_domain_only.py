#!/usr/bin/env python3
"""
Train PKS-MPNN with domain-only cropping strategy.

Trains on individual PKS domains (KS, AT, DH, ER, KR, ACP, etc.)
without context from neighboring domains or linkers.

Usage:
    python scripts/04_train_domain_only.py --config configs/exp_domain_only.yaml
    
    # Train on specific domain:
    python scripts/04_train_domain_only.py --config configs/exp_domain_only.yaml --domain_type KS
    
    # Disable wandb:
    python scripts/04_train_domain_only.py --config configs/exp_domain_only.yaml --wandb_mode disabled
"""

import argparse
import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config, merge_configs
from src.data.dataset import DomainOnlyDataset, PKSBatchSampler, collate_pks_batch
from src.data.clustering import load_splits
from src.model.protein_mpnn import create_model
from src.training.trainer import PKSTrainer
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Train PKS-MPNN (domain-only)")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/exp_domain_only.yaml"),
        help="Path to config file"
    )
    parser.add_argument("--domain_type", type=str, default=None,
                       help="Specific domain to train on (e.g., KS, AT)")
    parser.add_argument("--cif_dir", type=Path, default=None)
    parser.add_argument("--annotation_csv", type=Path, default=None)
    parser.add_argument("--splits_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None,
                       help="wandb entity (username or team) for your project")
    parser.add_argument("--wandb_mode", type=str, default=None,
                       choices=["online", "offline", "disabled"])
    parser.add_argument("--checkpoint", type=Path, default=None,
                       help="Resume from checkpoint")
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None,
                       help="Limit training set size (for smoke tests)")
    parser.add_argument("--max_val_samples", type=int, default=None,
                       help="Limit validation set size (for smoke tests)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override with command line args
    if args.domain_type:
        config.setdefault('domain', {})['domain_type'] = args.domain_type
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
    
    # Limit samples for smoke tests
    max_train = args.max_train_samples
    max_val = args.max_val_samples
    
    # Set up paths
    repo_root = Path(__file__).parent.parent
    cif_dir = Path(config.get('data', {}).get('cif_dir', 'data/raw'))
    annotation_csv = Path(config.get('data', {}).get('annotation_csv'))
    splits_dir = Path(config.get('data', {}).get('splits_dir', 'data/splits'))
    
    domain_type = config.get('domain', {}).get('domain_type')
    output_dir = Path(config.get('output_dir', 'outputs'))
    if domain_type:
        output_dir = output_dir / f"domain_only_{domain_type}"
    else:
        output_dir = output_dir / "domain_only_all"
    
    print("=" * 60)
    print("PKS-MPNN Training: Domain-Only")
    print("=" * 60)
    print(f"\nDomain type: {domain_type or 'all'}")
    print(f"CIF directory: {cif_dir}")
    print(f"Annotations: {annotation_csv}")
    print(f"Output: {output_dir}")
    
    # Load splits
    print("\nLoading train/val/test splits...")
    train_ids, val_ids, test_ids = load_splits(splits_dir)
    if max_train is not None:
        train_ids = set(list(train_ids)[:max_train])
        print(f"  (Limited to {max_train} train samples)")
    if max_val is not None:
        val_ids = set(list(val_ids)[:max_val])
        print(f"  (Limited to {max_val} val samples)")
    print(f"  Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    
    # Create datasets
    print("\nCreating datasets...")
    plddt_threshold = config.get('cropping', {}).get('plddt_threshold', 70.0)
    
    train_dataset = DomainOnlyDataset(
        cif_dir=cif_dir,
        annotation_csv=annotation_csv,
        domain_type=domain_type,
        split_ids=list(train_ids),
        plddt_threshold=plddt_threshold,
    )
    
    val_dataset = DomainOnlyDataset(
        cif_dir=cif_dir,
        annotation_csv=annotation_csv,
        domain_type=domain_type,
        split_ids=list(val_ids),
        plddt_threshold=plddt_threshold,
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
    wandb_run_name = wandb_config.get('run_name', f'domain_only_{domain_type}')
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
    num_epochs = config.get('training', {}).get('num_epochs', 50)
    save_every = config.get('training', {}).get('save_every', 10)
    
    print(f"\nStarting training for {num_epochs} epochs...")
    trainer.train(num_epochs=num_epochs, save_every=save_every)


if __name__ == "__main__":
    main()
