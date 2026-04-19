#!/usr/bin/env python3
"""
Evaluate trained PKS-MPNN models on test set.

Computes:
- Overall perplexity, NLL, and sequence recovery
- Per-domain metrics
- Confidence-stratified metrics
- Comparison across experiment types

Usage:
    python scripts/07_evaluate.py \
        --checkpoint outputs/full_module/checkpoints/best_model.pt \
        --config configs/exp_full_module.yaml \
        --output_dir outputs/evaluation
"""

import argparse
import json
import sys
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.data.dataset import PKSDataset, PKSBatchSampler, collate_pks_batch
from src.data.clustering import load_splits
from src.model.protein_mpnn import ProteinMPNNWrapper
from src.model.loss import PLDDTWeightedLoss
from src.model.metrics import (
    compute_perplexity,
    compute_recovery,
    PerDomainMetrics,
    ConfidenceStratifiedMetrics
)
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PKS-MPNN")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to config file used for training"
    )
    parser.add_argument("--cif_dir", type=Path, default=None)
    parser.add_argument("--annotation_csv", type=Path, default=None)
    parser.add_argument("--splits_dir", type=Path, default=None)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/evaluation"),
        help="Output directory for results"
    )
    parser.add_argument("--split", type=str, default="test",
                       choices=["train", "val", "test"])
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, data_loader, criterion, device):
    """Run evaluation on dataset."""
    model.eval()
    
    total_loss = 0.0
    total_correct = 0.0
    total_residues = 0.0
    
    per_domain_metrics = PerDomainMetrics()
    confidence_metrics = ConfidenceStratifiedMetrics()
    
    all_results = []
    
    for batch in tqdm(data_loader, desc="Evaluating"):
        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        log_probs = model(
            batch['X'],
            batch['S'],
            batch['mask'],
            batch['chain_M'],
            batch['residue_idx'],
            batch['chain_encoding']
        )
        
        loss, metrics = criterion(
            log_probs,
            batch['S'],
            batch['mask'],
            plddt=batch.get('plddt'),
            loss_mask=batch.get('mask_for_loss')
        )
        
        # Update totals
        total_loss += metrics['loss'].item() * metrics['n_residues'].item()
        total_correct += metrics['accuracy'].item() * metrics['n_residues'].item()
        total_residues += metrics['n_residues'].item()
        
        # Per-domain metrics
        if 'domain_labels' in batch:
            per_domain_metrics.update(
                log_probs,
                batch['S'],
                batch['mask'],
                batch['domain_labels']
            )
        
        # Confidence-stratified metrics
        if 'plddt' in batch:
            confidence_metrics.update(
                log_probs,
                batch['S'],
                batch['mask'],
                batch['plddt']
            )
        
        # Store per-sample results
        predictions = log_probs.argmax(dim=-1)
        for i in range(len(batch['names'])):
            mask = batch['mask'][i].cpu().numpy().astype(bool)
            preds = predictions[i].cpu().numpy()[mask]
            targets = batch['S'][i].cpu().numpy()[mask]
            
            all_results.append({
                'name': batch['names'][i],
                'length': int(mask.sum()),
                'recovery': float((preds == targets).mean()),
            })
    
    # Compute final metrics
    avg_loss = total_loss / (total_residues + 1e-8)
    avg_recovery = total_correct / (total_residues + 1e-8)
    
    return {
        'overall': {
            'loss': avg_loss,
            'perplexity': float(np.exp(avg_loss)),
            'recovery': avg_recovery,
            'n_residues': int(total_residues),
        },
        'per_domain': per_domain_metrics.compute(),
        'per_confidence': confidence_metrics.compute(),
        'per_sample': all_results,
    }


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override paths if provided
    if args.cif_dir:
        config.setdefault('data', {})['cif_dir'] = str(args.cif_dir)
    if args.annotation_csv:
        config.setdefault('data', {})['annotation_csv'] = str(args.annotation_csv)
    if args.splits_dir:
        config.setdefault('data', {})['splits_dir'] = str(args.splits_dir)
    
    # Set up paths
    cif_dir = Path(config.get('data', {}).get('cif_dir', 'data/raw'))
    annotation_csv = Path(config.get('data', {}).get('annotation_csv'))
    splits_dir = Path(config.get('data', {}).get('splits_dir', 'data/splits'))
    
    print("=" * 60)
    print("PKS-MPNN Evaluation")
    print("=" * 60)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Split: {args.split}")
    
    # Load splits
    train_ids, val_ids, test_ids = load_splits(splits_dir)
    split_ids = {'train': train_ids, 'val': val_ids, 'test': test_ids}[args.split]
    print(f"\n{args.split.capitalize()} set: {len(split_ids)} samples")
    
    # Determine cropping strategy from config
    experiment_type = config.get('experiment', {}).get('type', 'full_module')
    cropping_strategy = config.get('cropping', {}).get('strategy', experiment_type)
    
    # Create dataset
    dataset = PKSDataset(
        cif_dir=cif_dir,
        annotation_csv=annotation_csv,
        split_ids=list(split_ids),
        cropping_strategy=cropping_strategy,
        cropper_kwargs=config.get('cropping', {}),
    )
    
    print(f"Dataset samples: {len(dataset)}")
    
    # Create data loader
    batch_size = config.get('training', {}).get('batch_size', 8000)
    data_loader = DataLoader(
        dataset,
        batch_sampler=PKSBatchSampler(dataset, batch_size=batch_size, shuffle=False),
        collate_fn=collate_pks_batch,
        num_workers=4,
    )
    
    # Load model
    print("\nLoading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    model = ProteinMPNNWrapper(
        hidden_dim=config.get('model', {}).get('hidden_dim', 128),
        num_encoder_layers=config.get('model', {}).get('num_encoder_layers', 3),
        num_decoder_layers=config.get('model', {}).get('num_decoder_layers', 3),
        k_neighbors=config.get('model', {}).get('k_neighbors', 48),
        dropout=0.0,  # No dropout during evaluation
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"  Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Loss function
    criterion = PLDDTWeightedLoss(
        high_threshold=config.get('plddt', {}).get('high_threshold', 70.0),
        low_threshold=config.get('plddt', {}).get('low_threshold', 50.0),
    )
    
    # Run evaluation
    print("\nRunning evaluation...")
    results = evaluate(model, data_loader, criterion, device)
    
    # Print results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    
    overall = results['overall']
    print(f"\nOverall ({args.split} set):")
    print(f"  Perplexity:    {overall['perplexity']:.3f}")
    print(f"  Recovery:      {overall['recovery']*100:.2f}%")
    print(f"  NLL:           {overall['loss']:.4f}")
    print(f"  N residues:    {overall['n_residues']:,}")
    
    if results['per_domain']:
        print("\nPer-Domain:")
        for domain, metrics in sorted(results['per_domain'].items()):
            print(f"  {domain:8s}: ppl={metrics['perplexity']:.2f}, "
                  f"rec={metrics['recovery']*100:.1f}%, "
                  f"n={metrics['n_residues']:,}")
    
    if results['per_confidence']:
        print("\nPer-Confidence:")
        for bin_name, metrics in sorted(results['per_confidence'].items()):
            print(f"  {bin_name:15s}: ppl={metrics['perplexity']:.2f}, "
                  f"rec={metrics['recovery']*100:.1f}%, "
                  f"n={metrics['n_residues']:,}")
    
    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save overall and per-domain/confidence (without per-sample for readability)
    summary = {
        'checkpoint': str(args.checkpoint),
        'config': str(args.config),
        'split': args.split,
        'overall': results['overall'],
        'per_domain': results['per_domain'],
        'per_confidence': results['per_confidence'],
    }
    
    output_file = args.output_dir / f"evaluation_{args.split}.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save per-sample results separately
    per_sample_file = args.output_dir / f"per_sample_{args.split}.json"
    with open(per_sample_file, 'w') as f:
        json.dump(results['per_sample'], f, indent=2)
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
