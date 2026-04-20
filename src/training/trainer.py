"""
Training loop for PKS-MPNN with wandb integration.

Supports:
- Single GPU training (A40)
- Checkpointing and resumption
- wandb logging of metrics
- Gradual unfreezing
"""

import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from ..model.protein_mpnn import ProteinMPNNWrapper
from ..model.loss import PLDDTWeightedLoss, compute_nll_loss
from ..model.metrics import (
    compute_perplexity, 
    compute_recovery,
    PerDomainMetrics,
    ConfidenceStratifiedMetrics,
    log_metrics_to_wandb
)
from .optimizer import NoamScheduler, get_optimizer


class PKSTrainer:
    """
    Trainer class for PKS-MPNN fine-tuning.
    
    Handles the training loop, validation, checkpointing, and wandb logging.
    """
    
    def __init__(
        self,
        model: ProteinMPNNWrapper,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        output_dir: Path,
        device: str = "cuda",
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        wandb_entity: Optional[str] = None,
    ):
        """
        Initialize the trainer.
        
        Args:
            model: ProteinMPNN model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            output_dir: Directory for checkpoints and logs
            device: Device to train on
            wandb_project: wandb project name (optional)
            wandb_run_name: wandb run name (optional)
            wandb_entity: wandb entity/team name (optional, for team projects)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = Path(output_dir)
        self.device = device
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        
        # Loss function
        self.criterion = PLDDTWeightedLoss(
            high_threshold=config.get('plddt_high_threshold', 70.0),
            low_threshold=config.get('plddt_low_threshold', 50.0),
            label_smoothing=config.get('label_smoothing', 0.0)
        )
        
        # Optimizer
        self.optimizer = get_optimizer(
            self.model.parameters(),
            d_model=config.get('hidden_dim', 128),
            warmup_steps=config.get('warmup_steps', 4000),
            use_noam=config.get('use_noam', True)
        )
        
        # Mixed precision training
        self.use_amp = config.get('mixed_precision', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Gradient clipping
        self.gradient_norm = config.get('gradient_norm', 1.0)
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Metrics tracking
        self.per_domain_metrics = PerDomainMetrics()
        self.confidence_metrics = ConfidenceStratifiedMetrics()
        
        # Initialize wandb
        self.use_wandb = WANDB_AVAILABLE and wandb_project is not None
        if self.use_wandb:
            init_kwargs = dict(
                project=wandb_project,
                name=wandb_run_name,
                config=config,
                dir=str(self.output_dir)
            )
            if wandb_entity:
                init_kwargs["entity"] = wandb_entity
            wandb.init(**init_kwargs)
            wandb.watch(self.model, log_freq=100)
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dict of training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0.0
        total_residues = 0.0
        
        self.per_domain_metrics.reset()
        self.confidence_metrics.reset()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    log_probs = self.model(
                        batch['X'],
                        batch['S'],
                        batch['mask'],
                        batch['chain_M'],
                        batch['residue_idx'],
                        batch['chain_encoding']
                    )
                    
                    loss, metrics = self.criterion(
                        log_probs,
                        batch['S'],
                        batch['mask'],
                        plddt=batch.get('plddt'),
                        loss_mask=batch.get('mask_for_loss')
                    )
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                
                if self.gradient_norm > 0:
                    self.scaler.unscale_(self.optimizer.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.gradient_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                log_probs = self.model(
                    batch['X'],
                    batch['S'],
                    batch['mask'],
                    batch['chain_M'],
                    batch['residue_idx'],
                    batch['chain_encoding']
                )
                
                loss, metrics = self.criterion(
                    log_probs,
                    batch['S'],
                    batch['mask'],
                    plddt=batch.get('plddt'),
                    loss_mask=batch.get('mask_for_loss')
                )
                
                loss.backward()
                
                if self.gradient_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_norm
                    )
                
                self.optimizer.step()
            
            # Update metrics
            total_loss += metrics['loss'].item() * metrics['n_residues'].item()
            total_correct += metrics['accuracy'].item() * metrics['n_residues'].item()
            total_residues += metrics['n_residues'].item()
            
            # Per-domain and confidence metrics
            if 'domain_labels' in batch:
                self.per_domain_metrics.update(
                    log_probs.detach(),
                    batch['S'],
                    batch['mask'],
                    batch['domain_labels']
                )
            
            if 'plddt' in batch:
                self.confidence_metrics.update(
                    log_probs.detach(),
                    batch['S'],
                    batch['mask'],
                    batch['plddt']
                )
            
            # Update progress bar
            avg_loss = total_loss / (total_residues + 1e-8)
            avg_acc = total_correct / (total_residues + 1e-8)
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'ppl': f'{np.exp(avg_loss):.2f}',
                'acc': f'{avg_acc:.3f}'
            })
            
            self.global_step += 1
            
            # Log to wandb
            if self.use_wandb and batch_idx % 50 == 0:
                wandb.log({
                    'train/batch_loss': metrics['loss'].item(),
                    'train/batch_perplexity': metrics['perplexity'].item(),
                    'train/batch_accuracy': metrics['accuracy'].item(),
                    'train/lr': self.optimizer._rate if hasattr(self.optimizer, '_rate') else 0,
                }, step=self.global_step)
        
        # Compute epoch metrics
        epoch_loss = total_loss / (total_residues + 1e-8)
        epoch_acc = total_correct / (total_residues + 1e-8)
        epoch_ppl = np.exp(epoch_loss)
        
        metrics = {
            'loss': epoch_loss,
            'perplexity': epoch_ppl,
            'accuracy': epoch_acc,
            'n_residues': total_residues,
        }
        
        # Log epoch metrics
        if self.use_wandb:
            log_metrics_to_wandb(
                metrics,
                per_domain=self.per_domain_metrics.compute(),
                per_confidence=self.confidence_metrics.compute(),
                prefix='train',
                step=self.global_step
            )
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Run validation.
        
        Returns:
            Dict of validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0.0
        total_residues = 0.0
        
        self.per_domain_metrics.reset()
        self.confidence_metrics.reset()
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            log_probs = self.model(
                batch['X'],
                batch['S'],
                batch['mask'],
                batch['chain_M'],
                batch['residue_idx'],
                batch['chain_encoding']
            )
            
            loss, metrics = self.criterion(
                log_probs,
                batch['S'],
                batch['mask'],
                plddt=batch.get('plddt'),
                loss_mask=batch.get('mask_for_loss')
            )
            
            # Update metrics
            total_loss += metrics['loss'].item() * metrics['n_residues'].item()
            total_correct += metrics['accuracy'].item() * metrics['n_residues'].item()
            total_residues += metrics['n_residues'].item()
            
            # Per-domain and confidence metrics
            if 'domain_labels' in batch:
                self.per_domain_metrics.update(
                    log_probs,
                    batch['S'],
                    batch['mask'],
                    batch['domain_labels']
                )
            
            if 'plddt' in batch:
                self.confidence_metrics.update(
                    log_probs,
                    batch['S'],
                    batch['mask'],
                    batch['plddt']
                )
        
        # Compute validation metrics
        val_loss = total_loss / (total_residues + 1e-8)
        val_acc = total_correct / (total_residues + 1e-8)
        val_ppl = np.exp(val_loss)
        
        metrics = {
            'loss': val_loss,
            'perplexity': val_ppl,
            'accuracy': val_acc,
            'n_residues': total_residues,
        }
        
        # Log validation metrics
        if self.use_wandb:
            log_metrics_to_wandb(
                metrics,
                per_domain=self.per_domain_metrics.compute(),
                per_confidence=self.confidence_metrics.compute(),
                prefix='val',
                step=self.global_step
            )
        
        return metrics
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if hasattr(self.optimizer, 'state_dict') else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }
        
        path = self.output_dir / "checkpoints" / filename
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = self.output_dir / "checkpoints" / "best_model.pt"
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if checkpoint.get('optimizer_state_dict') and hasattr(self.optimizer, 'load_state_dict'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"Loaded checkpoint from epoch {self.epoch}")
    
    def train(self, num_epochs: int, save_every: int = 10):
        """
        Run full training loop.
        
        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
        """
        print(f"Starting training for {num_epochs} epochs")
        print(f"Output directory: {self.output_dir}")
        
        for _ in range(num_epochs):
            self.epoch += 1
            
            # Train epoch
            train_metrics = self.train_epoch()
            print(f"\nEpoch {self.epoch} Train: "
                  f"loss={train_metrics['loss']:.4f}, "
                  f"ppl={train_metrics['perplexity']:.2f}, "
                  f"acc={train_metrics['accuracy']:.3f}")
            
            # Validate
            val_metrics = self.validate()
            print(f"Epoch {self.epoch} Val: "
                  f"loss={val_metrics['loss']:.4f}, "
                  f"ppl={val_metrics['perplexity']:.2f}, "
                  f"acc={val_metrics['accuracy']:.3f}")
            
            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            
            if self.epoch % save_every == 0 or is_best:
                self.save_checkpoint(
                    f"epoch_{self.epoch}.pt",
                    is_best=is_best
                )
            
            # Always save latest
            self.save_checkpoint("latest.pt")
        
        print(f"\nTraining complete! Best val loss: {self.best_val_loss:.4f}")
        
        if self.use_wandb:
            wandb.finish()
