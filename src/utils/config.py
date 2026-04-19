"""
Configuration management for PKS-MPNN experiments.

Supports YAML config files with inheritance and command-line overrides.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from copy import deepcopy


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Handle inheritance
    if 'base_config' in config:
        base_path = config_path.parent / config['base_config']
        base_config = load_config(base_path)
        config = merge_configs(base_config, config)
        del config['base_config']
    
    return config


def merge_configs(base: Dict, override: Dict) -> Dict:
    """
    Recursively merge two config dictionaries.
    
    Args:
        base: Base configuration
        override: Overriding configuration
        
    Returns:
        Merged configuration
    """
    result = deepcopy(base)
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = deepcopy(value)
    
    return result


def override_from_args(config: Dict, args: Any) -> Dict:
    """
    Override config values from argparse arguments.
    
    Args:
        config: Base configuration
        args: Parsed arguments
        
    Returns:
        Updated configuration
    """
    config = deepcopy(config)
    
    for key, value in vars(args).items():
        if value is not None:
            # Handle nested keys (e.g., "training.lr" -> config["training"]["lr"])
            keys = key.split('.')
            d = config
            for k in keys[:-1]:
                if k not in d:
                    d[k] = {}
                d = d[k]
            d[keys[-1]] = value
    
    return config


def save_config(config: Dict, output_path: Path):
    """Save configuration to YAML file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


@dataclass
class ExperimentConfig:
    """
    Experiment configuration dataclass.
    
    Provides typed access to configuration values.
    """
    # Experiment info
    name: str = "pks_mpnn_experiment"
    experiment_type: str = "full_module"  # domain_only, full_module, context_aware
    
    # Data configuration
    cif_dir: str = "data/raw"
    annotation_csv: str = "data/annotations/fragments_for_prediction_COREONLY.csv"
    splits_dir: str = "data/splits"
    
    # Cropping configuration
    plddt_high_threshold: float = 70.0
    plddt_low_threshold: float = 50.0
    k_neighbors: int = 48
    domain_type: Optional[str] = None  # For domain_only experiments
    
    # Model configuration
    hidden_dim: int = 128
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    dropout: float = 0.1
    backbone_noise: float = 0.2
    
    # Training configuration
    num_epochs: int = 100
    batch_size: int = 10000  # Tokens per batch
    warmup_steps: int = 4000
    gradient_norm: float = 1.0
    mixed_precision: bool = True
    label_smoothing: float = 0.0
    
    # Fine-tuning configuration
    pretrained_weights: Optional[str] = None
    unfreeze_phase: str = "decoder_only"  # decoder_only, encoder_decoder, full
    
    # Output configuration
    output_dir: str = "outputs"
    save_every: int = 10
    
    # wandb configuration
    wandb_project: Optional[str] = "PKS-MPNN"
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"  # online, offline, disabled
    
    @classmethod
    def from_dict(cls, config: Dict) -> 'ExperimentConfig':
        """Create config from dictionary."""
        # Filter to only known fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config.items() if k in known_fields}
        return cls(**filtered)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items()}
