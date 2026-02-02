"""
Centralized result filename generation following project conventions.

Canonical format: NN_<Model>_<Dataset>_<Optimizer>_lr<lr>_seed<seed>[_tag].csv

This module ensures consistent filename generation across all experiment scripts
and provides parsing utilities for result analysis.
"""

import warnings
from typing import Optional, Dict, Any
import re


def generate_result_filename(
    model: str,
    dataset: str,
    optimizer: str,
    lr: float,
    seed: int,
    tag: Optional[str] = None
) -> str:
    """
    Generate canonical result filename following project convention.
    
    Format: NN_<Model>_<Dataset>_<Optimizer>_lr<lr>_seed<seed>[_tag].csv
    
    Args:
        model: Model name (e.g., "ResNet18", "SimpleMLP")
        dataset: Dataset name (e.g., "CIFAR10", "MNIST")
        optimizer: Optimizer name (e.g., "Adam", "SGD")
        lr: Learning rate (float)
        seed: Random seed (int)
        tag: Optional tag for experiment variants
        
    Returns:
        Canonical filename string
        
    Examples:
        >>> generate_result_filename("ResNet18", "CIFAR10", "Adam", 0.001, 42)
        'NN_ResNet18_CIFAR10_Adam_lr0.001_seed42.csv'
        
        >>> generate_result_filename("BERT", "IMDB", "Adam", 0.001, 42, "application")
        'NN_BERT_IMDB_Adam_lr0.001_seed42_application.csv'
    """
    base = f"NN_{model}_{dataset}_{optimizer}_lr{lr}_seed{seed}"
    return f"{base}_{tag}.csv" if tag else f"{base}.csv"


def parse_result_filename(filename: str) -> Dict[str, Any]:
    """
    Parse result filename to extract experiment parameters.
    
    Supports both canonical and legacy formats with deprecation warnings.
    
    Args:
        filename: Result filename (with or without .csv extension)
        
    Returns:
        Dictionary with keys: model, dataset, optimizer, lr, seed, tag (optional)
        
    Raises:
        ValueError: If filename doesn't match any known format
        
    Examples:
        >>> parse_result_filename("NN_ResNet18_CIFAR10_Adam_lr0.001_seed42.csv")
        {'model': 'ResNet18', 'dataset': 'CIFAR10', 'optimizer': 'Adam', 
         'lr': 0.001, 'seed': 42, 'tag': None}
    """
    # Remove .csv extension if present
    name = filename.replace('.csv', '')
    
    # Canonical pattern: NN_<Model>_<Dataset>_<Optimizer>_lr<lr>_seed<seed>[_tag]
    canonical_pattern = r'^NN_([^_]+)_([^_]+)_([^_]+)_lr([\d.]+)_seed(\d+)(?:_(.+))?$'
    match = re.match(canonical_pattern, name)
    
    if match:
        model, dataset, optimizer, lr, seed, tag = match.groups()
        return {
            'model': model,
            'dataset': dataset,
            'optimizer': optimizer,
            'lr': float(lr),
            'seed': int(seed),
            'tag': tag
        }
    
    # Try legacy patterns with warning
    legacy_patterns = [
        # Pattern: NN_Simple<Dataset>_<Optimizer>_lr<lr>_seed<seed>
        (r'^NN_Simple([^_]+)_([^_]+)_lr([\d.]+)_seed(\d+)$', 'SimpleDirect'),
        # Pattern: <Model>_<Dataset>_<Optimizer>_lr<lr>_seed<seed> (missing NN_ prefix)
        (r'^([^_]+)_([^_]+)_([^_]+)_lr([\d.]+)_seed(\d+)$', 'NoPrefix'),
    ]
    
    for pattern, format_name in legacy_patterns:
        match = re.match(pattern, name)
        if match:
            warnings.warn(
                f"Legacy filename format detected: {filename} ({format_name}). "
                f"Use generate_result_filename() for new experiments. "
                f"Legacy format support will be removed in v2.0.",
                DeprecationWarning,
                stacklevel=2
            )
            
            groups = match.groups()
            if format_name == 'SimpleDirect':
                # NN_Simple<Dataset>_<Optimizer>_lr<lr>_seed<seed>
                dataset, optimizer, lr, seed = groups
                return {
                    'model': f'Simple{dataset}',
                    'dataset': dataset,
                    'optimizer': optimizer,
                    'lr': float(lr),
                    'seed': int(seed),
                    'tag': None
                }
            elif format_name == 'NoPrefix':
                # <Model>_<Dataset>_<Optimizer>_lr<lr>_seed<seed>
                model, dataset, optimizer, lr, seed = groups
                return {
                    'model': model,
                    'dataset': dataset,
                    'optimizer': optimizer,
                    'lr': float(lr),
                    'seed': int(seed),
                    'tag': None
                }
    
    raise ValueError(
        f"Filename '{filename}' doesn't match canonical format: "
        f"NN_<Model>_<Dataset>_<Optimizer>_lr<lr>_seed<seed>[_tag].csv\n"
        f"Example: NN_ResNet18_CIFAR10_Adam_lr0.001_seed42.csv"
    )


def validate_result_filename(filename: str) -> bool:
    """
    Validate that a filename follows canonical format.
    
    Args:
        filename: Filename to validate
        
    Returns:
        True if canonical format, False otherwise
    """
    try:
        parse_result_filename(filename)
        return True
    except ValueError:
        return False


def get_filename_components(filename: str) -> str:
    """
    Get human-readable description of filename components.
    
    Args:
        filename: Result filename
        
    Returns:
        String description of components
        
    Example:
        >>> get_filename_components("NN_ResNet18_CIFAR10_Adam_lr0.001_seed42.csv")
        'Model: ResNet18, Dataset: CIFAR10, Optimizer: Adam, LR: 0.001, Seed: 42'
    """
    try:
        components = parse_result_filename(filename)
        parts = [
            f"Model: {components['model']}",
            f"Dataset: {components['dataset']}",
            f"Optimizer: {components['optimizer']}",
            f"LR: {components['lr']}",
            f"Seed: {components['seed']}"
        ]
        if components.get('tag'):
            parts.append(f"Tag: {components['tag']}")
        return ', '.join(parts)
    except ValueError as e:
        return f"Invalid filename: {e}"


# Constants for backward compatibility
CANONICAL_FORMAT = "NN_<Model>_<Dataset>_<Optimizer>_lr<lr>_seed<seed>[_tag].csv"
REQUIRED_COMPONENTS = ['model', 'dataset', 'optimizer', 'lr', 'seed']
