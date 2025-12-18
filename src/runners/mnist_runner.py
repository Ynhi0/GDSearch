"""MNIST Experiment Runner

Extracted from run_all_kaggle.py for better modularity.
"""

from typing import Optional, List, Any
import logging
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd

from src.core.config import ExperimentConfig
from src.core.models import SimpleMLP
from src.core.data_utils import get_mnist_loaders
from src.core.pytorch_optimizers import get_optimizer


def run_mnist_experiment(
    config: ExperimentConfig,
    exp_profiler: Optional[Any] = None,
    exp_tracker: Optional[Any] = None,
    exp_checkpoint_manager: Optional[Any] = None
) -> Optional[pd.DataFrame]:
    """Run MNIST benchmark with multiple optimizers.
    
    Args:
        config: Experiment configuration
        exp_profiler: Performance profiler instance
        exp_tracker: Experiment tracker instance
        exp_checkpoint_manager: Checkpoint manager instance
        
    Returns:
        DataFrame with results or None if failed
    """
    logging.info("Starting MNIST experiment...")
    logging.info(f"  Seeds: {config.seeds}")
    logging.info(f"  Quick mode: {config.quick}")
    logging.info(f"  Resume: {config.resume}")
    
    # Import the actual implementation from run_all_kaggle
    # This is a placeholder - full extraction would copy the implementation
    from run_all_kaggle import run_mnist_experiment as _run_mnist_impl
    
    return _run_mnist_impl(
        results_dir=str(config.results_dir / "mnist"),
        seeds=config.seeds,
        quick=config.quick,
        skip_tuning=config.skip_tuning,
        profiler=exp_profiler,
        tracker=exp_tracker,
        checkpoint_manager=exp_checkpoint_manager,
        resume=config.resume
    )
