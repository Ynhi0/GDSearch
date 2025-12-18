"""CIFAR-10 Experiment Runner

Extracted from run_all_kaggle.py for better modularity.
"""

from typing import Optional, List, Any
import logging
from pathlib import Path

import torch
import pandas as pd

from src.core.config import ExperimentConfig
from src.core.models import ResNet18
from src.core.data_utils import get_cifar10_loaders


def run_cifar10_experiment(
    config: ExperimentConfig,
    exp_profiler: Optional[Any] = None,
    exp_tracker: Optional[Any] = None,
    exp_checkpoint_manager: Optional[Any] = None
) -> Optional[pd.DataFrame]:
    """Run CIFAR-10 benchmark with ResNet-18.
    
    Args:
        config: Experiment configuration
        exp_profiler: Performance profiler instance
        exp_tracker: Experiment tracker instance
        exp_checkpoint_manager: Checkpoint manager instance
        
    Returns:
        DataFrame with results or None if failed
    """
    logging.info("Starting CIFAR-10 experiment...")
    logging.info(f"  Seeds: {config.seeds}")
    logging.info(f"  Quick mode: {config.quick}")
    
    # Import the actual implementation
    from run_all_kaggle import run_cifar10_experiment as _run_cifar10_impl
    
    return _run_cifar10_impl(
        results_dir=str(config.results_dir / "cifar10"),
        seeds=config.seeds,
        quick=config.quick,
        skip_tuning=config.skip_tuning,
        profiler=exp_profiler,
        tracker=exp_tracker,
        checkpoint_manager=exp_checkpoint_manager,
        resume=config.resume
    )
