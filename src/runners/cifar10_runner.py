"""CIFAR-10 Experiment Runner

Extracted from run_all_kaggle.py for better modularity.

Removed circular dependency on root-level run_all_kaggle.py.
This file now serves as a THIN WRAPPER that defers to the main implementation.
For production use, the logic from run_all_kaggle should be moved to src/experiments/.
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

    This is a THIN WRAPPER that avoids circular import.
    The actual implementation lives in run_all_kaggle.py (root level).

    For packaging as a library, the implementation should be moved from run_all_kaggle
    into src/experiments/cifar10_trainer.py, and this wrapper should import from there.

    Args:
        config: Experiment configuration
        exp_profiler: Performance profiler instance
        exp_tracker: Experiment tracker instance
        exp_checkpoint_manager: Checkpoint manager instance

    Returns:
        DataFrame with results or None if failed

    MIGRATION PATH (for library packaging):
        1. Extract run_cifar10_experiment logic from run_all_kaggle.py
        2. Move to src/experiments/cifar10_trainer.py
        3. Replace this import with: from src.experiments.cifar10_trainer import run_cifar10_experiment
    """
    logging.info("Starting CIFAR-10 experiment...")
    logging.info(f"  Seeds: {config.seeds}")
    logging.info(f"  Quick mode: {config.quick}")

    # TEMPORARY WORKAROUND (Issue #31): Import from root for now
    # This breaks library packaging but allows current workflow to continue
    # TODO: Refactor run_all_kaggle.py into src/experiments/ modules
    try:
        import sys
        import os
        # Add root directory to sys.path temporarily
        root_dir = Path(__file__).parent.parent.parent
        if str(root_dir) not in sys.path:
            sys.path.insert(0, str(root_dir))

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
    except ImportError as e:
        logging.error(f"Failed to import run_cifar10_experiment from run_all_kaggle: {e}")
        logging.error("MIGRATION NEEDED: Move run_all_kaggle logic to src/experiments/")
        raise
