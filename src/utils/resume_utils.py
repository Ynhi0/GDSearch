"""
Resume support utilities for run_all_kaggle.py

This module provides helper functions for determining if experiments should be skipped
based on existing results (resume mode) and for validating experiment completion.
"""
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd


def should_skip_experiment(
    experiment_name: str,
    config: Dict[str, Any],
    results_dir: Path,
    resume: bool = False
) -> bool:
    """
    Determine if experiment should be skipped (already completed).
    
    Checks:
    1. Result CSV exists and has expected number of epochs
    2. Result file is not corrupted (can be read)
    3. All required columns are present
    
    Args:
        experiment_name: Name of experiment (for logging)
        config: Experiment configuration dictionary
        results_dir: Base results directory
        resume: If False, never skip (always re-run)
        
    Returns:
        True if experiment should be skipped (already complete), False otherwise
    """
    if not resume:
        return False
    
    # Try to determine result file path
    try:
        from src.utils.result_filename import generate_result_filename
        result_file = results_dir / 'experiments' / experiment_name / generate_result_filename(
            model=config['model'],
            dataset=config['dataset'],
            optimizer=config['optimizer'],
            lr=config['lr'],
            seed=config['seed']
        )
    except Exception as e:
        logging.debug(f"Could not generate result filename for {experiment_name}: {e}")
        return False
    
    if not result_file.exists():
        logging.debug(f"Result file does not exist: {result_file}")
        return False
    
    # Check if result file is complete
    try:
        df = pd.read_csv(result_file)
        expected_epochs = config.get('epochs', 50)
        
        # Check for required columns
        required_cols = ['epoch', 'train_loss', 'test_acc']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logging.warning(
                f"Result file {result_file.name} missing columns: {missing_cols}. Re-running."
            )
            return False
        
        # Check if we have at least the expected number of epochs
        # Note: We use >= to allow for experiments that ran with MORE epochs
        # than currently configured. This is intentional for backward compatibility.
        if len(df) >= expected_epochs:
            logging.info(
                f"Skipping completed experiment: {experiment_name} "
                f"({result_file.name}, {len(df)}/{expected_epochs} epochs)"
            )
            return True
        else:
            logging.info(
                f"Incomplete experiment found: {experiment_name} "
                f"(has {len(df)}/{expected_epochs} epochs). Re-running."
            )
            return False
        
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
        logging.warning(f"Corrupted result file {result_file}: {e}. Re-running experiment.")
        return False
    except (FileNotFoundError, PermissionError) as e:
        logging.error(f"Cannot access result file {result_file}: {e}")
        raise  # Don't hide permission errors
    except Exception as e:
        logging.error(f"Unexpected error reading {result_file}: {e}", exc_info=True)
        return False


def validate_experiment_result(
    result_file: Path,
    expected_epochs: int,
    required_columns: Optional[list] = None
) -> bool:
    """
    Validate that an experiment result file is complete and well-formed.
    
    Args:
        result_file: Path to result CSV file
        expected_epochs: Expected number of epochs
        required_columns: List of required column names
        
    Returns:
        True if result is valid and complete, False otherwise
    """
    if required_columns is None:
        required_columns = ['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc']
    
    if not result_file.exists():
        return False
    
    try:
        df = pd.read_csv(result_file)
        
        # Check columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            logging.warning(f"Result {result_file.name} missing columns: {missing_cols}")
            return False
        
        # Check number of rows
        if len(df) < expected_epochs:
            logging.warning(
                f"Result {result_file.name} has {len(df)}/{expected_epochs} epochs (incomplete)"
            )
            return False
        
        # Check for NaN values in critical columns
        critical_cols = ['epoch', 'test_acc']
        for col in critical_cols:
            if col in df.columns and df[col].isna().any():
                logging.warning(f"Result {result_file.name} has NaN values in {col}")
                return False
        
        return True
        
    except Exception as e:
        logging.error(f"Error validating result {result_file}: {e}")
        return False


def count_completed_experiments(
    experiments: list,
    results_dir: Path,
    expected_epochs: int = 50
) -> Dict[str, int]:
    """
    Count how many experiments are already completed.
    
    Args:
        experiments: List of experiment configurations
        results_dir: Base results directory
        expected_epochs: Expected epochs per experiment
        
    Returns:
        Dictionary with 'completed', 'incomplete', 'total' counts
    """
    completed = 0
    incomplete = 0
    
    for exp_config in experiments:
        exp_name = exp_config.get('name', 'unknown')
        
        # Check if experiment is complete
        if should_skip_experiment(exp_name, exp_config, results_dir, resume=True):
            completed += 1
        else:
            incomplete += 1
    
    return {
        'completed': completed,
        'incomplete': incomplete,
        'total': len(experiments)
    }
