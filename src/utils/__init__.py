"""
Utilities package for GDSearch.

This package provides infrastructure utilities for:
- CSV handling with safety and error handling (csv_utils)
- Checkpoint management with atomic saves (checkpoint_utils)
- Parallel experiment execution across multiple GPUs (parallel_experiment_runner)
- File safety and I/O utilities
- Device management and safety
- Configuration loading and validation
- Experiment state management
- Reproducibility utilities

All utilities are designed to be import-safe (no side effects on import) and
support both local and Kaggle environments.

Usage:
    from src.utils.csv_utils import safe_read_csv
    from src.utils.checkpoint_utils import CheckpointManager, create_checkpoint
    from src.utils.parallel_experiment_runner import ParallelExperimentRunner
"""

__all__ = [
    # CSV utilities
    'csv_utils',
    'safe_read_csv',
    'cleanup_empty_csvs',
    
    # Checkpoint utilities
    'checkpoint_utils',
    'CheckpointManager',
    'create_checkpoint',
    'load_checkpoint_safe',
    'save_checkpoint_atomic',
    
    # Parallel execution
    'parallel_experiment_runner',
    'ParallelExperimentRunner',
    'detect_gpu_configuration',
    'run_experiment_on_gpu',
    
    # File safety
    'file_safety',
    'atomic_io',
    
    # Device safety
    'device_safety',
    
    # Configuration
    'config_loader',
    'config_validator',
    
    # Experiment utilities
    'experiment_config',
    'experiment_state',
    'resume_utils',
    
    # Reproducibility
    'reproducibility',
    
    # Analysis utilities
    'metric_aggregation',
    'metric_normalization',
    'convergence_detection',
    
    # Plotting utilities
    'plot_helpers',
    
    # Type guards and safety
    'type_guards',
    'safe_len',
    'sanity_checks',
    
    # Filename utilities
    'filename',
    'result_filename',
    
    # Fairness checking
    'fairness_check',
    'fair_ablation',
    
    # Data utilities
    'dataloader_optimization',
    'transformed_subset',
    'loader_meta',
    
    # Numeric utilities
    'num_utils',
    
    # Constants
    'constants',
    
    # Error handling
    'error_handling_patterns',
    
    # Kaggle-specific
    'kaggle_memory_optimizer',
]

# Note: We intentionally do NOT import anything here to maintain import-safety.
# All imports should be explicit: `from src.utils.csv_utils import safe_read_csv`
# This ensures no side effects occur when importing the package itself.
