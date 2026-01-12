"""
Experiment runners: single runs, multi-seed, full analysis pipeline.
"""

__all__ = []
try:
    from .run_experiment import run_single_experiment  # noqa: F401
    __all__.append('run_single_experiment')
except Exception as e:
    import logging
    logging.debug("Optional import run_single_experiment failed: %s", e, exc_info=True)

try:
    from .run_multi_seed import run_multi_seed_experiment  # noqa: F401
    __all__.append('run_multi_seed_experiment')
except Exception as e:
    import logging
    logging.debug("Optional import run_multi_seed_experiment failed: %s", e, exc_info=True)

try:
    from .run_full_analysis import run_full_pipeline  # noqa: F401
    __all__.append('run_full_pipeline')
except Exception as e:
    import logging
    logging.debug("Optional import run_full_pipeline failed: %s", e, exc_info=True)

# Note: 'run_nn_experiment' may be available in bundled scripts (e.g., run_all_kaggle.py); import it explicitly where needed.
