"""
Optuna-based Hyperparameter Optimization for GDSearch

Provides automated hyperparameter tuning using Optuna for:
- Optimizer hyperparameters (lr, momentum, betas, weight_decay)
- Model architectures (hidden sizes, dropout rates)
- Training parameters (batch size, learning rate schedules)

Supports:
- Grid search, Random search, TPE (Tree-structured Parzen Estimator)
- Multi-objective optimization
- Pruning of unpromising trials
- Visualization of optimization results
"""

import os
# Optuna is an optional dependency. Import lazily to avoid import-time failures
try:
    import optuna
    from optuna.pruners import MedianPruner, PercentilePruner
    from optuna.samplers import TPESampler, RandomSampler, GridSampler
    HAS_OPTUNA = True
except Exception:
    optuna = None
    MedianPruner = None
    PercentilePruner = None
    TPESampler = None
    RandomSampler = None
    GridSampler = None
    HAS_OPTUNA = False

import torch
import numpy as np
import logging
from typing import Dict, Any, Callable, Optional, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    # Import for static typing only (Optuna may be unavailable at runtime)
    import optuna  # type: ignore
import json
from pathlib import Path


class OptunaHyperparameterTuner:
    """
    Hyperparameter tuner using Optuna for GDSearch experiments.

    NOTE: Use `create_tuner(objective_fn, use_optuna=None, **kwargs)` to obtain a tuner
    instance. Optuna is disabled by default unless the environment variable
    `GDSEARCH_ENABLE_OPTUNA` is set to '1'/'true'/'yes' or `use_optuna=True` is passed.
    When Optuna is disabled this class will still be importable but the factory will
    return a `RandomTuner` fallback instead.
    """

    def __init__(
        self,
        objective_fn: Callable,
        direction: str = "maximize",
        study_name: str = "gdsearch_optimization",
        storage: Optional[str] = None,
        sampler: str = "tpe",
        pruner: Optional[str] = "median",
        n_startup_trials: int = 10,
        seed: int = 42
    ):
        """
        Initialize hyperparameter tuner.

        Args:
            objective_fn: Function to optimize (takes trial, returns metric)
            direction: "maximize" or "minimize"
            study_name: Name for the optimization study
            storage: Database URL for distributed optimization (optional)
            sampler: Sampling algorithm ("tpe", "random", "grid")
            pruner: Pruning algorithm ("median", "percentile", None)
            n_startup_trials: Number of random trials before TPE
            seed: Random seed for reproducibility
        """
        self.objective_fn = objective_fn
        self.direction = direction
        self.study_name = study_name
        self.seed = seed

        # Fail fast if Optuna is not available
        if optuna is None:
            raise RuntimeError(
                "Optuna is not available in this environment. Install it with `pip install optuna` "
                "or call `create_tuner(..., use_optuna=False)` to use the dependency-free RandomTuner fallback."
            )
        # Narrow optional imports for static analyzer: ensure the sampler/pruner factories are available
        assert TPESampler is not None and RandomSampler is not None and GridSampler is not None and MedianPruner is not None and PercentilePruner is not None, "Optuna imports not properly initialized"

        # Create sampler
        if sampler == "tpe":
            self.sampler = TPESampler(seed=seed, n_startup_trials=n_startup_trials)
        elif sampler == "random":
            self.sampler = RandomSampler(seed=seed)
        elif sampler == "grid":
            # Grid sampler requires search space upfront - use TPE as fallback
            logging.warning("GridSampler requires predefined search space. Using TPE sampler instead.")
            self.sampler = TPESampler(seed=seed, n_startup_trials=n_startup_trials)
        else:
            raise ValueError(f"Unknown sampler: {sampler}")

        # Create pruner
        if pruner == "median":
            self.pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        elif pruner == "percentile":
            self.pruner = PercentilePruner(percentile=25.0, n_startup_trials=5)
        elif pruner is None:
            self.pruner = None
        else:
            raise ValueError(f"Unknown pruner: {pruner}")

        # Create study
        # Changed default to load_if_exists=False to prevent contamination
        # Users must explicitly set study_name with timestamp/UUID for shared storage
        # or accept risk of reusing old trials
        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=self.sampler,
            pruner=self.pruner,
            storage=storage,
            load_if_exists=False  # Prevents accidental trial contamination
        )

        if storage is not None:
            logging.warning(
                f"Using shared storage with study_name='{study_name}'. "
                f"To prevent trial contamination, ensure study_name is unique (include timestamp/UUID). "
                f"If you want to resume an existing study, manually set load_if_exists=True in create_study() call."
            )

    def optimize(
        self,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        show_progress_bar: bool = True,
        callbacks: Optional[List[Callable]] = None,
        val_loader = None,  # CRITICAL: Validation loader parameter for test-leakage checks
        test_dataset = None,  # Reference test dataset for identity check
        enforce_validation: bool = True  # FIXED: New parameter to enforce validation loader requirement
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.

        Args:
            n_trials: Number of trials to run
            timeout: Time limit in seconds (optional)
            show_progress_bar: Show progress bar during optimization
            callbacks: List of callback functions
            val_loader: Validation DataLoader (required for test-leakage checks when enforce_validation=True)
            test_dataset: Reference to test dataset for identity validation (RECOMMENDED)
            enforce_validation: If True, raises error if val_loader is None (default: True to enforce validation)

        Returns:
            Dictionary with best parameters and statistics

        Raises:
            ValueError: If enforce_validation=True and val_loader is None or lacks proper metadata
            RuntimeError: If validation loader fails test-leakage check
        """
        # FIXED: Make validation loader mandatory by default to prevent test-leakage
        if val_loader is None:
            if enforce_validation:
                raise ValueError(
                    "INTEGRITY ERROR: No validation loader provided to OptunaHyperparameterTuner.optimize().\n"
                    "\nTest-leakage prevention requires a validation loader to verify that the test set is not used during tuning.\n"
                    "\nREMEDIATION OPTIONS:\n"
                    "  1. RECOMMENDED: Use create_validated_loaders() from src.core.loader_validation:\n"
                    "     train_loader, val_loader, test_loader = create_validated_loaders(get_mnist_loaders, val_split=0.15, batch_size=128)\n"
                    "  2. Tag your existing loader: setattr(your_val_loader, '_split_type', 'validation')\n"
                    "  3. NOT RECOMMENDED: Set enforce_validation=False (invalidates research claims)\n"
                    "\nSee docs: src/core/loader_validation.py for examples."
                )
            else:
                logging.warning(
                    "INTEGRITY WARNING: No validation loader provided and enforce_validation=False. "
                    "Cannot enforce test-leakage prevention. Ensure you are not using the test set for tuning."
                )
        else:
            # Enforce test-leakage prevention with stricter checks
            try:
                from src.core.loader_validation import enforce_no_test_in_tuning, validate_loader_for_tuning

                # Use validate_loader_for_tuning with test_dataset for stronger checks
                if test_dataset is not None:
                    validate_loader_for_tuning(val_loader, expected_split='validation', test_dataset=test_dataset)
                    logging.info("PASSED: Validation loader test-leakage check (with dataset identity verification)")
                else:
                    # Fallback to metadata-only check, but require proper tagging
                    split_type = getattr(val_loader, '_split_type', None)
                    loader_name = getattr(val_loader, 'name', None)

                    if split_type != 'validation' and 'val' not in str(loader_name).lower():
                        raise ValueError(
                            "INTEGRITY ERROR: Validation loader lacks proper metadata. "
                            f"Expected _split_type='validation' or name containing 'val', got split_type={split_type}, name={loader_name}. "
                            "Either: (1) provide test_dataset parameter for identity check, or (2) ensure loader has proper metadata tags. "
                            "This strict check prevents accidental test-set leakage during hyperparameter tuning."
                        )

                    enforce_no_test_in_tuning(val_loader)
                    logging.warning("PASSED: Validation loader test-leakage check (metadata-only; consider providing test_dataset for stronger verification)")
            except ImportError:
                logging.error("Could not import validation utilities. Tuning cannot proceed safely.")
                raise RuntimeError("Missing src.core.loader_validation module required for test-leakage prevention.")
            except Exception as e:
                logging.error(f"CRITICAL: Validation loader failed test-leakage check: {e}")
                raise RuntimeError(
                    f"Cannot start hyperparameter tuning: validation loader failed test-leakage check. "
                    f"This indicates that the test set may be used during tuning, which would invalidate results. "
                    f"Error: {e}"
                ) from e

        print(f"Starting Optuna optimization: {self.study_name}")
        print(f"Direction: {self.direction}")
        print(f"Trials: {n_trials}")
        print(f"Sampler: {self.sampler.__class__.__name__ if self.sampler else 'None'}")
        print(f"Pruner: {self.pruner.__class__.__name__ if self.pruner else 'None'}")
        print("-" * 80)

        # Run optimization
        self.study.optimize(
            self.objective_fn,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=show_progress_bar,
            callbacks=callbacks
        )

        # Get best trial
        best_trial = self.study.best_trial  # type: ignore[union-attr]

        if best_trial is None:
            raise RuntimeError("No trials completed successfully. Cannot determine best trial.")

        results = {
            'best_value': best_trial.value,
            'best_params': best_trial.params,
            'best_trial_number': best_trial.number,
            'n_trials': len(self.study.trials),
            'n_pruned': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]),  # type: ignore[union-attr]
            'n_complete': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]),  # type: ignore[union-attr]
            'study_name': self.study_name
        }

        print("\n" + "=" * 80)
        print(f"Optimization Complete!")
        print(f"Best value: {results['best_value']:.6f}")
        print(f"Best trial: #{results['best_trial_number']}")
        print(f"Total trials: {results['n_trials']} ({results['n_complete']} complete, {results['n_pruned']} pruned)")
        print("\nBest parameters:")
        for param, value in results['best_params'].items():
            print(f"  {param}: {value}")
        print("=" * 80)

        return results

    def get_importance(self) -> Dict[str, float]:
        """Get parameter importance scores."""
        try:
            importance = optuna.importance.get_param_importances(self.study)  # type: ignore[attr-defined]
            return importance
        except Exception as e:
            print(f"Could not compute importance: {e}")
            return {}

    def save_results(self, filepath: str):
        """Save optimization results to JSON."""
        results = {
            'study_name': self.study_name,
            'direction': self.direction,
            'best_value': self.study.best_value,
            'best_params': self.study.best_params,
            'best_trial': self.study.best_trial.number,
            'n_trials': len(self.study.trials),
            'all_trials': [
                {
                    'number': t.number,
                    'value': t.value,
                    'params': t.params,
                    'state': str(t.state)
                }
                for t in self.study.trials
            ]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

        print(f"Saved results to {filepath}")


def suggest_optimizer_params(trial: Any, optimizer_name: str) -> Dict[str, Any]:
    """
    Suggest hyperparameters for optimizers.

    Args:
        trial: Optuna trial object
        optimizer_name: Name of optimizer ("sgd", "adam", "rmsprop", etc.)

    Returns:
        Dictionary of suggested hyperparameters
    """
    params: Dict[str, Any] = {}

    # Learning rate (universal)
    params['lr'] = trial.suggest_float('lr', 1e-5, 1e-1, log=True)

    if optimizer_name.lower() in ['sgd', 'sgdmomentum']:
        if 'momentum' in optimizer_name.lower():
            params['momentum'] = trial.suggest_float('momentum', 0.0, 0.99)

    elif optimizer_name.lower() == 'adam':
        params['beta1'] = trial.suggest_float('beta1', 0.8, 0.999)
        params['beta2'] = trial.suggest_float('beta2', 0.9, 0.9999)
        params['epsilon'] = trial.suggest_float('epsilon', 1e-10, 1e-6, log=True)

    elif optimizer_name.lower() == 'adamw':
        params['beta1'] = trial.suggest_float('beta1', 0.8, 0.999)
        params['beta2'] = trial.suggest_float('beta2', 0.9, 0.9999)
        params['epsilon'] = trial.suggest_float('epsilon', 1e-10, 1e-6, log=True)
        params['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

    elif optimizer_name.lower() == 'rmsprop':
        params['alpha'] = trial.suggest_float('alpha', 0.9, 0.999)
        params['epsilon'] = trial.suggest_float('epsilon', 1e-10, 1e-6, log=True)

    return params


def suggest_lr_scheduler_params(trial: Any, scheduler_name: str, max_epochs: int) -> Dict[str, Any]:
    """
    Suggest hyperparameters for LR schedulers.

    Args:
        trial: Optuna trial object
        scheduler_name: Name of scheduler
        max_epochs: Maximum number of training epochs

    Returns:
        Dictionary of suggested hyperparameters
    """
    params: Dict[str, Any] = {'scheduler': scheduler_name}

    if scheduler_name == 'step':
        # Guard against invalid ranges when max_epochs is too small
        if max_epochs < 3:
            # Too few epochs for meaningful stepping
            params['step_size'] = 1
            params['gamma'] = 0.1
        else:
            step_min = max(1, max_epochs // 10)
            step_max = max(step_min + 1, max_epochs // 2)
            params['step_size'] = trial.suggest_int('step_size', step_min, step_max)
            params['gamma'] = trial.suggest_float('gamma', 0.05, 0.5)

    elif scheduler_name == 'multistep':
        # Guard against invalid milestone ranges
        if max_epochs < 10:
            # Too few epochs for milestones - use simple defaults
            params['milestones'] = [max_epochs // 2] if max_epochs >= 2 else [1]
            params['gamma'] = 0.1
        else:
            n_milestones = trial.suggest_int('n_milestones', 2, min(4, max_epochs - 1))
            milestone_min = max(1, max_epochs // 10)
            milestone_max = max(milestone_min + n_milestones, max_epochs - 5)
            if milestone_max <= milestone_min:
                milestone_max = max_epochs - 1
            milestones = sorted([
                trial.suggest_int(f'milestone_{i}', milestone_min, milestone_max)
                for i in range(n_milestones)
            ])
            params['milestones'] = milestones
            params['gamma'] = trial.suggest_float('gamma', 0.05, 0.5)

    elif scheduler_name == 'cosine':
        params['T_max'] = max_epochs
        params['eta_min'] = trial.suggest_float('eta_min', 1e-6, 1e-4, log=True)

    elif scheduler_name == 'exponential':
        params['gamma'] = trial.suggest_float('gamma', 0.90, 0.99)

    elif scheduler_name == 'onecycle':
        params['max_lr'] = trial.suggest_float('max_lr', 1e-3, 1e-1, log=True)
        params['total_steps'] = max_epochs
        params['pct_start'] = trial.suggest_float('pct_start', 0.2, 0.4)

    # Optional warmup
    use_warmup = trial.suggest_categorical('use_warmup', [True, False])
    if use_warmup:
        params['warmup_epochs'] = trial.suggest_int('warmup_epochs', 3, min(10, max_epochs // 5))

    return params


def suggest_model_params(trial: Any, model_type: str) -> Dict[str, Any]:
    """
    Suggest hyperparameters for models.

    Args:
        trial: Optuna trial object
        model_type: Type of model ("mlp", "cnn")

    Returns:
        Dictionary of suggested hyperparameters
    """
    params: Dict[str, Any] = {}

    if model_type == 'mlp':
        n_layers = trial.suggest_int('n_layers', 1, 4)
        hidden_sizes = []
        for i in range(n_layers):
            size = trial.suggest_categorical(f'hidden_size_{i}', [64, 128, 256, 512])
            hidden_sizes.append(size)
        params['hidden_sizes'] = hidden_sizes
        params['dropout'] = trial.suggest_float('dropout', 0.0, 0.5)

    elif model_type == 'cnn':
        n_conv_layers = trial.suggest_int('n_conv_layers', 2, 4)
        channels = []
        for i in range(n_conv_layers):
            ch = trial.suggest_categorical(f'channels_{i}', [32, 64, 128, 256])
            channels.append(ch)
        params['channels'] = channels
        params['dropout'] = trial.suggest_float('dropout', 0.0, 0.5)
        params['kernel_size'] = trial.suggest_categorical('kernel_size', [3, 5])

    return params


# ---------------------------
# Optional/Opt-in Tuner API
# ---------------------------

class _RandomTrial:
    """Lightweight trial-like object with suggest_* methods supported by common objective functions."""
    def __init__(self):
        self.params = {}

    def suggest_float(self, name, low, high, log=False):
        if log:
            val = np.exp(np.random.uniform(np.log(low), np.log(high)))
        else:
            val = float(np.random.uniform(low, high))
        self.params[name] = val
        return val

    def suggest_int(self, name, low, high):
        val = int(np.random.randint(low, high + 1))
        self.params[name] = val
        return val

    def suggest_categorical(self, name, choices):
        val = choices[int(np.random.randint(0, len(choices)))]
        self.params[name] = val
        return val


class RandomTuner:
    """A simple random-search tuner that mimics the basic Optuna interface for objective functions.

    Use this tuner when Optuna is explicitly disabled (default). It provides a safe, dependency-free
    fallback so tuning remains possible without Optuna.
    """

    def __init__(self, objective_fn: Callable, direction: str = "maximize", seed: int = 42):
        self.objective_fn = objective_fn
        self.direction = direction
        self.seed = seed
        np.random.seed(seed)

    def optimize(self, n_trials: int = 50, timeout: Optional[int] = None, show_progress_bar: bool = False,
                 callbacks: Optional[List[Callable]] = None, val_loader=None, test_dataset=None, enforce_validation: bool = True) -> Dict[str, Any]:
        best_value = None
        best_params = None
        trials = []

        for i in range(n_trials):
            trial = _RandomTrial()
            try:
                val = self.objective_fn(trial)
            except Exception as e:
                logging.warning("RandomTuner: objective function failed on trial %d: %s", i, e)
                continue

            trials.append({'number': i, 'value': val, 'params': trial.params})

            if best_value is None:
                best_value = val
                best_params = trial.params
            else:
                improved = (val > best_value) if self.direction == 'maximize' else (val < best_value)
                if improved:
                    best_value = val
                    best_params = trial.params

        results = {
            'best_value': best_value,
            'best_params': best_params or {},
            'n_trials': len(trials),
            'n_pruned': 0,
            'n_complete': len(trials),
            'study_name': 'random_fallback'
        }

        return results


def create_tuner(objective_fn: Callable, use_optuna: Optional[bool] = None, **kwargs):
    """Factory: create an Optuna tuner if the environment opts-in; otherwise return RandomTuner.

    Preference resolution:
      - If use_optuna is not None, respect it (True -> try to create Optuna tuner, False -> RandomTuner)
      - Else, check env var `GDSEARCH_ENABLE_OPTUNA` (case-insensitive). If set to '1'/'true'/'yes', try Optuna.
      - If Optuna is requested but not importable, raise RuntimeError instructing how to install.
    """
    # Determine opt-in flag
    if use_optuna is None:
        flag = os.environ.get('GDSEARCH_ENABLE_OPTUNA', '').lower() in ('1', 'true', 'yes')
    else:
        flag = bool(use_optuna)

    if flag:
        # Try to instantiate Optuna tuner (may raise if optuna not available)
        try:
            tuner = OptunaHyperparameterTuner(objective_fn=objective_fn, **kwargs)
            return tuner
        except Exception as e:
            raise RuntimeError(
                "Optuna requested but could not be initialized. Install optuna and retry: `pip install optuna`. "
                f"Underlying error: {e}"
            )
    else:
        # Return a lightweight fallback
        return RandomTuner(objective_fn=objective_fn, direction=kwargs.get('direction', 'maximize'), seed=kwargs.get('seed', 42))


def _deep_merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Return a new dict that is a deep merge of a and b (b takes precedence).

    Rules:
      - If both values are dicts, merge recursively.
      - Otherwise, b's value overrides a's value.
      - Does not mutate inputs.
    """
    result = {} if a is None else dict(a)
    for k, v in (b or {}).items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge_dicts(result[k], v)
        else:
            result[k] = v
    return result


def apply_best_params_to_config(config: Dict[str, Any], best_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge Optuna `best_params` into an experiment `config` dictionary.

    - Performs a deep merge so nested configuration sections are preserved and
      only the suggested keys are overwritten.
    - Normalizes optimizer names using `src.core.optimizer_registry.normalize_optimizer_name` if present.
    - Returns a new merged dict (does not mutate inputs).
    """
    base = config.copy() if isinstance(config, dict) else dict(config)
    merged = _deep_merge_dicts(base, best_params or {})

    # Normalize optimizer name if present
    if 'optimizer' in merged and isinstance(merged['optimizer'], str):
        try:
            from src.core.optimizer_registry import normalize_optimizer_name
            merged['optimizer'] = normalize_optimizer_name(merged['optimizer'])
        except Exception:
            # If normalization fails, keep original name but log for debugging
            logging.debug("Could not normalize optimizer name: %s", merged.get('optimizer'))

    return merged


def suggest_training_params(trial: Any) -> Dict[str, Any]:
    """
    Suggest training hyperparameters.

    Args:
        trial: Optuna trial object

    Returns:
        Dictionary of suggested hyperparameters
    """
    params = {
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
        'epochs': trial.suggest_int('epochs', 10, 50)
    }

    return params


# Visualization utilities
from typing import Any

def plot_optimization_history(study: Any, save_path: Optional[str] = None):
    """Plot optimization history."""
    try:
        fig = optuna.visualization.plot_optimization_history(study)  # type: ignore[attr-defined]
        if save_path:
            fig.write_image(str(save_path))
            print(f"Saved optimization history to {save_path}")
        else:
            fig.show()
    except Exception as e:
        print(f"Could not plot optimization history: {e}")


def plot_param_importances(study: Any, save_path: Optional[str] = None):
    """Plot parameter importances."""
    try:
        fig = optuna.visualization.plot_param_importances(study)  # type: ignore[attr-defined]
        if save_path:
            fig.write_image(str(save_path))
            print(f"Saved parameter importances to {save_path}")
        else:
            fig.show()
    except Exception as e:
        print(f"Could not plot parameter importances: {e}")


def plot_slice(study: Any, save_path: Optional[str] = None):
    """Plot parameter slice plots."""
    try:
        fig = optuna.visualization.plot_slice(study)  # type: ignore[attr-defined]
        if save_path:
            fig.write_image(str(save_path))
            print(f"Saved slice plot to {save_path}")
        else:
            fig.show()
    except Exception as e:
        print(f"Could not plot slice: {e}")


def plot_contour(study: Any, params: Optional[List[str]] = None, save_path: Optional[str] = None):
    """Plot contour plot of parameter interactions."""
    try:
        fig = optuna.visualization.plot_contour(study, params=params)  # type: ignore[attr-defined]
        if save_path:
            fig.write_image(str(save_path))
            print(f"Saved contour plot to {save_path}")
        else:
            fig.show()
    except Exception as e:
        print(f"Could not plot contour: {e}")


if __name__ == '__main__':
    # Demo: Simple optimization example
    print("="*80)
    print(" "*25 + "OPTUNA DEMO")
    print("="*80)

    def simple_objective(trial):
        """Simple quadratic objective for testing."""
        x = trial.suggest_float('x', -10, 10)
        y = trial.suggest_float('y', -10, 10)
        return (x - 2)**2 + (y + 3)**2

    # Create tuner
    tuner = OptunaHyperparameterTuner(
        objective_fn=simple_objective,
        direction="minimize",
        study_name="demo_optimization",
        sampler="tpe",
        pruner=None
    )

    # Run optimization (demo mode: disable strict validation enforcement)
    results = tuner.optimize(n_trials=50, show_progress_bar=True, enforce_validation=False)

    print("\nDemo complete!")
    print(f"Optimum found: x={results['best_params']['x']:.4f}, y={results['best_params']['y']:.4f}")
    print(f"Expected optimum: x=2.0, y=-3.0")
