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

import optuna
from optuna.pruners import MedianPruner, PercentilePruner
from optuna.samplers import TPESampler, RandomSampler, GridSampler
import torch
import numpy as np
from typing import Dict, Any, Callable, Optional, List, Tuple
import json
from pathlib import Path


class OptunaHyperparameterTuner:
    """
    Hyperparameter tuner using Optuna for GDSearch experiments.
    
    Supports automatic search over optimizer and model hyperparameters.
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
        
        # Create sampler
        if sampler == "tpe":
            self.sampler = TPESampler(seed=seed, n_startup_trials=n_startup_trials)
        elif sampler == "random":
            self.sampler = RandomSampler(seed=seed)
        elif sampler == "grid":
            # Grid sampler requires search space upfront
            self.sampler = None  # Will be set when search space is defined
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
        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=self.sampler,
            pruner=self.pruner,
            storage=storage,
            load_if_exists=True
        )
        
    def optimize(
        self,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        show_progress_bar: bool = True,
        callbacks: Optional[List[Callable]] = None
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Args:
            n_trials: Number of trials to run
            timeout: Time limit in seconds (optional)
            show_progress_bar: Show progress bar during optimization
            callbacks: List of callback functions
            
        Returns:
            Dictionary with best parameters and statistics
        """
        print(f"Starting Optuna optimization: {self.study_name}")
        print(f"Direction: {self.direction}")
        print(f"Trials: {n_trials}")
        print(f"Sampler: {self.sampler.__class__.__name__}")
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
        best_trial = self.study.best_trial
        
        results = {
            'best_value': best_trial.value,
            'best_params': best_trial.params,
            'best_trial_number': best_trial.number,
            'n_trials': len(self.study.trials),
            'n_pruned': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            'n_complete': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'study_name': self.study_name
        }
        
        print("\n" + "=" * 80)
        print(f"✅ Optimization Complete!")
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
            importance = optuna.importance.get_param_importances(self.study)
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
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Saved results to {filepath}")


def suggest_optimizer_params(trial: optuna.Trial, optimizer_name: str) -> Dict[str, Any]:
    """
    Suggest hyperparameters for optimizers.
    
    Args:
        trial: Optuna trial object
        optimizer_name: Name of optimizer ("sgd", "adam", "rmsprop", etc.)
        
    Returns:
        Dictionary of suggested hyperparameters
    """
    params = {}
    
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


def suggest_lr_scheduler_params(trial: optuna.Trial, scheduler_name: str, max_epochs: int) -> Dict[str, Any]:
    """
    Suggest hyperparameters for LR schedulers.
    
    Args:
        trial: Optuna trial object
        scheduler_name: Name of scheduler
        max_epochs: Maximum number of training epochs
        
    Returns:
        Dictionary of suggested hyperparameters
    """
    params = {'scheduler': scheduler_name}
    
    if scheduler_name == 'step':
        params['step_size'] = trial.suggest_int('step_size', max_epochs // 10, max_epochs // 2)
        params['gamma'] = trial.suggest_float('gamma', 0.05, 0.5)
    
    elif scheduler_name == 'multistep':
        n_milestones = trial.suggest_int('n_milestones', 2, 4)
        milestones = sorted([
            trial.suggest_int(f'milestone_{i}', max_epochs // 10, max_epochs - 5)
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


def suggest_model_params(trial: optuna.Trial, model_type: str) -> Dict[str, Any]:
    """
    Suggest hyperparameters for models.
    
    Args:
        trial: Optuna trial object
        model_type: Type of model ("mlp", "cnn")
        
    Returns:
        Dictionary of suggested hyperparameters
    """
    params = {}
    
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


def suggest_training_params(trial: optuna.Trial) -> Dict[str, Any]:
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
def plot_optimization_history(study: optuna.Study, save_path: Optional[str] = None):
    """Plot optimization history."""
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        if save_path:
            fig.write_image(save_path)
            print(f"✅ Saved optimization history to {save_path}")
        else:
            fig.show()
    except Exception as e:
        print(f"Could not plot optimization history: {e}")


def plot_param_importances(study: optuna.Study, save_path: Optional[str] = None):
    """Plot parameter importances."""
    try:
        fig = optuna.visualization.plot_param_importances(study)
        if save_path:
            fig.write_image(save_path)
            print(f"✅ Saved parameter importances to {save_path}")
        else:
            fig.show()
    except Exception as e:
        print(f"Could not plot parameter importances: {e}")


def plot_slice(study: optuna.Study, save_path: Optional[str] = None):
    """Plot parameter slice plots."""
    try:
        fig = optuna.visualization.plot_slice(study)
        if save_path:
            fig.write_image(save_path)
            print(f"✅ Saved slice plot to {save_path}")
        else:
            fig.show()
    except Exception as e:
        print(f"Could not plot slice: {e}")


def plot_contour(study: optuna.Study, params: Optional[List[str]] = None, save_path: Optional[str] = None):
    """Plot contour plot of parameter interactions."""
    try:
        fig = optuna.visualization.plot_contour(study, params=params)
        if save_path:
            fig.write_image(save_path)
            print(f"✅ Saved contour plot to {save_path}")
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
    
    # Run optimization
    results = tuner.optimize(n_trials=50, show_progress_bar=True)
    
    print("\n✅ Demo complete!")
    print(f"Optimum found: x={results['best_params']['x']:.4f}, y={results['best_params']['y']:.4f}")
    print(f"Expected optimum: x=2.0, y=-3.0")
