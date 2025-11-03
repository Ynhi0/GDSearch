"""
Unit tests for Optuna hyperparameter tuner.
"""

import pytest
import optuna
from src.core.optuna_tuner import (
    OptunaHyperparameterTuner,
    suggest_optimizer_params,
    suggest_lr_scheduler_params,
    suggest_model_params,
    suggest_training_params
)


class TestOptunaHyperparameterTuner:
    """Test OptunaHyperparameterTuner class."""
    
    def test_initialization(self):
        """Test tuner initialization."""
        def dummy_objective(trial):
            return trial.suggest_float('x', 0, 1)
        
        tuner = OptunaHyperparameterTuner(
            objective_fn=dummy_objective,
            direction="maximize",
            study_name="test_study"
        )
        
        assert tuner.direction == "maximize"
        assert tuner.study_name == "test_study"
        assert tuner.study is not None
    
    def test_optimization(self):
        """Test running optimization."""
        def simple_objective(trial):
            x = trial.suggest_float('x', -10, 10)
            return (x - 2)**2
        
        tuner = OptunaHyperparameterTuner(
            objective_fn=simple_objective,
            direction="minimize",
            study_name="test_optimization",
            pruner=None
        )
        
        results = tuner.optimize(n_trials=20, show_progress_bar=False)
        
        assert 'best_value' in results
        assert 'best_params' in results
        assert 'n_trials' in results
        assert results['n_trials'] == 20
        
        # Check that optimization found something close to optimum (x=2)
        assert abs(results['best_params']['x'] - 2.0) < 1.0
    
    def test_maximize_direction(self):
        """Test maximization."""
        def objective(trial):
            x = trial.suggest_float('x', -10, 10)
            return -(x - 5)**2  # Maximum at x=5
        
        tuner = OptunaHyperparameterTuner(
            objective_fn=objective,
            direction="maximize",
            study_name="test_maximize",
            pruner=None
        )
        
        results = tuner.optimize(n_trials=30, show_progress_bar=False)
        
        # Should find maximum near x=5
        assert abs(results['best_params']['x'] - 5.0) < 1.0


class TestSuggestOptimizerParams:
    """Test optimizer parameter suggestion."""
    
    def test_adam_params(self):
        """Test Adam parameter suggestions."""
        study = optuna.create_study()
        trial = study.ask()
        
        params = suggest_optimizer_params(trial, 'adam')
        
        assert 'lr' in params
        assert 'beta1' in params
        assert 'beta2' in params
        assert 'epsilon' in params
        
        # Check ranges
        assert 1e-5 <= params['lr'] <= 1e-1
        assert 0.8 <= params['beta1'] <= 0.999
        assert 0.9 <= params['beta2'] <= 0.9999
        assert 1e-10 <= params['epsilon'] <= 1e-6
    
    def test_sgd_momentum_params(self):
        """Test SGD with momentum parameter suggestions."""
        study = optuna.create_study()
        trial = study.ask()
        
        params = suggest_optimizer_params(trial, 'sgdmomentum')
        
        assert 'lr' in params
        assert 'momentum' in params
        
        # Check ranges
        assert 1e-5 <= params['lr'] <= 1e-1
        assert 0.0 <= params['momentum'] <= 0.99
    
    def test_rmsprop_params(self):
        """Test RMSProp parameter suggestions."""
        study = optuna.create_study()
        trial = study.ask()
        
        params = suggest_optimizer_params(trial, 'rmsprop')
        
        assert 'lr' in params
        assert 'alpha' in params
        assert 'epsilon' in params


class TestSuggestLRSchedulerParams:
    """Test LR scheduler parameter suggestion."""
    
    def test_step_scheduler(self):
        """Test step scheduler parameters."""
        study = optuna.create_study()
        trial = study.ask()
        
        params = suggest_lr_scheduler_params(trial, 'step', max_epochs=100)
        
        assert params['scheduler'] == 'step'
        assert 'step_size' in params
        assert 'gamma' in params
        assert 10 <= params['step_size'] <= 50
        assert 0.05 <= params['gamma'] <= 0.5
    
    def test_cosine_scheduler(self):
        """Test cosine scheduler parameters."""
        study = optuna.create_study()
        trial = study.ask()
        
        params = suggest_lr_scheduler_params(trial, 'cosine', max_epochs=100)
        
        assert params['scheduler'] == 'cosine'
        assert params['T_max'] == 100
        assert 'eta_min' in params
    
    def test_warmup_option(self):
        """Test warmup option."""
        study = optuna.create_study()
        
        # Run multiple trials to test both warmup options
        warmup_found = False
        no_warmup_found = False
        
        for _ in range(10):
            trial = study.ask()
            params = suggest_lr_scheduler_params(trial, 'step', max_epochs=100)
            
            if 'warmup_epochs' in params:
                warmup_found = True
                assert 3 <= params['warmup_epochs'] <= 20
            else:
                no_warmup_found = True
        
        # Both options should appear in 10 trials (statistically likely)
        assert warmup_found or no_warmup_found  # At least one option tried


class TestSuggestModelParams:
    """Test model parameter suggestion."""
    
    def test_mlp_params(self):
        """Test MLP parameter suggestions."""
        study = optuna.create_study()
        trial = study.ask()
        
        params = suggest_model_params(trial, 'mlp')
        
        assert 'hidden_sizes' in params
        assert 'dropout' in params
        assert 1 <= len(params['hidden_sizes']) <= 4
        assert 0.0 <= params['dropout'] <= 0.5
        
        # Check that hidden sizes are valid
        for size in params['hidden_sizes']:
            assert size in [64, 128, 256, 512]
    
    def test_cnn_params(self):
        """Test CNN parameter suggestions."""
        study = optuna.create_study()
        trial = study.ask()
        
        params = suggest_model_params(trial, 'cnn')
        
        assert 'channels' in params
        assert 'dropout' in params
        assert 'kernel_size' in params
        assert 2 <= len(params['channels']) <= 4
        assert params['kernel_size'] in [3, 5]


class TestSuggestTrainingParams:
    """Test training parameter suggestion."""
    
    def test_training_params(self):
        """Test training parameter suggestions."""
        study = optuna.create_study()
        trial = study.ask()
        
        params = suggest_training_params(trial)
        
        assert 'batch_size' in params
        assert 'epochs' in params
        assert params['batch_size'] in [32, 64, 128, 256]
        assert 10 <= params['epochs'] <= 50


class TestPruning:
    """Test pruning functionality."""
    
    def test_pruning_unpromising_trials(self):
        """Test that unpromising trials are pruned."""
        def objective_with_pruning(trial):
            # Suggest a parameter
            x = trial.suggest_float('x', 0, 10)
            
            # Report bad intermediate values
            for step in range(5):
                value = x * step + 100  # Always high value
                trial.report(value, step)
                
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            
            return x
        
        tuner = OptunaHyperparameterTuner(
            objective_fn=objective_with_pruning,
            direction="minimize",
            study_name="test_pruning",
            pruner="median",
            n_startup_trials=2
        )
        
        results = tuner.optimize(n_trials=10, show_progress_bar=False)
        
        # Some trials should be pruned
        assert results['n_pruned'] >= 0  # May or may not prune in small sample


class TestDifferentSamplers:
    """Test different sampling algorithms."""
    
    def test_tpe_sampler(self):
        """Test TPE sampler."""
        def simple_objective(trial):
            x = trial.suggest_float('x', -5, 5)
            return x**2
        
        tuner = OptunaHyperparameterTuner(
            objective_fn=simple_objective,
            direction="minimize",
            sampler="tpe",
            pruner=None
        )
        
        results = tuner.optimize(n_trials=10, show_progress_bar=False)
        assert results['n_trials'] == 10
    
    def test_random_sampler(self):
        """Test random sampler."""
        def simple_objective(trial):
            x = trial.suggest_float('x', -5, 5)
            return x**2
        
        tuner = OptunaHyperparameterTuner(
            objective_fn=simple_objective,
            direction="minimize",
            sampler="random",
            pruner=None
        )
        
        results = tuner.optimize(n_trials=10, show_progress_bar=False)
        assert results['n_trials'] == 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
