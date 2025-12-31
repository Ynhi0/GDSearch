#!/usr/bin/env python3
"""Tests for Optimizer-Initialization Interaction Ablation Study"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiments.initialization_ablation import (
    SimpleCNN, set_seed, run_single_experiment, run_initialization_ablation
)
from torch.utils.data import DataLoader, TensorDataset


@pytest.fixture
def dummy_data():
    """Create dummy MNIST-like data for testing"""
    # Small dataset for fast testing
    X_train = torch.randn(100, 1, 28, 28)
    y_train = torch.randint(0, 10, (100,))
    X_test = torch.randn(50, 1, 28, 28)
    y_test = torch.randint(0, 10, (50,))
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)
    
    return train_loader, test_loader


class TestInitializationMethods:
    """Test initialization methods are applied correctly"""
    
    def test_zero_initialization(self):
        """Zero initialization should set all weights to zero"""
        model = SimpleCNN(num_classes=10)
        model.apply_initialization('zero')
        
        # Check that all weights are zero (except BatchNorm which isn't initialized)
        for name, param in model.named_parameters():
            if 'weight' in name and 'bn' not in name:
                assert torch.allclose(param, torch.zeros_like(param)), \
                    f"{name} should be all zeros"
    
    def test_xavier_initialization_shape(self):
        """Xavier initialization should preserve variance"""
        model = SimpleCNN(num_classes=10)
        model.apply_initialization('xavier_normal')
        
        # Xavier should initialize with variance ~2/(fan_in + fan_out)
        # Just check it's not zero and not too large
        for layer in model.init_layers:
            if hasattr(layer, 'weight'):
                weight = layer.weight
                assert not torch.allclose(weight, torch.zeros_like(weight))
                assert weight.abs().max() < 10  # Not too large
    
    def test_kaiming_initialization_shape(self):
        """Kaiming initialization should preserve variance for ReLU"""
        model = SimpleCNN(num_classes=10, activation='relu')
        model.apply_initialization('kaiming_normal')
        
        # Kaiming should initialize with variance ~2/fan_in
        for layer in model.init_layers:
            if hasattr(layer, 'weight'):
                weight = layer.weight
                assert not torch.allclose(weight, torch.zeros_like(weight))
                assert weight.abs().max() < 10
    
    def test_all_initialization_methods(self):
        """All initialization methods should run without error"""
        methods = [
            'zero', 'uniform_small', 'normal_small',
            'xavier_uniform', 'xavier_normal',
            'kaiming_uniform', 'kaiming_normal'
        ]
        
        for method in methods:
            model = SimpleCNN(num_classes=10)
            model.apply_initialization(method)
            
            # Just verify it runs without error
            assert model is not None


class TestReproducibility:
    """Test that experiments are reproducible"""
    
    def test_same_seed_same_init(self):
        """Same seed should give same initialization"""
        set_seed(42)
        model1 = SimpleCNN(num_classes=10)
        model1.apply_initialization('xavier_normal')
        weights1 = model1.conv1.weight.clone()
        
        set_seed(42)
        model2 = SimpleCNN(num_classes=10)
        model2.apply_initialization('xavier_normal')
        weights2 = model2.conv1.weight.clone()
        
        assert torch.allclose(weights1, weights2), \
            "Same seed should give same initialization"
    
    def test_different_seed_different_init(self):
        """Different seeds should give different initializations"""
        set_seed(42)
        model1 = SimpleCNN(num_classes=10)
        model1.apply_initialization('xavier_normal')
        weights1 = model1.conv1.weight.clone()
        
        set_seed(123)
        model2 = SimpleCNN(num_classes=10)
        model2.apply_initialization('xavier_normal')
        weights2 = model2.conv1.weight.clone()
        
        assert not torch.allclose(weights1, weights2), \
            "Different seeds should give different initializations"


class TestSingleExperiment:
    """Test single experiment execution"""
    
    @pytest.mark.slow
    def test_single_experiment_runs(self, dummy_data):
        """Single experiment should complete without error"""
        train_loader, test_loader = dummy_data
        device = torch.device("cpu")
        
        result = run_single_experiment(
            init_method='xavier_normal',
            optimizer_name='Adam',
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=2,
            seed=42
        )
        
        # Check result structure
        assert 'init_method' in result
        assert 'optimizer' in result
        assert 'final_test_acc' in result
        assert 'best_test_acc' in result
        assert 'convergence_epoch' in result
        assert 'training_time' in result
        assert 'diverged' in result
        assert 'history' in result
        
        # Check values are reasonable
        assert 0 <= result['final_test_acc'] <= 100
        assert result['convergence_epoch'] >= 1
        assert result['training_time'] > 0
    
    @pytest.mark.slow
    def test_different_optimizers(self, dummy_data):
        """Test with different optimizers"""
        train_loader, test_loader = dummy_data
        device = torch.device("cpu")
        
        optimizers = ['SGD', 'SGD_Momentum', 'Adam', 'AdamW']
        
        for opt_name in optimizers:
            result = run_single_experiment(
                init_method='kaiming_normal',
                optimizer_name=opt_name,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                epochs=1,
                seed=42
            )
            
            assert result['optimizer'] == opt_name
            assert not result['diverged']  # Should not diverge on dummy data


class TestAblationStudy:
    """Test full ablation study"""
    
    @pytest.mark.slow
    def test_quick_ablation_runs(self, tmp_path):
        """Quick ablation study should complete"""
        result_df = run_initialization_ablation(
            results_dir=str(tmp_path),
            seeds=[1, 2],
            epochs=2,
            quick=True
        )
        
        # Check DataFrame structure
        assert len(result_df) > 0
        assert 'initialization' in result_df.columns
        assert 'optimizer' in result_df.columns
        assert 'mean_test_acc' in result_df.columns
        assert 'std_test_acc' in result_df.columns
        
        # Check results file was created
        assert (tmp_path / "initialization_ablation_summary.csv").exists()
    
    @pytest.mark.slow
    def test_ablation_multi_seed_variance(self, tmp_path):
        """Multi-seed experiments should have variance statistics"""
        result_df = run_initialization_ablation(
            results_dir=str(tmp_path),
            seeds=[1, 2, 3],
            epochs=2,
            quick=True
        )
        
        # All configurations should have std_test_acc
        assert all(result_df['std_test_acc'] >= 0)
        assert all(result_df['n_seeds'] == 3)


class TestReproducibilityAblation:
    """Test reproducibility of the ablation study"""
    
    @pytest.mark.slow
    def test_controlled_comparison(self, tmp_path):
        """Study should systematically vary one variable at a time"""
        result_df = run_initialization_ablation(
            results_dir=str(tmp_path),
            seeds=[1, 2],
            epochs=2,
            quick=True
        )
        
        # For each optimizer, there should be multiple initializations
        for opt in result_df['optimizer'].unique():
            opt_configs = result_df[result_df['optimizer'] == opt]
            assert len(opt_configs) > 1, \
                f"Optimizer {opt} should be tested with multiple initializations"
        
        # For each initialization, there should be multiple optimizers
        for init in result_df['initialization'].unique():
            init_configs = result_df[result_df['initialization'] == init]
            assert len(init_configs) > 1, \
                f"Initialization {init} should be tested with multiple optimizers"
    
    @pytest.mark.slow
    def test_robustness_analysis(self, tmp_path):
        """Study should enable robustness analysis"""
        result_df = run_initialization_ablation(
            results_dir=str(tmp_path),
            seeds=[1, 2, 3],
            epochs=2,
            quick=True
        )
        
        # Group by optimizer and compute variance across initializations
        for opt in result_df['optimizer'].unique():
            opt_df = result_df[result_df['optimizer'] == opt]
            
            # Variance in performance across initializations
            init_variance = opt_df['mean_test_acc'].std()
            
            # This measures robustness: low variance = robust to initialization
            assert init_variance >= 0  # Just check it's computable


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
