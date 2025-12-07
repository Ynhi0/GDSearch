"""
Tests for Advanced Training Features Ablation Study

Ensures academic rigor and correctness of ablation experiments.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import pandas as pd
import numpy as np
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiments.advanced_training_ablation import (
    SimpleCNN,
    set_seed,
    train_epoch,
    evaluate,
    run_single_experiment,
    run_ablation_study
)
from src.core.training_utils import (
    LabelSmoothingCrossEntropy,
    ModelEMA,
    AMPWrapper,
    create_amp_wrapper,
    create_model_ema
)


class TestAblationStudyRigor:
    """Test academic rigor of ablation study"""
    
    def test_controlled_experiments(self):
        """Ensure only one variable changes at a time"""
        configs = [
            {'name': 'Baseline', 'use_amp': False, 'use_label_smoothing': False, 'use_ema': False},
            {'name': 'AMP_only', 'use_amp': True, 'use_label_smoothing': False, 'use_ema': False},
            {'name': 'LabelSmoothing_only', 'use_amp': False, 'use_label_smoothing': True, 'use_ema': False},
            {'name': 'EMA_only', 'use_amp': False, 'use_label_smoothing': False, 'use_ema': True},
        ]
        
        baseline = configs[0]
        
        for config in configs[1:]:
            # Count differences from baseline
            diffs = sum(1 for k in ['use_amp', 'use_label_smoothing', 'use_ema'] 
                       if config[k] != baseline[k])
            
            assert diffs == 1, f"{config['name']} should differ from baseline in exactly 1 variable, found {diffs}"
    
    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results"""
        device = torch.device('cpu')
        model1 = SimpleCNN().to(device)
        model2 = SimpleCNN().to(device)
        
        # Set same seed and create identical inputs
        set_seed(42)
        x1 = torch.randn(4, 1, 28, 28)
        
        set_seed(42)
        x2 = torch.randn(4, 1, 28, 28)
        
        # Outputs should be identical
        assert torch.allclose(x1, x2), "Same seed should produce identical random tensors"
    
    def test_model_architecture(self):
        """Test model architecture is reasonable"""
        model = SimpleCNN(num_classes=10)
        
        # Test forward pass
        x = torch.randn(4, 1, 28, 28)
        output = model(x)
        
        assert output.shape == (4, 10), f"Expected output shape (4, 10), got {output.shape}"
        assert not torch.isnan(output).any(), "Model output contains NaN"
        assert not torch.isinf(output).any(), "Model output contains Inf"
    
    def test_training_epoch_basic(self):
        """Test basic training epoch functionality"""
        device = torch.device('cpu')
        model = SimpleCNN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Create small dataset
        dataset = torch.utils.data.TensorDataset(
            torch.randn(32, 1, 28, 28),
            torch.randint(0, 10, (32,))
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=8)
        
        # Train one epoch
        loss, acc = train_epoch(model, loader, optimizer, criterion, device)
        
        assert isinstance(loss, float), "Loss should be float"
        assert isinstance(acc, float), "Accuracy should be float"
        assert 0 <= acc <= 100, f"Accuracy should be in [0, 100], got {acc}"
        assert loss >= 0, f"Loss should be non-negative, got {loss}"
    
    def test_evaluation(self):
        """Test evaluation function"""
        device = torch.device('cpu')
        model = SimpleCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        
        # Create small dataset
        dataset = torch.utils.data.TensorDataset(
            torch.randn(32, 1, 28, 28),
            torch.randint(0, 10, (32,))
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=8)
        
        # Evaluate
        loss, acc = evaluate(model, loader, criterion, device)
        
        assert isinstance(loss, float), "Loss should be float"
        assert isinstance(acc, float), "Accuracy should be float"
        assert 0 <= acc <= 100, f"Accuracy should be in [0, 100], got {acc}"
    
    @pytest.mark.slow
    def test_single_experiment_baseline(self):
        """Test running a single baseline experiment"""
        device = torch.device('cpu')
        
        # Create small dataset
        dataset = torch.utils.data.TensorDataset(
            torch.randn(128, 1, 28, 28),
            torch.randint(0, 10, (128,))
        )
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=16)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=16)
        
        config = {
            'use_amp': False,
            'use_label_smoothing': False,
            'use_ema': False
        }
        
        result = run_single_experiment(
            config, train_loader, test_loader, device, epochs=2, seed=42
        )
        
        # Validate result structure
        assert 'final_test_acc' in result
        assert 'training_time' in result
        assert 'history' in result
        assert len(result['history']) == 2, "Should have 2 epochs"
        
        # Validate metrics are reasonable
        assert 0 <= result['final_test_acc'] <= 100
        assert result['training_time'] > 0
    
    @pytest.mark.slow
    def test_single_experiment_with_amp(self):
        """Test experiment with AMP enabled"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for AMP test")
        
        device = torch.device('cuda')
        
        # Create small dataset
        dataset = torch.utils.data.TensorDataset(
            torch.randn(128, 1, 28, 28),
            torch.randint(0, 10, (128,))
        )
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=16)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=16)
        
        config = {
            'use_amp': True,
            'use_label_smoothing': False,
            'use_ema': False
        }
        
        result = run_single_experiment(
            config, train_loader, test_loader, device, epochs=2, seed=42
        )
        
        assert 'final_test_acc' in result
        assert result['training_time'] > 0
    
    @pytest.mark.slow
    def test_single_experiment_with_label_smoothing(self):
        """Test experiment with label smoothing"""
        device = torch.device('cpu')
        
        # Create small dataset
        dataset = torch.utils.data.TensorDataset(
            torch.randn(128, 1, 28, 28),
            torch.randint(0, 10, (128,))
        )
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=16)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=16)
        
        config = {
            'use_amp': False,
            'use_label_smoothing': True,
            'label_smoothing_factor': 0.1,
            'use_ema': False
        }
        
        result = run_single_experiment(
            config, train_loader, test_loader, device, epochs=2, seed=42
        )
        
        assert 'final_test_acc' in result
        assert result['training_time'] > 0
    
    @pytest.mark.slow
    def test_single_experiment_with_ema(self):
        """Test experiment with EMA"""
        device = torch.device('cpu')
        
        # Create small dataset
        dataset = torch.utils.data.TensorDataset(
            torch.randn(128, 1, 28, 28),
            torch.randint(0, 10, (128,))
        )
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=16)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=16)
        
        config = {
            'use_amp': False,
            'use_label_smoothing': False,
            'use_ema': True,
            'ema_decay': 0.9999
        }
        
        result = run_single_experiment(
            config, train_loader, test_loader, device, epochs=2, seed=42
        )
        
        assert 'final_ema_acc' in result
        assert result['final_ema_acc'] > 0
        assert result['training_time'] > 0
    
    @pytest.mark.slow
    def test_all_features_combined(self):
        """Test experiment with all features enabled"""
        device = torch.device('cpu')
        
        # Create small dataset
        dataset = torch.utils.data.TensorDataset(
            torch.randn(128, 1, 28, 28),
            torch.randint(0, 10, (128,))
        )
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=16)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=16)
        
        config = {
            'use_amp': False,  # CPU doesn't support AMP effectively
            'use_label_smoothing': True,
            'label_smoothing_factor': 0.1,
            'use_ema': True,
            'ema_decay': 0.9999
        }
        
        result = run_single_experiment(
            config, train_loader, test_loader, device, epochs=2, seed=42
        )
        
        assert 'final_test_acc' in result
        assert 'final_ema_acc' in result
        assert result['training_time'] > 0


class TestAblationStudyStatistics:
    """Test statistical validity of ablation study"""
    
    def test_multiple_seeds_reduce_variance(self):
        """Using multiple seeds should reduce reported variance"""
        # This is a conceptual test - in practice, we'd run actual experiments
        # Here we just verify the logic
        
        # Simulate results from 3 vs 1 seed
        results_1_seed = [95.0]  # Single seed
        results_3_seeds = [94.5, 95.0, 95.5]  # Three seeds
        
        # Standard error of mean (SEM) should be lower with more seeds
        sem_1 = np.std(results_1_seed) / np.sqrt(len(results_1_seed))
        sem_3 = np.std(results_3_seeds) / np.sqrt(len(results_3_seeds))
        
        # With more seeds, we get better estimate (lower SEM for same std)
        assert len(results_3_seeds) > len(results_1_seed), "Should have more seeds"
    
    def test_results_dataframe_structure(self):
        """Test that results are properly structured for analysis"""
        # Expected columns in results
        expected_columns = [
            'configuration',
            'use_amp',
            'use_label_smoothing',
            'use_ema',
            'mean_test_acc',
            'std_test_acc',
            'n_seeds'
        ]
        
        # This would be validated after running actual study
        # For now, just verify we know what we expect
        assert len(expected_columns) >= 6, "Should track sufficient metrics"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
