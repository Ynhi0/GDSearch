"""
Unit tests for new modular runner components.
"""

import pytest
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import shutil
from src.runners.data_loading import (
    get_mnist_loaders, validate_dataset_split, log_dataset_provenance
)
from src.runners.training import (
    train_epoch, evaluate, check_divergence, compute_gradient_norm
)
from src.runners.reporting import (
    create_results_csv, generate_experiment_summary
)


class SimpleMLP(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)


def create_dummy_loader(n_samples=100, input_dim=10, n_classes=2, batch_size=32):
    """Create a dummy DataLoader for testing."""
    X = torch.randn(n_samples, input_dim)
    y = torch.randint(0, n_classes, (n_samples,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class TestDataLoading:
    """Test data loading module."""
    
    def test_validate_dataset_split(self):
        """Test dataset split validation."""
        train_loader = create_dummy_loader(n_samples=200)
        val_loader = create_dummy_loader(n_samples=50)
        test_loader = create_dummy_loader(n_samples=100)
        
        result = validate_dataset_split(train_loader, val_loader, test_loader)
        
        assert result['train_size'] == 200
        assert result['val_size'] == 50
        assert result['test_size'] == 100
        assert result['total_size'] == 350
        assert result['has_validation'] is True
    
    def test_validate_dataset_split_no_val(self):
        """Test validation without validation set."""
        train_loader = create_dummy_loader(n_samples=200)
        test_loader = create_dummy_loader(n_samples=100)
        
        result = validate_dataset_split(train_loader, None, test_loader)
        
        assert result['val_size'] == 0
        assert result['has_validation'] is False


class TestTraining:
    """Test training module."""
    
    def test_train_epoch(self):
        """Test single epoch training."""
        model = SimpleMLP()
        train_loader = create_dummy_loader()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        device = torch.device('cpu')
        
        metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        
        assert 'train_loss' in metrics
        assert 'train_accuracy' in metrics
        assert 'samples_processed' in metrics
        assert metrics['samples_processed'] == 100
    
    def test_evaluate(self):
        """Test model evaluation."""
        model = SimpleMLP()
        test_loader = create_dummy_loader()
        criterion = nn.CrossEntropyLoss()
        device = torch.device('cpu')
        
        metrics = evaluate(model, test_loader, criterion, device)
        
        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert 'samples_evaluated' in metrics
        assert metrics['samples_evaluated'] == 100
    
    def test_check_divergence_normal(self):
        """Test divergence check with normal training."""
        history = {
            'train_loss': [2.3, 1.8, 1.5, 1.2, 1.0]
        }
        
        assert not check_divergence(history)
    
    def test_check_divergence_exploding(self):
        """Test divergence check with exploding loss."""
        history = {
            'train_loss': [1.0, 2.0, 5.0, 15.0, 50.0]
        }
        
        assert check_divergence(history, threshold=10.0)
    
    def test_compute_gradient_norm(self):
        """Test gradient norm computation."""
        model = SimpleMLP()
        
        # Forward and backward pass
        x = torch.randn(10, 10)
        y = torch.randint(0, 2, (10,))
        criterion = nn.CrossEntropyLoss()
        
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        
        grad_norm = compute_gradient_norm(model)
        
        assert grad_norm > 0
        assert not torch.isnan(torch.tensor(grad_norm))


class TestReporting:
    """Test reporting module."""
    
    def test_create_results_csv(self):
        """Test CSV creation."""
        results = [
            {'optimizer': 'Adam', 'accuracy': 95.0, 'loss': 0.1},
            {'optimizer': 'SGD', 'accuracy': 93.0, 'loss': 0.15}
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_results.csv"
            create_results_csv(results, output_path)
            
            assert output_path.exists()
            
            import pandas as pd
            df = pd.read_csv(output_path)
            assert len(df) == 2
            assert 'optimizer' in df.columns
            assert 'accuracy' in df.columns
    
    def test_generate_experiment_summary(self):
        """Test experiment summary generation."""
        import pandas as pd
        
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir)
            
            # Create mock experiment results
            experiment_results = {
                'mnist': pd.DataFrame({'accuracy': [95.0, 96.0]}),
                'cifar10': pd.DataFrame({'accuracy': [75.0, 76.0]})
            }
            
            report_path = generate_experiment_summary(results_dir, experiment_results)
            
            assert Path(report_path).exists()
            
            with open(report_path, 'r') as f:
                content = f.read()
                assert 'MNIST' in content
                assert 'CIFAR10' in content


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
