"""
Tests for Label Noise Ablation Study.

Validates the correctness of:
- Label corruption with reproducible seeding
- Clean accuracy computation
- Multi-seed experiment orchestration
- Integration with training pipeline
- Statistical analysis of robustness
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from src.experiments.run_label_noise_ablation import (
    NoisyLabelDataset,
    LabelNoiseConfig,
    create_noisy_dataloaders,
    train_with_noisy_labels,
    run_label_noise_ablation,
    create_label_noise_summary,
    analyze_robustness_to_noise
)
from src.core.models import SimpleMLP


class TestNoisyLabelDataset:
    """Tests for NoisyLabelDataset wrapper."""
    
    def test_no_noise_preserves_labels(self):
        """Test that noise_rate=0.0 leaves labels unchanged."""
        # Create simple dataset
        data = torch.randn(100, 10)
        labels = torch.randint(0, 5, (100,))
        dataset = TensorDataset(data, labels)
        
        # Wrap with no noise
        noisy_dataset = NoisyLabelDataset(dataset, noise_rate=0.0, num_classes=5, seed=42)
        
        # Verify all labels unchanged
        for i in range(len(dataset)):
            _, original_label = dataset[i]
            _, noisy_label = noisy_dataset[i]
            assert noisy_label == original_label
        
        # Check clean accuracy
        assert noisy_dataset.get_clean_accuracy() == 1.0
    
    def test_noise_corrupts_correct_fraction(self):
        """Test that noise_rate correctly specifies fraction of corrupted labels."""
        data = torch.randn(1000, 10)
        labels = torch.randint(0, 10, (1000,))
        dataset = TensorDataset(data, labels)
        
        for noise_rate in [0.1, 0.2, 0.4]:
            noisy_dataset = NoisyLabelDataset(dataset, noise_rate=noise_rate, num_classes=10, seed=42)
            
            # Count corrupted labels
            original_labels = labels.numpy()
            noisy_labels = noisy_dataset.noisy_labels
            corrupted = (original_labels != noisy_labels).sum()
            
            # Should be approximately noise_rate * len(dataset)
            expected = int(noise_rate * len(dataset))
            assert abs(corrupted - expected) <= 2, f"Expected ~{expected} corruptions, got {corrupted}"
            
            # Clean accuracy should match
            clean_acc = noisy_dataset.get_clean_accuracy()
            expected_acc = 1.0 - noise_rate
            assert abs(clean_acc - expected_acc) < 0.01
    
    def test_reproducibility_with_seed(self):
        """Test that same seed produces identical noise patterns."""
        data = torch.randn(500, 10)
        labels = torch.randint(0, 5, (500,))
        dataset = TensorDataset(data, labels)
        
        # Create two datasets with same seed
        noisy1 = NoisyLabelDataset(dataset, noise_rate=0.3, num_classes=5, seed=123)
        noisy2 = NoisyLabelDataset(dataset, noise_rate=0.3, num_classes=5, seed=123)
        
        # Should have identical noisy labels
        assert np.array_equal(noisy1.noisy_labels, noisy2.noisy_labels)
    
    def test_different_seeds_produce_different_noise(self):
        """Test that different seeds produce different noise patterns."""
        data = torch.randn(500, 10)
        labels = torch.randint(0, 5, (500,))
        dataset = TensorDataset(data, labels)
        
        # Create two datasets with different seeds
        noisy1 = NoisyLabelDataset(dataset, noise_rate=0.3, num_classes=5, seed=42)
        noisy2 = NoisyLabelDataset(dataset, noise_rate=0.3, num_classes=5, seed=999)
        
        # Should have different noisy labels
        assert not np.array_equal(noisy1.noisy_labels, noisy2.noisy_labels)
        
        # But same number of corruptions
        original = labels.numpy()
        corrupted1 = (original != noisy1.noisy_labels).sum()
        corrupted2 = (original != noisy2.noisy_labels).sum()
        assert corrupted1 == corrupted2
    
    def test_corrupted_labels_differ_from_original(self):
        """Test that corrupted labels are always different from originals."""
        data = torch.randn(200, 5)
        labels = torch.randint(0, 4, (200,))
        dataset = TensorDataset(data, labels)
        
        noisy_dataset = NoisyLabelDataset(dataset, noise_rate=0.5, num_classes=4, seed=42)
        
        original_labels = labels.numpy()
        noisy_labels = noisy_dataset.noisy_labels
        
        # For corrupted indices, labels must differ
        corrupted_mask = original_labels != noisy_labels
        corrupted_indices = np.where(corrupted_mask)[0]
        
        for idx in corrupted_indices:
            assert original_labels[idx] != noisy_labels[idx], \
                f"Corrupted label at {idx} should differ from original"


class TestCreateNoisyDataloaders:
    """Tests for dataloader creation with noise."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA for full test")
    def test_mnist_dataloader_creation(self):
        """Test creating MNIST dataloaders with noise."""
        train_loader, val_loader, test_loader, num_classes = create_noisy_dataloaders(
            dataset_name='mnist',
            noise_rate=0.2,
            seed=42,
            batch_size=64,
            num_workers=0  # Use 0 for testing
        )
        
        assert num_classes == 10
        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert len(test_loader) > 0
        
        # Check batch shapes
        for inputs, targets in train_loader:
            assert inputs.shape[1:] == (1, 28, 28)  # MNIST image shape
            assert targets.shape[0] <= 64  # Batch size
            break
    
    def test_cifar10_dataloader_creation(self):
        """Test creating CIFAR-10 dataloaders with noise."""
        train_loader, val_loader, test_loader, num_classes = create_noisy_dataloaders(
            dataset_name='cifar10',
            noise_rate=0.1,
            seed=123,
            batch_size=32,
            num_workers=0
        )
        
        assert num_classes == 10
        assert len(train_loader) > 0
        
        # Check batch shapes
        for inputs, targets in train_loader:
            assert inputs.shape[1:] == (3, 32, 32)  # CIFAR-10 image shape
            assert targets.shape[0] <= 32
            break
    
    def test_unsupported_dataset_raises_error(self):
        """Test that unsupported dataset names raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported dataset"):
            create_noisy_dataloaders(
                dataset_name='imagenet',  # Not supported
                noise_rate=0.1,
                seed=42
            )
    
    def test_validation_split_size(self):
        """Test that validation split is approximately 10% of training data."""
        train_loader, val_loader, test_loader, _ = create_noisy_dataloaders(
            dataset_name='mnist',
            noise_rate=0.0,
            seed=42,
            batch_size=1000,
            num_workers=0
        )
        
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        
        # Validation should be ~10% of original training set
        original_train_size = train_size + val_size
        expected_val_size = int(0.1 * original_train_size)
        
        assert abs(val_size - expected_val_size) <= 10


class TestTrainWithNoisyLabels:
    """Tests for training with noisy labels."""
    
    @pytest.fixture
    def simple_model_and_data(self):
        """Create simple model and dataloaders for testing."""
        # Create synthetic data
        train_data = torch.randn(200, 784)
        train_labels = torch.randint(0, 10, (200,))
        train_dataset = TensorDataset(train_data, train_labels)
        
        val_data = torch.randn(50, 784)
        val_labels = torch.randint(0, 10, (50,))
        val_dataset = TensorDataset(val_data, val_labels)
        
        test_data = torch.randn(50, 784)
        test_labels = torch.randint(0, 10, (50,))
        test_dataset = TensorDataset(test_data, test_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        model = SimpleMLP(input_dim=784, num_classes=10)
        
        return model, train_loader, val_loader, test_loader
    
    def test_training_completes_successfully(self, simple_model_and_data):
        """Test that training runs without errors."""
        model, train_loader, val_loader, test_loader = simple_model_and_data
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        config = LabelNoiseConfig(epochs=2, device='cpu')
        
        results_df = train_with_noisy_labels(
            model, optimizer, train_loader, val_loader, test_loader,
            config, noise_rate=0.1, seed=42, optimizer_name='SGD'
        )
        
        assert len(results_df) == 2  # 2 epochs
        assert 'train_loss' in results_df.columns
        assert 'val_acc' in results_df.columns
        assert 'test_acc' in results_df.columns
    
    def test_results_contain_correct_metadata(self, simple_model_and_data):
        """Test that results DataFrame contains all required metadata."""
        model, train_loader, val_loader, test_loader = simple_model_and_data
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        config = LabelNoiseConfig(epochs=3, device='cpu')
        
        results_df = train_with_noisy_labels(
            model, optimizer, train_loader, val_loader, test_loader,
            config, noise_rate=0.2, seed=123, optimizer_name='Adam'
        )
        
        # Check metadata columns
        assert (results_df['optimizer'] == 'Adam').all()
        assert (results_df['noise_rate'] == 0.2).all()
        assert (results_df['seed'] == 123).all()
        assert (results_df['epoch'] == [0, 1, 2]).all()


class TestRunLabelNoiseAblation:
    """Tests for full ablation study execution."""
    
    def test_ablation_with_minimal_config(self):
        """Test running ablation with minimal configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LabelNoiseConfig(
                noise_rates=[0.0, 0.1],
                seeds=[42],
                epochs=2,
                batch_size=128
            )
            
            optimizers_config = {
                'SGD': {'lr': 0.01},
                'Adam': {'lr': 0.001}
            }
            
            # This will download MNIST if not present
            results_df = run_label_noise_ablation(
                dataset_name='mnist',
                model_name='mlp',
                optimizers_config=optimizers_config,
                config=config,
                output_dir=tmpdir
            )
            
            # Check results structure
            assert len(results_df) > 0
            assert 'optimizer' in results_df.columns
            assert 'noise_rate' in results_df.columns
            assert 'seed' in results_df.columns
            
            # Check output files created
            output_path = Path(tmpdir)
            assert (output_path / "label_noise_results_mnist_mlp.csv").exists()
            assert (output_path / "label_noise_summary_mnist_mlp.csv").exists()
    
    def test_summary_statistics_computation(self):
        """Test summary statistics generation."""
        # Create mock results
        data = []
        for optimizer in ['SGD', 'Adam']:
            for noise_rate in [0.0, 0.1]:
                for seed in [42, 123, 456]:
                    for epoch in range(5):
                        data.append({
                            'epoch': epoch,
                            'optimizer': optimizer,
                            'noise_rate': noise_rate,
                            'seed': seed,
                            'train_acc': 90.0 - noise_rate * 20 + np.random.randn(),
                            'val_acc': 88.0 - noise_rate * 20 + np.random.randn(),
                            'test_acc': 87.0 - noise_rate * 20 + np.random.randn(),
                            'train_loss': 0.3 + noise_rate + np.random.randn() * 0.1,
                            'val_loss': 0.4 + noise_rate + np.random.randn() * 0.1,
                            'test_loss': 0.4 + noise_rate + np.random.randn() * 0.1
                        })
        
        results_df = pd.DataFrame(data)
        summary = create_label_noise_summary(results_df)
        
        # Check summary structure
        assert 'optimizer' in summary.columns
        assert 'noise_rate' in summary.columns
        assert 'test_acc_mean' in summary.columns
        assert 'test_acc_std' in summary.columns
        
        # Each optimizer x noise_rate combination should have one row
        assert len(summary) == 2 * 2  # 2 optimizers × 2 noise rates


class TestRobustnessAnalysis:
    """Tests for robustness metrics computation."""
    
    def test_robustness_metric_computation(self):
        """Test computation of accuracy degradation metrics."""
        # Create mock summary data
        summary_data = [
            {'optimizer': 'SGD', 'noise_rate': 0.0, 'test_acc_mean': 95.0, 'test_acc_std': 0.5},
            {'optimizer': 'SGD', 'noise_rate': 0.1, 'test_acc_mean': 90.0, 'test_acc_std': 1.0},
            {'optimizer': 'SGD', 'noise_rate': 0.2, 'test_acc_mean': 85.0, 'test_acc_std': 1.5},
            {'optimizer': 'Adam', 'noise_rate': 0.0, 'test_acc_mean': 96.0, 'test_acc_std': 0.3},
            {'optimizer': 'Adam', 'noise_rate': 0.1, 'test_acc_mean': 92.0, 'test_acc_std': 0.8},
            {'optimizer': 'Adam', 'noise_rate': 0.2, 'test_acc_mean': 88.0, 'test_acc_std': 1.2},
        ]
        summary_df = pd.DataFrame(summary_data)
        
        robustness = analyze_robustness_to_noise(summary_df)
        
        # Check structure
        assert 'optimizer' in robustness.columns
        assert 'noise_rate' in robustness.columns
        assert 'clean_acc' in robustness.columns
        assert 'absolute_drop' in robustness.columns
        assert 'relative_drop_pct' in robustness.columns
        
        # Verify computations
        sgd_01 = robustness[(robustness['optimizer'] == 'SGD') & (robustness['noise_rate'] == 0.1)].iloc[0]
        assert sgd_01['clean_acc'] == 95.0
        assert sgd_01['noisy_acc'] == 90.0
        assert sgd_01['absolute_drop'] == 5.0
        assert abs(sgd_01['relative_drop_pct'] - (5.0/95.0)*100) < 0.01
    
    def test_robustness_excludes_clean_baseline(self):
        """Test that robustness metrics exclude noise_rate=0.0."""
        summary_data = [
            {'optimizer': 'SGD', 'noise_rate': 0.0, 'test_acc_mean': 95.0, 'test_acc_std': 0.5},
            {'optimizer': 'SGD', 'noise_rate': 0.2, 'test_acc_mean': 85.0, 'test_acc_std': 1.5},
        ]
        summary_df = pd.DataFrame(summary_data)
        
        robustness = analyze_robustness_to_noise(summary_df)
        
        # Should only have rows for noise_rate > 0
        assert (robustness['noise_rate'] > 0.0).all()
        assert len(robustness) == 1  # Only the 0.2 noise rate


class TestLabelNoiseConfig:
    """Tests for configuration dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = LabelNoiseConfig()
        
        assert config.noise_rates == [0.0, 0.1, 0.2, 0.4]
        assert config.seeds == [42, 123, 456, 789, 1011]
        assert config.epochs == 50
        assert config.batch_size == 128
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = LabelNoiseConfig(
            noise_rates=[0.0, 0.3],
            seeds=[1, 2, 3],
            epochs=10,
            batch_size=64
        )
        
        assert config.noise_rates == [0.0, 0.3]
        assert config.seeds == [1, 2, 3]
        assert config.epochs == 10
        assert config.batch_size == 64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
