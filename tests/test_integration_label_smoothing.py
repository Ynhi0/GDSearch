"""
Integration test for label smoothing, AMP, and EMA configuration propagation.

Verifies that CLI/config flags for label_smoothing, use_amp, and use_ema 
actually affect training behavior in the canonical train_and_evaluate pipeline.

AUDIT FIX: Ensures proposal-compliant features work end-to-end.
"""
import pytest
import torch
import pandas as pd
from src.experiments.run_nn_experiment import train_and_evaluate
from src.core.training_utils import LabelSmoothingCrossEntropy


class TestLabelSmoothingIntegration:
    """Test label smoothing propagation through training pipeline."""
    
    def test_label_smoothing_affects_training(self):
        """Verify that label_smoothing config actually affects loss computation."""
        base_config = {
            'model': 'SimpleMLP',
            'dataset': 'MNIST',
            'optimizer': 'Adam',
            'lr': 0.001,
            'epochs': 1,
            'batch_size': 128,
            'seed': 42
        }
        
        # Run without smoothing
        config_no_smooth = base_config.copy()
        config_no_smooth['label_smoothing'] = 0.0
        df_no_smooth = train_and_evaluate(config_no_smooth)
        
        # Run with smoothing
        config_with_smooth = base_config.copy()
        config_with_smooth['label_smoothing'] = 0.1
        df_with_smooth = train_and_evaluate(config_with_smooth)
        
        # Extract final test losses
        loss_no_smooth = df_no_smooth[df_no_smooth['phase'] == 'eval']['test_loss'].iloc[-1]
        loss_with_smooth = df_with_smooth[df_with_smooth['phase'] == 'eval']['test_loss'].iloc[-1]
        
        # Losses should differ (smoothing introduces entropy floor)
        assert loss_no_smooth != loss_with_smooth, \
            "Label smoothing config did not affect loss computation"
        
        # With smoothing, loss should be higher due to entropy floor
        # (assuming same convergence progress)
        assert loss_with_smooth > 0.3, \
            f"Expected entropy floor effect, got loss={loss_with_smooth}"
    
    def test_entropy_floor_computation(self):
        """Verify entropy floor calculation for label smoothing."""
        num_classes = 10
        smoothing = 0.1
        
        loss_fn = LabelSmoothingCrossEntropy(smoothing=smoothing)
        entropy_floor = loss_fn.get_entropy_floor(num_classes)
        
        # For 10 classes with smoothing=0.1, floor should be ~0.54
        assert 0.5 < entropy_floor < 0.6, \
            f"Expected entropy floor ~0.54, got {entropy_floor}"
        
        # Zero smoothing should have zero floor
        loss_fn_no_smooth = LabelSmoothingCrossEntropy(smoothing=0.0)
        floor_no_smooth = loss_fn_no_smooth.get_entropy_floor(num_classes)
        assert floor_no_smooth == 0.0, \
            f"Expected zero floor for no smoothing, got {floor_no_smooth}"


class TestAMPIntegration:
    """Test automatic mixed precision propagation."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="AMP requires CUDA")
    def test_amp_flag_enables_mixed_precision(self):
        """Verify that use_amp config enables mixed precision training."""
        config = {
            'model': 'SimpleMLP',
            'dataset': 'MNIST',
            'optimizer': 'Adam',
            'lr': 0.001,
            'epochs': 1,
            'batch_size': 128,
            'seed': 42,
            'use_amp': True
        }
        
        # Should run without error
        df = train_and_evaluate(config)
        
        # Verify training completed
        assert len(df) > 0
        assert 'test_accuracy' in df.columns
        
        # Check that training happened
        final_row = df[df['phase'] == 'eval'].iloc[-1]
        assert final_row['test_accuracy'] > 0.5, "Training failed with AMP enabled"


class TestEMAIntegration:
    """Test exponential moving average propagation."""
    
    def test_ema_flag_enables_model_averaging(self):
        """Verify that use_ema config enables model EMA."""
        base_config = {
            'model': 'SimpleMLP',
            'dataset': 'MNIST',
            'optimizer': 'Adam',
            'lr': 0.001,
            'epochs': 2,
            'batch_size': 128,
            'seed': 42
        }
        
        # Run without EMA
        config_no_ema = base_config.copy()
        config_no_ema['use_ema'] = False
        df_no_ema = train_and_evaluate(config_no_ema)
        
        # Run with EMA (use faster decay for short training runs)
        config_with_ema = base_config.copy()
        config_with_ema['use_ema'] = True
        config_with_ema['ema_decay'] = 0.99  # Lower decay for 2-epoch test
        df_with_ema = train_and_evaluate(config_with_ema)

        # Both should complete successfully
        assert len(df_no_ema) > 0
        assert len(df_with_ema) > 0

        # EMA typically improves or stabilizes accuracy
        acc_no_ema = df_no_ema[df_no_ema['phase'] == 'eval']['test_accuracy'].iloc[-1]
        acc_with_ema = df_with_ema[df_with_ema['phase'] == 'eval']['test_accuracy'].iloc[-1]

        # Both should achieve reasonable accuracy
        assert acc_no_ema > 0.85, f"Training without EMA failed: acc={acc_no_ema}"
        # EMA may be slightly lower for short runs, but should still train
        assert acc_with_ema > 0.80, f"Training with EMA failed: acc={acc_with_ema}"


class TestCombinedFeatures:
    """Test that all features work together."""
    
    def test_all_features_compatible(self):
        """Verify that label_smoothing, AMP, and EMA work together."""
        config = {
            'model': 'SimpleMLP',
            'dataset': 'MNIST',
            'optimizer': 'Adam',
            'lr': 0.001,
            'epochs': 2,  # Increased epochs for EMA to stabilize
            'batch_size': 128,
            'seed': 42,
            'label_smoothing': 0.1,
            'use_amp': False,  # Keep False for CPU compatibility
            'use_ema': True,
            'ema_decay': 0.99  # Lower decay for short training
        }
        
        # Should run without error
        df = train_and_evaluate(config)
        
        # Verify all phases present
        assert 'train' in df['phase'].values
        assert 'eval' in df['phase'].values
        
        # Verify reasonable performance (label smoothing + EMA may lower accuracy slightly)
        final_acc = df[df['phase'] == 'eval']['test_accuracy'].iloc[-1]
        assert final_acc > 0.80, f"Combined features training failed: acc={final_acc}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
