"""
Test suite for error handling utilities.

Run:
    pytest tests/test_error_handling_patterns.py -v
"""

import pytest
import torch
import torch.nn as nn
import logging
from pathlib import Path
import tempfile

from src.utils.error_handling_patterns import (
    gpu_safe_operation,
    model_cleanup_guard,
    validate_preconditions,
    atomic_save_checkpoint,
    ErrorContext,
    safe_gpu_operation
)


class TestGPUSafeOperation:
    """Test GPU-safe operation context manager."""
    
    def test_normal_operation(self):
        """Test that normal operations work correctly."""
        result = []
        with gpu_safe_operation("Test operation"):
            result.append(1)
        assert result == [1]
    
    def test_oom_error_handling(self):
        """Test that OOM errors are caught and re-raised with context."""
        with pytest.raises(RuntimeError) as exc_info:
            with gpu_safe_operation("Test OOM"):
                raise RuntimeError("CUDA out of memory")
        
        assert "out of memory" in str(exc_info.value).lower()
        assert "Test OOM" in str(exc_info.value)
    
    def test_cuda_error_handling(self):
        """Test that CUDA errors are caught and re-raised with context."""
        with pytest.raises(RuntimeError) as exc_info:
            with gpu_safe_operation("Test CUDA"):
                raise RuntimeError("CUDA error: device-side assert triggered")
        
        assert "cuda" in str(exc_info.value).lower()
        assert "Test CUDA" in str(exc_info.value)
    
    def test_other_runtime_error(self):
        """Test that other RuntimeErrors are re-raised as-is."""
        with pytest.raises(RuntimeError):
            with gpu_safe_operation("Test other"):
                raise RuntimeError("Some other error")


class TestModelCleanupGuard:
    """Test model cleanup guard context manager."""
    
    def test_cleanup_without_model(self):
        """Test that cleanup works when model is None."""
        with model_cleanup_guard(None):
            pass  # Should not raise
    
    def test_cleanup_with_model(self):
        """Test that model is deleted after context."""
        model = nn.Linear(10, 5)
        with model_cleanup_guard(model):
            pass
        # Model should be deleted (can't test directly, but no error should occur)
    
    def test_cleanup_on_error(self):
        """Test that cleanup happens even when error occurs."""
        model = nn.Linear(10, 5)
        with pytest.raises(ValueError):
            with model_cleanup_guard(model):
                raise ValueError("Test error")
        # Cleanup should still happen


class TestValidatePreconditions:
    """Test precondition validation."""
    
    def test_valid_model(self):
        """Test that valid model passes validation."""
        model = nn.Linear(10, 5)
        validate_preconditions(model=model)  # Should not raise
    
    def test_invalid_model_type(self):
        """Test that invalid model type is caught."""
        with pytest.raises(TypeError) as exc_info:
            validate_preconditions(model="not a model")
        assert "torch.nn.Module" in str(exc_info.value)
    
    def test_model_no_parameters(self):
        """Test that model with no parameters is caught."""
        class EmptyModel(nn.Module):
            def forward(self, x):
                return x
        
        with pytest.raises(ValueError) as exc_info:
            validate_preconditions(model=EmptyModel())
        assert "no parameters" in str(exc_info.value)
    
    def test_empty_data_loader(self):
        """Test that empty data loader is caught."""
        class EmptyLoader:
            def __len__(self):
                return 0
        
        with pytest.raises(ValueError) as exc_info:
            validate_preconditions(data_loader=EmptyLoader())
        assert "empty" in str(exc_info.value).lower()
    
    def test_invalid_epochs(self):
        """Test that invalid epochs are caught."""
        with pytest.raises(TypeError):
            validate_preconditions(epochs="10")
        
        with pytest.raises(ValueError):
            validate_preconditions(epochs=0)
        
        with pytest.raises(ValueError):
            validate_preconditions(epochs=-5)
    
    def test_invalid_learning_rate(self):
        """Test that invalid learning rates are caught."""
        with pytest.raises(TypeError):
            validate_preconditions(learning_rate="0.001")
        
        with pytest.raises(ValueError):
            validate_preconditions(learning_rate=0)
        
        with pytest.raises(ValueError):
            validate_preconditions(learning_rate=-0.001)
    
    def test_invalid_batch_size(self):
        """Test that invalid batch sizes are caught."""
        with pytest.raises(TypeError):
            validate_preconditions(batch_size="32")
        
        with pytest.raises(ValueError):
            validate_preconditions(batch_size=0)
        
        with pytest.raises(ValueError):
            validate_preconditions(batch_size=-32)
    
    def test_all_valid(self):
        """Test that all valid parameters pass."""
        model = nn.Linear(10, 5)
        
        class DummyLoader:
            def __len__(self):
                return 10
        
        validate_preconditions(
            model=model,
            data_loader=DummyLoader(),
            epochs=100,
            learning_rate=0.001,
            batch_size=32
        )  # Should not raise


class TestAtomicSaveCheckpoint:
    """Test atomic checkpoint saving."""
    
    def test_atomic_save_success(self):
        """Test that checkpoint is saved atomically."""
        checkpoint = {
            'epoch': 10,
            'model_state_dict': {'weight': torch.randn(5, 3)},
            'loss': 0.5
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "test_checkpoint.pt"
            
            atomic_save_checkpoint(
                checkpoint=checkpoint,
                path=str(checkpoint_path),
                operation_name="test save"
            )
            
            # Verify file exists and is loadable
            assert checkpoint_path.exists()
            loaded = torch.load(checkpoint_path)
            assert loaded['epoch'] == 10
            assert loaded['loss'] == 0.5
    
    def test_atomic_save_creates_directory(self):
        """Test that parent directory is created if needed."""
        checkpoint = {'data': 'test'}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "subdir" / "checkpoint.pt"
            
            atomic_save_checkpoint(
                checkpoint=checkpoint,
                path=str(checkpoint_path)
            )
            
            assert checkpoint_path.exists()


class TestErrorContext:
    """Test error context manager."""
    
    def test_normal_operation(self):
        """Test that normal operations work."""
        with ErrorContext("Test context"):
            result = 1 + 1
        assert result == 2
    
    def test_error_with_context(self):
        """Test that errors include context in logs."""
        with pytest.raises(ValueError):
            with ErrorContext("Test operation"):
                raise ValueError("Test error")


class TestSafeGPUOperationDecorator:
    """Test safe GPU operation decorator."""
    
    def test_normal_function(self):
        """Test that normal function execution works."""
        @safe_gpu_operation
        def simple_function(x):
            return x * 2
        
        result = simple_function(5)
        assert result == 10
    
    def test_oom_in_decorated_function(self):
        """Test that OOM in decorated function is caught."""
        @safe_gpu_operation
        def oom_function():
            raise RuntimeError("CUDA out of memory")
        
        with pytest.raises(RuntimeError) as exc_info:
            oom_function()
        
        assert "out of memory" in str(exc_info.value).lower()
    
    def test_cuda_error_in_decorated_function(self):
        """Test that CUDA errors in decorated function are caught."""
        @safe_gpu_operation
        def cuda_error_function():
            raise RuntimeError("CUDA error occurred")
        
        with pytest.raises(RuntimeError) as exc_info:
            cuda_error_function()
        
        assert "cuda" in str(exc_info.value).lower()


class TestIntegration:
    """Integration tests combining multiple utilities."""
    
    def test_complete_training_pattern(self):
        """Test complete training pattern with all utilities."""
        model = nn.Linear(10, 5)
        device = torch.device("cpu")  # Use CPU for testing
        
        # Validate preconditions
        validate_preconditions(
            model=model,
            epochs=1,
            learning_rate=0.01,
            batch_size=8
        )
        
        # Training with cleanup guard
        with model_cleanup_guard(model):
            model = model.to(device)
            
            with ErrorContext("Training epoch 1"):
                with gpu_safe_operation("Forward pass"):
                    # Simulate training
                    batch = torch.randn(8, 10)
                    output = model(batch)
                    loss = output.sum()
        
        # Test completed without errors
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
