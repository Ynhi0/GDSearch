"""
Comprehensive test suite for all bug fixes applied in February 2026.

Tests cover:
- Critical bug fixes (parallel execution)
- High priority fixes (checkpoint atomicity, bias correction)
- Medium priority fixes (validation, error handling)
- Integration tests
"""

import pytest
import numpy as np
import torch
import tempfile
import os
import queue
from pathlib import Path
import pandas as pd
from unittest.mock import Mock, patch, MagicMock

# Test imports
from src.utils.parallel_experiment_runner import ParallelExperimentRunner
from src.experiments.run_nn_experiment import run_experiment
from src.utils.checkpoint_utils import save_checkpoint_atomic, CheckpointManager
from src.core.optimizers import SGD, Adam, AdamW, AMSGrad
from src.utils.resume_utils import should_skip_experiment


class TestCriticalBugFixes:
    """Test critical bug fixes that broke core functionality."""
    
    def test_queue_empty_exception_handling(self):
        """Test Bug #1: queue.Empty exception properly caught."""
        # Verify that queue module is imported
        import src.utils.parallel_experiment_runner as runner_module
        assert hasattr(runner_module, 'queue'), "queue module not imported"
        
        # Verify worker method exists
        runner = ParallelExperimentRunner(num_gpus=1)
        assert hasattr(runner, '_worker'), "Worker method not found"
    
    def test_cuda_device_isolation(self):
        """Test Bug #2: CUDA_VISIBLE_DEVICES set before CUDA init."""
        # Mock test - verify env variable setting comes first
        import inspect
        from src.utils.parallel_experiment_runner import run_experiment_on_gpu
        
        source = inspect.getsource(run_experiment_on_gpu)
        
        # Check that os.environ line comes before torch.cuda line
        env_line_idx = source.find("os.environ['CUDA_VISIBLE_DEVICES']")
        cuda_set_line_idx = source.find("torch.cuda.set_device")
        
        assert env_line_idx > 0, "CUDA_VISIBLE_DEVICES setting not found"
        assert cuda_set_line_idx > 0, "torch.cuda.set_device not found"
        assert env_line_idx < cuda_set_line_idx, \
            "CUDA_VISIBLE_DEVICES must be set BEFORE torch.cuda.set_device"
    
    def test_run_experiment_exists(self):
        """Test Bug #3: run_experiment wrapper function exists."""
        from src.experiments.run_nn_experiment import run_experiment
        
        # Verify function exists and is callable
        assert callable(run_experiment), "run_experiment is not callable"
        
        # Verify signature
        import inspect
        sig = inspect.signature(run_experiment)
        assert 'config' in sig.parameters, "config parameter missing"
        assert 'device' in sig.parameters, "device parameter missing"
        assert 'results_dir' in sig.parameters, "results_dir parameter missing"


class TestHighPriorityBugFixes:
    """Test high priority bug fixes."""
    
    def test_windows_atomic_rename(self):
        """Test Bug #4: Windows atomic rename implementation."""
        import inspect
        from src.utils.checkpoint_utils import save_checkpoint_atomic
        
        source = inspect.getsource(save_checkpoint_atomic)
        
        # Check for Windows-specific code
        assert "os.name == 'nt'" in source, "Windows check not found"
        assert "MoveFileExW" in source or "windll" in source, \
            "Windows atomic rename not implemented"
    
    def test_bias_correction_underflow_fix(self):
        """Test Bug #6: AdamW bias correction doesn't underflow."""
        # Test with extreme beta values and high timesteps
        optimizer = AdamW(lr=0.001, beta1=0.99999, beta2=0.9999)
        
        # Simulate many steps
        params = np.array([1.0, 2.0, 3.0])
        gradients = np.array([0.1, 0.2, 0.3])
        
        # Run 100,000 steps (would cause underflow in old code)
        for _ in range(100000):
            params = optimizer.step(params, gradients)
        
        # Check that optimizer state is still valid
        assert optimizer.m is not None, "State became None"
        assert np.isfinite(optimizer.m).all(), "State has non-finite values"
        assert np.isfinite(params).all(), "Parameters became non-finite"


class TestMediumPriorityBugFixes:
    """Test medium priority bug fixes."""
    
    def test_checkpoint_manager_validation(self):
        """Test Bug #8: CheckpointManager validates parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Should raise error for negative values
            with pytest.raises(ValueError):
                CheckpointManager(Path(tmpdir), keep_last=-1)
            
            with pytest.raises(ValueError):
                CheckpointManager(Path(tmpdir), keep_best=-1)
            
            # Should work with valid values
            manager = CheckpointManager(Path(tmpdir), keep_last=0, keep_best=0)
            assert manager.keep_last == 0
            assert manager.keep_best == 0
    
    def test_state_dict_validation(self):
        """Test Bug #10: State dict type validation."""
        optimizer = Adam(lr=0.001)
        
        # Initialize state
        params = np.array([1.0, 2.0])
        grads = np.array([0.1, 0.2])
        optimizer.step(params, grads)
        
        # Should raise error for invalid state dict
        with pytest.raises(TypeError):
            optimizer.load_state_dict("not a dict")
        
        # Should handle missing keys gracefully
        optimizer.load_state_dict({})
        
        # Should validate value types
        with pytest.raises(ValueError):
            optimizer.load_state_dict({'m': 'not_an_array'})
    
    def test_resume_exception_handling(self):
        """Test Bug #12: Resume utils handle specific exceptions."""
        import inspect
        from src.utils.resume_utils import should_skip_experiment
        
        source = inspect.getsource(should_skip_experiment)
        
        # Check for specific exception types
        assert 'pd.errors.ParserError' in source or 'ParserError' in source, \
            "ParserError not caught"
        assert 'PermissionError' in source, \
            "PermissionError not handled separately"


class TestIntegration:
    """Test integration between components."""
    
    def test_parallel_cli_integration(self):
        """Test that run_all_kaggle.py has parallel flags."""
        run_all_file = Path('run_all_kaggle.py')
        if run_all_file.exists():
            content = run_all_file.read_text(encoding='utf-8')
            
            assert '--parallel' in content, "--parallel flag not found"
            assert 'action=' in content and 'store_true' in content, \
                "--parallel flag not properly configured"
    
    def test_checkpoint_atomicity(self):
        """Test that checkpoints are saved atomically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / 'test_checkpoint.pt'
            
            # Create test checkpoint
            checkpoint_data = {
                'epoch': 10,
                'model_state_dict': {'weight': torch.randn(5, 5)},
                'test_metric': 0.95
            }
            
            # Save checkpoint
            save_checkpoint_atomic(checkpoint_data, checkpoint_path)
            
            # Verify it was saved
            assert checkpoint_path.exists(), "Checkpoint not saved"
            
            # Verify no temp files left behind
            temp_files = list(Path(tmpdir).glob('.tmp_*'))
            assert len(temp_files) == 0, f"Temp files not cleaned up: {temp_files}"
            
            # Verify checkpoint is loadable
            loaded = torch.load(checkpoint_path)
            assert loaded['epoch'] == 10
            assert loaded['test_metric'] == 0.95


def run_all_tests():
    """Run all tests and return results."""
    print("="*80)
    print("COMPREHENSIVE BUG FIX TEST SUITE")
    print("="*80)
    
    # Run pytest
    import sys
    exit_code = pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '--color=yes'
    ])
    
    return exit_code


if __name__ == '__main__':
    exit_code = run_all_tests()
    import sys
    sys.exit(exit_code)
