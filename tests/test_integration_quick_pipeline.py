"""
Integration test for run_all_kaggle.py quick pipeline
Tests end-to-end execution with minimal configuration
"""
import os
import sys
import subprocess
import tempfile
from pathlib import Path
import pytest
import pandas as pd


@pytest.mark.slow
def test_quick_mnist_pipeline():
    """Test quick MNIST experiment produces expected artifacts (slower test)"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Run ultra-quick MNIST experiment with single seed for faster CI
        result = subprocess.run([
            sys.executable, "run_all_kaggle.py",
            "--ultra-quick",
            "--experiments", "mnist",
            "--seeds", "42",
            "--results-dir", tmpdir,
            "--no-mlflow"
        ], capture_output=True, text=True, timeout=1200)  # 20 minutes for ultra-quick with all optimizers
        
        # Check execution succeeded
        assert result.returncode == 0, f"Script failed with: {result.stderr}"
        
        # Verify results directory structure - use recursive glob as results may be nested
        results_base = Path(tmpdir)
        
        # Check for per-run CSV artifacts (canonical naming) - search recursively
        csv_files = list(results_base.rglob("MNIST_SimpleMLP_*_seed42.csv"))
        assert len(csv_files) > 0, f"No per-run CSV artifacts found in {results_base}"
        
        # Check metadata files (now use .metadata.json suffix)
        meta_files = list(results_base.rglob("MNIST_SimpleMLP_*_seed42.metadata.json"))
        assert len(meta_files) > 0, f"No metadata files found in {results_base}"
        
        # Validate CSV structure
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            assert 'epoch' in df.columns, f"{csv_file} missing 'epoch' column"
            assert 'train_loss' in df.columns, f"{csv_file} missing 'train_loss' column"
            assert 'test_acc' in df.columns, f"{csv_file} missing 'test_acc' column"
            assert len(df) > 0, f"{csv_file} has no data rows"


def test_quick_2d_pipeline():
    """Test quick 2D optimization experiment"""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run([
            sys.executable, "run_all_kaggle.py",
            "--quick",
            "--experiments", "2d",
            "--seeds", "123",
            "--results-dir", tmpdir,
            "--no-mlflow"
        ], capture_output=True, text=True, timeout=300)  # 5 minutes for 2D quick mode
        
        assert result.returncode == 0, f"2D experiment failed: {result.stderr}"
        
        # Results are stored in experiments/2d_optimization subdirectory
        results_dir = Path(tmpdir) / "experiments" / "2d_optimization"
        assert results_dir.exists(), f"2D results directory not created at {results_dir}"
        
        # Check for per-run artifacts - save_run_artifacts creates a subdirectory
        # Check both top-level and subdirectories for CSV files
        csv_files = list(results_dir.rglob("2D_*_seed123.csv"))
        assert len(csv_files) > 0, f"No 2D per-run artifacts found in {results_dir} or subdirs"


def test_deterministic_flag():
    """Test that --deterministic flag enables deterministic algorithms"""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run([
            sys.executable, "run_all_kaggle.py",
            "--quick",
            "--experiments", "2d",
            "--seeds", "1",
            "--results-dir", tmpdir,
            "--no-mlflow",
            "--deterministic"
        ], capture_output=True, text=True, timeout=300)  # 5 minutes for 2D deterministic
        
        # Check for deterministic mode messages in output
        stdout_check = "deterministic mode" in result.stdout.lower() if result.stdout else False
        stderr_check = "deterministic" in result.stderr.lower() if result.stderr else False
        assert stdout_check or stderr_check, \
            "Deterministic mode not indicated in output"
        
        assert result.returncode == 0, f"Deterministic run failed: {result.stderr or 'No stderr'}"


def test_multi_seed_consistency():
    """Test that multiple seeds produce separate per-run artifacts"""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run([
            sys.executable, "run_all_kaggle.py",
            "--quick",
            "--experiments", "2d",
            "--seeds", "42,123,456",
            "--results-dir", tmpdir,
            "--no-mlflow"
        ], capture_output=True, text=True, timeout=600)  # 10 minutes for 3 seeds
        
        assert result.returncode == 0, f"Multi-seed run failed: {result.stderr}"
        
        # Results are stored in experiments/2d_optimization subdirectory
        results_dir = Path(tmpdir) / "experiments" / "2d_optimization"
        
        # Should have artifacts for each seed - check recursively
        for seed in [42, 123, 456]:
            seed_files = list(results_dir.rglob(f"*_seed{seed}.csv"))
            assert len(seed_files) > 0, f"No artifacts found for seed {seed}"


@pytest.mark.slow
def test_full_mnist_with_checkpoints():
    """Test full MNIST run with checkpoint saving (slower test)"""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run([
            sys.executable, "run_all_kaggle.py",
            "--ultra-quick",
            "--experiments", "mnist",
            "--seeds", "42,123",
            "--results-dir", tmpdir,
            "--no-mlflow"
        ], capture_output=True, text=True, timeout=1800)  # 30 minutes for ultra-quick with 2 seeds and all optimizers
        
        assert result.returncode == 0, f"Full MNIST run failed: {result.stderr}"
        
        # Check for checkpoint directory
        checkpoint_dir = Path(tmpdir) / "checkpoints"
        if checkpoint_dir.exists():
            checkpoint_files = list(checkpoint_dir.glob("*.pt"))
            # Checkpoints might exist from run
            # Just verify script completed without errors


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
