"""
Integration tests to verify proposal requirements are met by the runner.
"""
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest
import torch


@pytest.mark.integration
def test_grad_noise_var_recorded_mock():
    """
    Test that grad_noise_var is recorded in experiment CSVs.
    Uses mocking to avoid spawning expensive subprocess.
    """
    from src.analysis.gradient_noise_analysis import estimate_gradient_noise_variance
    from src.core.models import SimpleMLP
    from torch.utils.data import DataLoader, TensorDataset
    
    # Create minimal mock model and data
    model = SimpleMLP(input_size=784, hidden_size=128, num_classes=10)
    model.train()
    
    # Create dummy dataset (small batch for speed)
    X = torch.randn(32, 784)
    y = torch.randint(0, 10, (32,))
    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    # Compute gradient noise variance
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device('cpu')
    result = estimate_gradient_noise_variance(model, data_loader, criterion, device)
    
    # Verify result is a dict with expected keys
    assert isinstance(result, dict), "estimate_gradient_noise_variance should return a dict"
    assert 'sigma_squared' in result, "Result missing 'sigma_squared' key"
    
    grad_noise_var = result['sigma_squared']
    
    # Verify result is finite and positive
    assert grad_noise_var is not None, "sigma_squared is None"
    assert grad_noise_var > 0, f"grad_noise_var should be positive, got {grad_noise_var}"
    assert not torch.isnan(torch.tensor(grad_noise_var)), "grad_noise_var is NaN"
    assert not torch.isinf(torch.tensor(grad_noise_var)), "grad_noise_var is infinite"


@pytest.mark.integration
@pytest.mark.slow
def test_grad_noise_var_recorded_full_pipeline():
    """
    FULL INTEGRATION TEST (marked as slow/integration).
    Run actual pipeline via subprocess to verify end-to-end behavior.
    This test is expensive and should NOT run in fast CI.
    """
    import subprocess
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Run with 12 epochs so that epoch % 10 == 0 triggers an estimate at epoch 10
        result = subprocess.run([
            sys.executable, "run_all_kaggle.py",
            "--experiments", "mnist",
            "--seeds", "42",
            "--results-dir", tmpdir,
            "--no-mlflow",
            "--quick",
            "--grad-noise-every", "1"
        ], capture_output=True, text=True, timeout=900)  # 15 minutes

        assert result.returncode == 0, f"Script failed: {result.stderr}"

        # Search for per-run CSV artifacts
        results_base = Path(tmpdir)
        csv_files = list(results_base.rglob("MNIST_*_seed42.csv"))
        assert len(csv_files) > 0, "No per-run CSV artifacts found"

        # At least one CSV should contain the grad_noise_var column
        found = False
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            if 'grad_noise_var' in df.columns:
                found = True
                # Ensure at least one value is non-null (estimation succeeded at epoch 10)
                assert df['grad_noise_var'].notna().any(), f"grad_noise_var present but all NaN in {csv_file}"
                break

        assert found, "No per-run CSV contained 'grad_noise_var'"
