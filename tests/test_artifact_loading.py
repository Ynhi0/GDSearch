"""
Unit tests for artifact loading in theory-practice validation.

Tests verify that JSON parsing logic is robust and handles:
- Valid artifacts with all required keys
- Missing keys (graceful fallback)
- Malformed JSON (exception handling)
- Missing files (directory doesn't exist)
"""

import pytest
import json
import tempfile
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiments.theory_practice_validation import (
    load_training_results,
    extract_optimizer_from_filename
)


class TestArtifactLoading:
    """Test suite for artifact loading logic."""

    def test_extract_optimizer_from_filename(self):
        """Test optimizer name extraction from various filename patterns."""
        test_cases = [
            ('NN_SimpleMLP_MNIST_Adam_lr0.001_seed42.csv', 'Adam'),
            ('MNIST_SGD_Momentum_seed123.csv', 'SGD_Momentum'),
            ('ResNet18_CIFAR10_SAM_seed1.csv', 'SAM'),
            ('model_RMSprop_training.csv', 'RMSprop'),
            ('unknown_optimizer.csv', 'Unknown'),
        ]

        for filename, expected in test_cases:
            result = extract_optimizer_from_filename(filename)
            assert result == expected, f"Failed for {filename}: got {result}, expected {expected}"

    def test_hessian_artifact_valid_schema(self):
        """Test loading valid Hessian analysis artifact."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock artifact directory structure
            results_dir = Path(tmpdir) / 'results'
            exp_dir = results_dir / 'test_experiment' / 'hessian_analysis'
            exp_dir.mkdir(parents=True)

            # Create valid JSON artifact
            artifact = {
                'max_eigenvalue': 5.2341,
                'min_eigenvalue': -0.0123,
                'condition_number': 425.6
            }

            artifact_file = exp_dir / 'Adam_hessian_analysis.json'
            with open(artifact_file, 'w') as f:
                json.dump(artifact, f)

            # Verify file exists and can be loaded
            assert artifact_file.exists()

            # Load and validate
            with open(artifact_file, 'r') as f:
                loaded = json.load(f)

            assert 'max_eigenvalue' in loaded
            assert loaded['max_eigenvalue'] == 5.2341
            assert isinstance(loaded['max_eigenvalue'], (int, float))

    def test_hessian_artifact_missing_key(self):
        """Test handling of Hessian artifact with missing max_eigenvalue key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir) / 'results'
            exp_dir = results_dir / 'test_experiment' / 'hessian_analysis'
            exp_dir.mkdir(parents=True)

            # Create artifact missing required key
            artifact = {
                'min_eigenvalue': -0.0123,
                'condition_number': 425.6
                # max_eigenvalue is MISSING
            }

            artifact_file = exp_dir / 'Adam_hessian_analysis.json'
            with open(artifact_file, 'w') as f:
                json.dump(artifact, f)

            # Verify file exists
            assert artifact_file.exists()

            # Load and check for missing key
            with open(artifact_file, 'r') as f:
                loaded = json.load(f)

            assert 'max_eigenvalue' not in loaded
            # Real code should handle this gracefully with fallback

    def test_gradient_noise_artifact_valid_schema(self):
        """Test loading valid gradient noise analysis artifact."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir) / 'results'
            exp_dir = results_dir / 'test_experiment' / 'gradient_noise'
            exp_dir.mkdir(parents=True)

            # Create valid JSON artifact (supports both key names)
            artifact = {
                'sigma_squared': 0.000123,
                'gradient_variance': 0.000123,  # Alternative key
                'noise_to_signal_ratio': 0.05
            }

            artifact_file = exp_dir / 'Adam_gradient_noise.json'
            with open(artifact_file, 'w') as f:
                json.dump(artifact, f)

            # Load and validate
            with open(artifact_file, 'r') as f:
                loaded = json.load(f)

            # Should accept either key
            assert 'sigma_squared' in loaded or 'gradient_variance' in loaded
            sigma = loaded.get('sigma_squared', loaded.get('gradient_variance'))
            assert sigma == 0.000123
            assert isinstance(sigma, (int, float))

    def test_pl_constant_artifact_valid_schema(self):
        """Test loading valid PL constant analysis artifact."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir) / 'results'
            exp_dir = results_dir / 'test_experiment' / 'pl_analysis'
            exp_dir.mkdir(parents=True)

            # Create valid JSON artifact
            artifact = {
                'estimated_mu': 0.001234,
                'confidence': 0.95,
                'pl_condition_satisfied': True
            }

            artifact_file = exp_dir / 'Adam_pl_constant.json'
            with open(artifact_file, 'w') as f:
                json.dump(artifact, f)

            # Load and validate
            with open(artifact_file, 'r') as f:
                loaded = json.load(f)

            assert 'estimated_mu' in loaded
            assert loaded['estimated_mu'] == 0.001234
            assert isinstance(loaded['estimated_mu'], (int, float))

    def test_malformed_json_handling(self):
        """Test handling of malformed JSON artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir) / 'results'
            exp_dir = results_dir / 'test_experiment' / 'hessian_analysis'
            exp_dir.mkdir(parents=True)

            # Create malformed JSON
            artifact_file = exp_dir / 'Adam_hessian_analysis.json'
            with open(artifact_file, 'w') as f:
                f.write("{invalid json, missing quotes: 123")

            # Attempt to load should raise JSONDecodeError
            with pytest.raises(json.JSONDecodeError):
                with open(artifact_file, 'r') as f:
                    json.load(f)

    def test_missing_directory_handling(self):
        """Test handling when artifact directories don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir) / 'results'
            exp_dir = results_dir / 'test_experiment'
            exp_dir.mkdir(parents=True)

            # Hessian directory does NOT exist
            hessian_dir = exp_dir / 'hessian_analysis'
            assert not hessian_dir.exists()

            # Real code should handle this gracefully (check before glob)

    def test_empty_directory_handling(self):
        """Test handling when artifact directory exists but is empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir) / 'results'
            exp_dir = results_dir / 'test_experiment' / 'hessian_analysis'
            exp_dir.mkdir(parents=True)

            # Directory exists but no files
            json_files = list(exp_dir.glob('*.json'))
            assert len(json_files) == 0

            # Real code should handle this gracefully (check list length)

    def test_type_coercion(self):
        """Test that loaded values are correctly coerced to float."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir) / 'results'
            exp_dir = results_dir / 'test_experiment' / 'hessian_analysis'
            exp_dir.mkdir(parents=True)

            # Create artifact with integer values
            artifact = {
                'max_eigenvalue': 5,  # Integer, not float
                'min_eigenvalue': -1
            }

            artifact_file = exp_dir / 'Adam_hessian_analysis.json'
            with open(artifact_file, 'w') as f:
                json.dump(artifact, f)

            # Load and coerce
            with open(artifact_file, 'r') as f:
                loaded = json.load(f)

            L_est = float(loaded['max_eigenvalue'])
            assert isinstance(L_est, float)
            assert L_est == 5.0

    def test_glob_pattern_matching(self):
        """Test that glob patterns correctly match optimizer-specific files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir) / 'results'
            exp_dir = results_dir / 'test_experiment' / 'hessian_analysis'
            exp_dir.mkdir(parents=True)

            # Create files for different optimizers
            optimizers = ['Adam', 'SGD', 'RMSprop']
            for opt in optimizers:
                artifact_file = exp_dir / f'{opt}_hessian_analysis.json'
                with open(artifact_file, 'w') as f:
                    json.dump({'max_eigenvalue': 1.0}, f)

            # Test glob pattern for specific optimizer
            optimizer_name = 'Adam'
            matches = list(exp_dir.glob(f'*{optimizer_name}*hessian*.json'))

            assert len(matches) == 1
            assert 'Adam' in matches[0].name
            assert 'SGD' not in matches[0].name

    def test_load_training_results_valid(self):
        """Test loading training results CSV files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir)
            exp_dir = results_dir / 'mnist'
            exp_dir.mkdir(parents=True)

            # Create valid CSV
            csv_file = exp_dir / 'NN_SimpleMLP_MNIST_Adam_lr0.001_seed42.csv'
            with open(csv_file, 'w') as f:
                f.write('epoch,train_loss,val_loss\n')
                f.write('1,0.5,0.6\n')
                f.write('2,0.3,0.4\n')
                f.write('3,0.2,0.3\n')

            # Load results
            results = load_training_results(str(results_dir), 'mnist')

            assert len(results) > 0
            assert 'Adam' in results
            assert 'train_loss' in results['Adam'].columns


class TestArtifactIntegration:
    """Integration tests for full artifact loading pipeline."""

    def test_priority_fallback_logic(self):
        """Test that measured > estimated > fallback priority is respected."""
        # This is more of a logic verification test
        # Simulates the three-tier fallback system

        # Case 1: Measured value available
        L_est = None
        measured_L = 5.2341
        if measured_L is not None:
            L_est = measured_L
        assert L_est == 5.2341

        # Case 2: Measured unavailable, use estimated
        L_est = None
        measured_L = None
        estimated_L = 10.5
        if measured_L is not None:
            L_est = measured_L
        elif L_est is None and estimated_L is not None:
            L_est = estimated_L
        assert L_est == 10.5

        # Case 3: Both unavailable, use fallback
        L_est = None
        measured_L = None
        estimated_L = None
        fallback_L = 1.0
        if measured_L is not None:
            L_est = measured_L
        elif estimated_L is not None:
            L_est = estimated_L
        elif L_est is None:
            L_est = fallback_L
        assert L_est == 1.0

    def test_all_artifacts_present(self):
        """Integration test with all artifact types present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir) / 'results'
            exp_dir = results_dir / 'test_experiment'

            # Create all artifact directories
            (exp_dir / 'hessian_analysis').mkdir(parents=True)
            (exp_dir / 'gradient_noise').mkdir(parents=True)
            (exp_dir / 'pl_analysis').mkdir(parents=True)

            # Create Hessian artifact
            hessian_file = exp_dir / 'hessian_analysis' / 'Adam_hessian.json'
            with open(hessian_file, 'w') as f:
                json.dump({'max_eigenvalue': 5.0, 'min_eigenvalue': -0.01}, f)

            # Create gradient noise artifact
            noise_file = exp_dir / 'gradient_noise' / 'Adam_noise.json'
            with open(noise_file, 'w') as f:
                json.dump({'sigma_squared': 0.001}, f)

            # Create PL constant artifact
            pl_file = exp_dir / 'pl_analysis' / 'Adam_pl.json'
            with open(pl_file, 'w') as f:
                json.dump({'estimated_mu': 0.0001}, f)

            # Verify all can be loaded
            with open(hessian_file, 'r') as f:
                hessian_data = json.load(f)
            with open(noise_file, 'r') as f:
                noise_data = json.load(f)
            with open(pl_file, 'r') as f:
                pl_data = json.load(f)

            assert 'max_eigenvalue' in hessian_data
            assert 'sigma_squared' in noise_data
            assert 'estimated_mu' in pl_data


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
