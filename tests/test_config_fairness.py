"""
Test configuration fairness to ensure unbiased optimizer comparisons.

This test suite validates that all optimizers get equal hyperparameter search ranges,
preventing biased comparisons that could invalidate experimental conclusions.

Per review requirement: "Baseline Fairness - Are the search spaces symmetric?"
"""

import json
import os
import pytest
from pathlib import Path


def load_config(config_path):
    """Load JSON configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


class TestConfigFairness:
    """Test suite for configuration fairness validation."""

    @pytest.fixture
    def config_dir(self):
        """Get configs directory path."""
        return Path(__file__).parent.parent / 'configs'

    def test_nn_tuning_lr_symmetry(self, config_dir):
        """
        Test that all optimizers in nn_tuning.json have equal LR search ranges.

        REVIEW REQUIREMENT: Prevent bias where one optimizer gets extensive tuning
        while another gets a single fixed learning rate.
        """
        config = load_config(config_dir / 'nn_tuning.json')

        lr_counts = {}
        for sweep in config.get('sweeps', []):
            optimizer = sweep['optimizer']
            lr_values = sweep.get('lr_values', [])
            lr_counts[optimizer] = len(lr_values)

        # All optimizers should have at least 3 LR values
        for optimizer, count in lr_counts.items():
            assert count >= 3, \
                f"Optimizer '{optimizer}' has only {count} LR values (need ≥3 for fair comparison)"

        # Check variance (no optimizer should have 10x more values than another)
        if lr_counts:
            min_count = min(lr_counts.values())
            max_count = max(lr_counts.values())
            ratio = max_count / max(min_count, 1)

            assert ratio <= 3.0, \
                f"LR search space imbalance detected: {lr_counts}. " \
                f"Max/min ratio = {ratio:.1f} (should be ≤3.0)"

    def test_cifar10_tuning_lr_symmetry(self, config_dir):
        """Test LR symmetry in CIFAR-10 config."""
        config = load_config(config_dir / 'cifar10_tuning.json')

        lr_counts = {}
        for sweep in config.get('sweeps', []):
            optimizer = sweep['optimizer']
            lr_values = sweep.get('lr_values', [])
            lr_counts[optimizer] = len(lr_values)

        # All optimizers should have at least 3 LR values
        for optimizer, count in lr_counts.items():
            assert count >= 3, \
                f"CIFAR-10: Optimizer '{optimizer}' has only {count} LR values (need ≥3)"

        # Check balance
        if lr_counts:
            min_count = min(lr_counts.values())
            max_count = max(lr_counts.values())
            ratio = max_count / max(min_count, 1)

            assert ratio <= 3.0, \
                f"CIFAR-10 LR imbalance: {lr_counts}. Ratio = {ratio:.1f} (should be ≤3.0)"

    def test_momentum_parameter_symmetry(self, config_dir):
        """
        Test that momentum/beta parameters have sufficient range.

        REQUIREMENT: Optimizers with momentum should explore the parameter space,
        not just use a single hardcoded value.
        """
        config = load_config(config_dir / 'cifar10_tuning.json')

        for sweep in config.get('sweeps', []):
            optimizer = sweep['optimizer']

            # Check momentum-based optimizers
            if 'momentum' in optimizer.lower() or optimizer == 'SGD_Momentum':
                momentum_values = sweep.get('momentum_values', [])
                assert len(momentum_values) >= 3, \
                    f"{optimizer}: momentum_values has {len(momentum_values)} values (need ≥3)"

            # Check Adam beta parameters
            if optimizer == 'Adam':
                beta1_values = sweep.get('beta1_values', [0.9])  # Default if missing
                beta2_values = sweep.get('beta2_values', [0.999])

                # At least one beta should be explored
                total_beta_variations = len(beta1_values) * len(beta2_values)
                assert total_beta_variations >= 2, \
                    f"Adam: Insufficient beta exploration (beta1={len(beta1_values)}, beta2={len(beta2_values)})"

    def test_weight_decay_symmetry(self, config_dir):
        """Test that weight decay ranges are comparable across optimizers."""
        config = load_config(config_dir / 'cifar10_tuning.json')

        wd_counts = {}
        for sweep in config.get('sweeps', []):
            optimizer = sweep['optimizer']
            wd_values = sweep.get('weight_decay_values', [])
            if wd_values:  # Only if weight decay is used
                wd_counts[optimizer] = len(wd_values)

        # If weight decay is used, should have at least 3 values
        for optimizer, count in wd_counts.items():
            assert count >= 3, \
                f"{optimizer}: weight_decay_values has only {count} values (need ≥3)"

    def test_epoch_budget_equality(self, config_dir):
        """
        Test that all optimizers get equal training epochs during tuning.

        Important: If one optimizer trains for 10 epochs and another for 1,
        the comparison is meaningless.
        """
        for config_file in ['nn_tuning.json', 'cifar10_tuning.json']:
            config = load_config(config_dir / config_file)

            epoch_counts = {}
            for sweep in config.get('sweeps', []):
                optimizer = sweep['optimizer']
                epochs = sweep.get('epochs', 0)
                epoch_counts[optimizer] = epochs

            # All optimizers should train for the same number of epochs
            unique_epochs = set(epoch_counts.values())
            assert len(unique_epochs) <= 1, \
                f"{config_file}: Unequal epoch budgets detected: {epoch_counts}. " \
                f"All optimizers must train for the same number of epochs during tuning."

    def test_no_missing_required_fields(self, config_dir):
        """Test that all sweeps have required fields."""
        for config_file in ['nn_tuning.json', 'cifar10_tuning.json']:
            config = load_config(config_dir / config_file)

            for i, sweep in enumerate(config.get('sweeps', [])):
                # Required fields
                assert 'optimizer' in sweep, f"{config_file} sweep {i}: Missing 'optimizer'"
                assert 'lr_values' in sweep, f"{config_file} sweep {i}: Missing 'lr_values'"
                assert 'epochs' in sweep, f"{config_file} sweep {i}: Missing 'epochs'"

                # Should have at least one hyperparameter to tune
                tunable_keys = [k for k in sweep.keys() if k.endswith('_values')]
                assert len(tunable_keys) >= 1, \
                    f"{config_file} sweep {i} ({sweep['optimizer']}): No tunable parameters"

    def test_benchmark_config_exists(self, config_dir):
        """Test that benchmark hyperparameters config exists and is valid."""
        benchmark_path = config_dir / 'benchmark_hyperparameters.json'

        # Allow missing file (not all experiments may have been run)
        if benchmark_path.exists():
            config = load_config(benchmark_path)

            # Should contain optimizer configurations
            assert len(config) > 0, "benchmark_hyperparameters.json is empty"

            # Benchmark config has nested structure: experiment_name -> optimizers -> params
            # Validate that nested optimizer configs have lr
            for experiment_name, experiment_config in config.items():
                if isinstance(experiment_config, dict) and 'optimizers' in experiment_config:
                    optimizers = experiment_config['optimizers']
                    for optimizer, params in optimizers.items():
                        assert 'lr' in params, \
                            f"benchmark_hyperparameters.json[{experiment_name}]: {optimizer} missing 'lr' parameter"


class TestConfigStructure:
    """Test configuration file structure and consistency."""

    @pytest.fixture
    def config_dir(self):
        """Get configs directory path."""
        return Path(__file__).parent.parent / 'configs'

    def test_convergence_criteria_present(self, config_dir):
        """Test that convergence criteria are defined."""
        for config_file in ['nn_tuning.json', 'cifar10_tuning.json']:
            config = load_config(config_dir / config_file)

            assert 'convergence' in config, \
                f"{config_file}: Missing 'convergence' section"

            convergence = config['convergence']
            assert 'grad_norm_threshold' in convergence
            assert 'loss_delta_threshold' in convergence
            assert 'loss_window' in convergence

    def test_seed_specified(self, config_dir):
        """Test that random seeds are specified for reproducibility and statistical validity."""
        for config_file in ['nn_tuning.json', 'cifar10_tuning.json']:
            config = load_config(config_dir / config_file)

            assert 'seeds' in config, \
                f"{config_file}: Missing 'seeds' for reproducibility and statistical validity"
            assert isinstance(config['seeds'], list), \
                f"{config_file}: 'seeds' must be a list of integers"
            assert len(config['seeds']) >= 3, \
                f"{config_file}: Need at least 3 seeds for multi-seed experiments (found {len(config['seeds'])})"

    def test_batch_size_reasonable(self, config_dir):
        """Test that batch sizes are reasonable."""
        for config_file in ['nn_tuning.json', 'cifar10_tuning.json']:
            config = load_config(config_dir / config_file)

            batch_size = config.get('batch_size', 0)
            assert 16 <= batch_size <= 1024, \
                f"{config_file}: batch_size={batch_size} outside reasonable range [16, 1024]"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
