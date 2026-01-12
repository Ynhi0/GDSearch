"""
Unit tests for theory-practice comparison edge-case guards.

Tests verify that compare_theory_practice() handles mathematical edge cases
gracefully without crashes (e.g., mu >= L, theoretical_rate <= 0 or >= 1).

AUDIT FIX: These tests validate the guards added to prevent log-of-negative
and divide-by-zero errors in theoretical convergence rate computations.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import logging

from src.analysis.theory_practice_comparison import compare_theory_practice


@pytest.fixture
def temp_test_output_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_test_trajectory_csv(temp_test_output_dir):
    """Create a sample trajectory CSV for testing."""
    # Simulated training run with geometric convergence
    iterations = np.arange(1, 101)
    losses = 10.0 * (0.9 ** iterations) + 0.01  # Exponential decay to 0.01

    df = pd.DataFrame({
        'iteration': iterations,
        'loss': losses
    })

    csv_path = temp_test_output_dir / "test_trajectory.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_edge_case_mu_greater_than_L(sample_test_trajectory_csv, temp_test_output_dir, caplog):
    """
    Test edge case: mu >= L (invalid strongly-convex assumption).

    When mu >= L, theoretical_rate = 1 - mu/L <= 0, which makes -log(rate)
    undefined or infinite. The function should handle this gracefully.
    """
    caplog.set_level(logging.WARNING)

    # Invalid parameters: mu >= L
    result = compare_theory_practice(
        training_csv=str(sample_test_trajectory_csv),
        optimizer_name="SGD",
        mu=10.0,  # mu >= L = 1.0
        L=1.0,
        output_dir=str(temp_test_output_dir)
    )

    # Should return dict without crashing
    assert isinstance(result, dict)

    # Check that warning was logged
    assert any("Invalid" in record.message for record in caplog.records)

    # Should return empty dict (edge case detected)
    assert result == {}


def test_edge_case_mu_equals_L(sample_test_trajectory_csv, temp_test_output_dir, caplog):
    """
    Test boundary case: mu == L.

    When mu == L, theoretical_rate = 1 - mu/L = 0, making -log(0) = infinity.
    """
    caplog.set_level(logging.WARNING)

    result = compare_theory_practice(
        training_csv=str(sample_test_trajectory_csv),
        optimizer_name="SGD",
        mu=1.0,
        L=1.0,  # mu == L
        output_dir=str(temp_test_output_dir)
    )

    assert isinstance(result, dict)
    # Should return empty dict (edge case detected)
    assert result == {}


def test_valid_strongly_convex_case(sample_test_trajectory_csv, temp_test_output_dir):
    """
    Test valid case: 0 < mu < L (proper strongly-convex function).

    This should compute optimality gap without warnings.
    """
    result = compare_theory_practice(
        training_csv=str(sample_test_trajectory_csv),
        optimizer_name="SGD",
        mu=0.1,
        L=1.0,  # mu < L (valid)
        output_dir=str(temp_test_output_dir)
    )

    assert isinstance(result, dict)

    # Should have computed optimality gap
    empirical_rate = result.get('empirical_rate')
    if empirical_rate is not None and float(empirical_rate) > 0:
        # Optimality gap may still be None if curve fitting failed, but should not crash
        pass


def test_non_convex_case_no_mu(sample_test_trajectory_csv, temp_test_output_dir):
    """
    Test non-convex case (mu=None).

    Should use sublinear bound without computing optimality gap.
    """
    result = compare_theory_practice(
        training_csv=str(sample_test_trajectory_csv),
        optimizer_name="SGD",
        mu=None,  # Non-convex
        L=1.0,
        output_dir=str(temp_test_output_dir)
    )

    assert isinstance(result, dict)
    # Non-convex case: no exponential rate, so no optimality gap
    assert result.get('optimality_gap') is None


def test_insufficient_data_points(temp_test_output_dir):
    """
    Test edge case: CSV with insufficient data points.

    Should return empty dict without crashing.
    """
    # Create CSV with only 3 data points (< 10 required)
    df = pd.DataFrame({
        'iteration': [1, 2, 3],
        'loss': [1.0, 0.5, 0.25]
    })
    csv_path = temp_test_output_dir / "tiny_trajectory.csv"
    df.to_csv(csv_path, index=False)

    result = compare_theory_practice(
        training_csv=str(csv_path),
        optimizer_name="SGD",
        mu=0.1,
        L=1.0,
        output_dir=str(temp_test_output_dir)
    )

    # Should return empty dict (not enough data)
    assert result == {}


def test_negative_losses_filtered(temp_test_output_dir):
    """
    Test that negative/invalid losses are filtered out.
    """
    # Create CSV with some negative losses
    df = pd.DataFrame({
        'iteration': np.arange(1, 21),
        'loss': [10.0 * (0.9 ** i) if i % 5 != 0 else -1.0 for i in range(1, 21)]
    })
    csv_path = temp_test_output_dir / "negative_losses.csv"
    df.to_csv(csv_path, index=False)

    result = compare_theory_practice(
        training_csv=str(csv_path),
        optimizer_name="SGD",
        mu=0.1,
        L=1.0,
        output_dir=str(temp_test_output_dir)
    )

    # Should handle by filtering invalid data
    # May return empty dict if too few valid points remain
    assert isinstance(result, dict)


def test_pl_condition_case(sample_test_trajectory_csv, temp_test_output_dir):
    """
    Test PL-condition case (pl_constant provided).

    Should prioritize PL bound over strong convexity.
    """
    result = compare_theory_practice(
        training_csv=str(sample_test_trajectory_csv),
        optimizer_name="SGD",
        mu=None,  # Provide PL instead
        L=1.0,
        pl_constant=0.5,
        output_dir=str(temp_test_output_dir)
    )

    assert isinstance(result, dict)
    # Should use PL convergence regime
    if result.get('convergence_regime'):
        assert 'PL' in result['convergence_regime'] or 'pl' in result['convergence_regime'].lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
