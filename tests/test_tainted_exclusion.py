"""
Unit tests for tainted run exclusion in analysis pipelines.
Ensures that runs marked as tainted (OOM recovery) are properly excluded.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from src.experiments.run_multi_seed import aggregate_results
from src.analysis.statistical_analysis import extract_final_metric


def test_aggregate_results_excludes_tainted_runs():
    """Test that aggregate_results excludes runs with tainted=True."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test CSVs: 3 seeds, one is tainted
        files = []
        for seed, (acc, tainted) in enumerate([(0.95, False), (0.96, True), (0.94, False)], start=1):
            df = pd.DataFrame({
                'phase': ['train'] * 5 + ['eval'],
                'epoch': list(range(1, 6)) + [5],
                'test_accuracy': [0.5 + i*0.1 for i in range(5)] + [acc],
                'tainted': [tainted] * 6
            })
            filepath = os.path.join(tmpdir, f'test_seed{seed}.csv')
            df.to_csv(filepath, index=False)
            files.append(filepath)

        # Aggregate with exclude_tainted=True (default)
        results = aggregate_results(files, metric='test_accuracy', exclude_tainted=True)

        # Should only include seeds 1 and 3 (not seed 2 which is tainted)
        assert results['n'] == 2, f"Expected 2 runs, got {results['n']}"
        assert 0.96 not in results['values'], "Tainted run (0.96) should be excluded"
        assert 0.95 in results['values'], "Clean run (0.95) should be included"
        assert 0.94 in results['values'], "Clean run (0.94) should be included"


def test_aggregate_results_includes_tainted_when_disabled():
    """Test that aggregate_results includes tainted runs when exclude_tainted=False."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test CSVs
        files = []
        for seed, (acc, tainted) in enumerate([(0.95, False), (0.96, True), (0.94, False)], start=1):
            df = pd.DataFrame({
                'phase': ['train'] * 5 + ['eval'],
                'epoch': list(range(1, 6)) + [5],
                'test_accuracy': [0.5 + i*0.1 for i in range(5)] + [acc],
                'tainted': [tainted] * 6
            })
            filepath = os.path.join(tmpdir, f'test_seed{seed}.csv')
            df.to_csv(filepath, index=False)
            files.append(filepath)

        # Aggregate with exclude_tainted=False
        results = aggregate_results(files, metric='test_accuracy', exclude_tainted=False)

        # Should include all 3 runs
        assert results['n'] == 3, f"Expected 3 runs, got {results['n']}"
        assert 0.96 in results['values'], "Tainted run should be included when exclude_tainted=False"


def test_extract_final_metric_excludes_tainted():
    """Test that extract_final_metric excludes tainted runs from DataFrame list."""
    # Create test DataFrames
    dfs = []
    for acc, tainted in [(0.95, False), (0.96, True), (0.94, False)]:
        df = pd.DataFrame({
            'phase': ['train'] * 5 + ['eval'],
            'epoch': list(range(1, 6)) + [5],
            'test_accuracy': [0.5 + i*0.1 for i in range(5)] + [acc],
            'tainted': [tainted] * 6
        })
        dfs.append(df)

    # Extract with exclude_tainted=True (default)
    values = extract_final_metric(dfs, metric='test_accuracy', exclude_tainted=True)

    # Should only include 2 runs
    assert len(values) == 2, f"Expected 2 values, got {len(values)}"
    assert 0.96 not in values, "Tainted run should be excluded"
    assert 0.95 in values and 0.94 in values, "Clean runs should be included"


def test_aggregate_results_empty_when_all_tainted():
    """Test that aggregate_results returns n=0 when all runs are tainted."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test CSVs where all runs are tainted
        files = []
        for seed in range(3):
            df = pd.DataFrame({
                'phase': ['train'] * 5 + ['eval'],
                'epoch': list(range(1, 6)) + [5],
                'test_accuracy': [0.5 + i*0.1 for i in range(5)] + [0.95],
                'tainted': [True] * 6
            })
            filepath = os.path.join(tmpdir, f'test_seed{seed}.csv')
            df.to_csv(filepath, index=False)
            files.append(filepath)

        # Should return empty results
        results = aggregate_results(files, metric='test_accuracy', exclude_tainted=True)
        assert results['n'] == 0, "Should have 0 runs when all are tainted"
        assert np.isnan(results['mean']), "Mean should be NaN when no valid runs"


def test_aggregate_results_handles_string_boolean_tainted():
    """Test that aggregate_results correctly parses string boolean values for tainted column."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test CSVs with string boolean values
        files = []
        test_cases = [
            (0.95, "false"),  # Should be included
            (0.96, "true"),   # Should be excluded
            (0.94, "False"),  # Should be included (case insensitive)
            (0.93, "1"),      # Should be excluded
            (0.92, "0"),      # Should be included
            (0.91, "yes"),    # Should be excluded
            (0.90, "no"),     # Should be included
        ]
        for seed, (acc, tainted_str) in enumerate(test_cases, start=1):
            df = pd.DataFrame({
                'phase': ['train'] * 5 + ['eval'],
                'epoch': list(range(1, 6)) + [5],
                'test_accuracy': [0.5 + i*0.1 for i in range(5)] + [acc],
                'tainted': [tainted_str] * 6
            })
            filepath = os.path.join(tmpdir, f'test_seed{seed}.csv')
            df.to_csv(filepath, index=False)
            files.append(filepath)

        # Aggregate with exclude_tainted=True
        results = aggregate_results(files, metric='test_accuracy', exclude_tainted=True)

        # Should include: false, False, 0, no (0.95, 0.94, 0.92, 0.90)
        # Exclude: true, 1, yes (0.96, 0.93, 0.91)
        expected_values = [0.95, 0.94, 0.92, 0.90]
        assert results['n'] == 4, f"Expected 4 runs, got {results['n']}"
        for val in expected_values:
            assert val in results['values'], f"Expected {val} to be included"
        excluded_values = [0.96, 0.93, 0.91]
        for val in excluded_values:
            assert val not in results['values'], f"Expected {val} to be excluded"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
