"""
Test edge cases for statistical analysis and aggregation functions.

Tests for:
- n=1 samples (cannot compute CI, but should not crash)
- Empty arrays (should return NaN gracefully)
- All-tainted runs (should return NaN with n=0)
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from src.analysis.statistical_analysis import (
    compare_optimizers_ttest,
    extract_final_metric
)
from src.experiments.run_multi_seed import aggregate_results


class TestEdgeCases:
    """Test statistical functions with edge cases."""
    
    def test_ttest_single_sample(self):
        """Test T-test with n=1 samples - should not crash."""
        results_A = np.array([95.5])
        results_B = np.array([93.2])
        
        # Should not crash
        result = compare_optimizers_ttest(
            results_A, results_B,
            name_A="Optimizer A",
            name_B="Optimizer B"
        )
        
        # Should have valid means
        assert result['mean_A'] == 95.5
        assert result['mean_B'] == 93.2
        assert result['n_A'] == 1
        assert result['n_B'] == 1
        
        # CI should collapse to mean for n=1
        assert result['ci_A'][0] == result['ci_A'][1] == 95.5
        assert result['ci_B'][0] == result['ci_B'][1] == 93.2
        
        print("✓ T-test with n=1 handled gracefully")
    
    def test_ttest_zero_variance(self):
        """Test T-test with zero variance (all samples identical)."""
        results_A = np.array([95.0, 95.0, 95.0])
        results_B = np.array([93.0, 93.0, 93.0])
        
        result = compare_optimizers_ttest(
            results_A, results_B,
            name_A="Optimizer A",
            name_B="Optimizer B"
        )
        
        # Should detect zero variance and handle it
        assert result['std_A'] == 0.0
        assert result['std_B'] == 0.0
        assert not np.isnan(result['p_value'])
        
        print("✓ T-test with zero variance handled correctly")
    
    def test_aggregate_empty_results(self):
        """Test aggregation with no valid results (all tainted or missing)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create CSV files with all tainted runs
            files = []
            for i in range(3):
                filepath = Path(temp_dir) / f"run_{i}.csv"
                df = pd.DataFrame({
                    'phase': ['train', 'eval', 'eval'],
                    'epoch': [1, 1, 2],
                    'test_accuracy': [70.0, 75.0, 80.0],
                    'tainted': [True, True, True]  # All tainted
                })
                df.to_csv(filepath, index=False)
                files.append(str(filepath))
            
            # Aggregate with exclude_tainted=True
            result = aggregate_results(files, metric='test_accuracy', exclude_tainted=True)
            
            # Should return NaN values for empty aggregation
            assert np.isnan(result['mean'])
            assert np.isnan(result['std'])
            assert np.isnan(result['min'])
            assert np.isnan(result['max'])
            assert result['n'] == 0
            assert len(result['values']) == 0
            
            print("✓ Empty aggregation returns NaN gracefully")
    
    def test_aggregate_mixed_tainted_and_valid(self):
        """Test aggregation with mix of tainted and valid runs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            files = []
            
            # Create 2 tainted runs
            for i in range(2):
                filepath = Path(temp_dir) / f"tainted_{i}.csv"
                df = pd.DataFrame({
                    'phase': ['train', 'eval'],
                    'epoch': [1, 1],
                    'test_accuracy': [70.0, 75.0],
                    'tainted': [True, True]
                })
                df.to_csv(filepath, index=False)
                files.append(str(filepath))
            
            # Create 3 valid runs
            for i in range(3):
                filepath = Path(temp_dir) / f"valid_{i}.csv"
                df = pd.DataFrame({
                    'phase': ['train', 'eval'],
                    'epoch': [1, 1],
                    'test_accuracy': [70.0, 85.0 + i],  # 85, 86, 87
                    'tainted': [False, False]
                })
                df.to_csv(filepath, index=False)
                files.append(str(filepath))
            
            # Aggregate with exclude_tainted=True
            result = aggregate_results(files, metric='test_accuracy', exclude_tainted=True)
            
            # Should only include valid runs
            assert result['n'] == 3
            assert len(result['values']) == 3
            assert result['mean'] == 86.0  # (85 + 86 + 87) / 3
            assert not np.isnan(result['std'])
            
            print("✓ Mixed tainted/valid aggregation works correctly")
    
    def test_extract_final_metric_empty_dataframes(self):
        """Test extract_final_metric with empty DataFrames."""
        empty_dfs = []
        
        result = extract_final_metric(empty_dfs, metric='test_accuracy')
        
        # Should return empty array
        assert len(result) == 0
        assert isinstance(result, np.ndarray)
        
        print("✓ Empty DataFrame extraction handled gracefully")
    
    def test_extract_final_metric_all_tainted(self):
        """Test extract_final_metric with all tainted DataFrames."""
        tainted_dfs = []
        for i in range(3):
            df = pd.DataFrame({
                'phase': ['train', 'eval'],
                'epoch': [1, 1],
                'test_accuracy': [70.0, 85.0],
                'tainted': [True, True]
            })
            tainted_dfs.append(df)
        
        result = extract_final_metric(
            tainted_dfs, 
            metric='test_accuracy',
            exclude_tainted=True
        )
        
        # Should return empty array (all excluded)
        assert len(result) == 0
        
        print("✓ All-tainted extraction returns empty array")
    
    def test_ttest_identical_samples(self):
        """Test T-test with identical distributions."""
        results_A = np.array([85.0, 86.0, 87.0])
        results_B = np.array([85.0, 86.0, 87.0])
        
        result = compare_optimizers_ttest(
            results_A, results_B,
            name_A="Optimizer A",
            name_B="Optimizer B"
        )
        
        # Should show no significant difference
        assert result['mean_A'] == result['mean_B']
        assert result['p_value'] >= 0.05  # Not significant
        assert abs(result['cohens_d']) < 0.01  # Negligible effect size
        
        print("✓ Identical samples produce expected results")


class TestTaintIntegrity:
    """Test that taint tracking maintains integrity."""
    
    def test_taint_flag_propagation(self):
        """Test that taint flag is properly propagated."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create run with partial tainting (OOM in middle of training)
            filepath = Path(temp_dir) / "partially_tainted.csv"
            df = pd.DataFrame({
                'phase': ['train'] * 5 + ['eval'],
                'epoch': [1, 2, 3, 4, 5, 5],
                'test_accuracy': [70.0, 75.0, 80.0, 82.0, 84.0, 85.0],
                'tainted': [False, False, True, True, True, True]  # OOM at epoch 3
            })
            df.to_csv(filepath, index=False)
            
            # Aggregate should exclude this run
            result = aggregate_results([str(filepath)], 
                                     metric='test_accuracy',
                                     exclude_tainted=True)
            
            assert result['n'] == 0  # Run excluded due to tainting
            assert np.isnan(result['mean'])
            
            print("✓ Taint flag properly propagates and excludes compromised runs")
    
    def test_taint_exclusion_is_default(self):
        """Verify that excluding tainted runs is the default behavior."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "tainted.csv"
            df = pd.DataFrame({
                'phase': ['eval'],
                'epoch': [1],
                'test_accuracy': [85.0],
                'tainted': [True]
            })
            df.to_csv(filepath, index=False)
            
            # Default should exclude tainted
            result = aggregate_results([str(filepath)], metric='test_accuracy')
            assert result['n'] == 0
            
            # Explicit include should work
            result_include = aggregate_results([str(filepath)], 
                                             metric='test_accuracy',
                                             exclude_tainted=False)
            assert result_include['n'] == 1
            
            print("✓ Taint exclusion is default, can be overridden")


if __name__ == '__main__':
    # Run tests
    print("\n" + "="*80)
    print("EDGE CASE TESTS")
    print("="*80 + "\n")
    
    test_edge = TestEdgeCases()
    test_edge.test_ttest_single_sample()
    test_edge.test_ttest_zero_variance()
    test_edge.test_aggregate_empty_results()
    test_edge.test_aggregate_mixed_tainted_and_valid()
    test_edge.test_extract_final_metric_empty_dataframes()
    test_edge.test_extract_final_metric_all_tainted()
    test_edge.test_ttest_identical_samples()
    
    print("\n" + "="*80)
    print("TAINT INTEGRITY TESTS")
    print("="*80 + "\n")
    
    test_taint = TestTaintIntegrity()
    test_taint.test_taint_flag_propagation()
    test_taint.test_taint_exclusion_is_default()
    
    print("\n" + "="*80)
    print("ALL EDGE CASE TESTS PASSED!")
    print("="*80)
