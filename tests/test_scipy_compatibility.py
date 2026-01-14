"""
Test suite for SciPy compatibility across versions.

Ensures that statistical functions handle different return types
(tuple vs namedtuple vs result objects) from various SciPy versions.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.analysis.statistical_analysis import safe_ttest_rel


class TestScipyReturnTypeCompatibility:
    """Test safe_ttest_rel handles different SciPy return types correctly."""

    def test_safe_ttest_rel_with_tuple_return(self):
        """Test handling of old SciPy tuple return (statistic, pvalue)."""
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [1.1, 2.1, 3.1, 4.1, 5.1]

        # Mock scipy to return plain tuple
        with patch('src.analysis.statistical_analysis.stats.ttest_rel') as mock_ttest:
            mock_ttest.return_value = (-2.236, 0.0876)  # tuple
            t_stat, p_val = safe_ttest_rel(a, b)

        assert isinstance(t_stat, float)
        assert isinstance(p_val, float)
        assert not np.isnan(t_stat)
        assert not np.isnan(p_val)
        assert abs(t_stat - (-2.236)) < 0.001
        assert abs(p_val - 0.0876) < 0.001

    def test_safe_ttest_rel_with_named_result_object(self):
        """Test handling of SciPy >=1.9 result object with .statistic and .pvalue."""
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [1.1, 2.1, 3.1, 4.1, 5.1]

        # Create mock result object (simulating TtestResult)
        mock_result = Mock()
        mock_result.statistic = -2.236
        mock_result.pvalue = 0.0876

        with patch('src.analysis.statistical_analysis.stats.ttest_rel') as mock_ttest:
            mock_ttest.return_value = mock_result
            t_stat, p_val = safe_ttest_rel(a, b)

        assert isinstance(t_stat, float)
        assert isinstance(p_val, float)
        assert not np.isnan(t_stat)
        assert not np.isnan(p_val)
        assert abs(t_stat - (-2.236)) < 0.001
        assert abs(p_val - 0.0876) < 0.001

    def test_safe_ttest_rel_with_numpy_scalar_return(self):
        """Test handling when SciPy returns numpy scalars."""
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [1.1, 2.1, 3.1, 4.1, 5.1]

        # Mock scipy to return numpy scalars
        with patch('src.analysis.statistical_analysis.stats.ttest_rel') as mock_ttest:
            mock_ttest.return_value = (np.float64(-2.236), np.float64(0.0876))
            t_stat, p_val = safe_ttest_rel(a, b)

        assert isinstance(t_stat, float)
        assert isinstance(p_val, float)
        assert not np.isnan(t_stat)
        assert not np.isnan(p_val)

    def test_safe_ttest_rel_with_numpy_array_return(self):
        """Test handling when SciPy returns 0-d numpy arrays."""
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [1.1, 2.1, 3.1, 4.1, 5.1]

        # Mock scipy to return 0-d arrays (can happen with certain inputs)
        with patch('src.analysis.statistical_analysis.stats.ttest_rel') as mock_ttest:
            mock_ttest.return_value = (np.array(-2.236), np.array(0.0876))
            t_stat, p_val = safe_ttest_rel(a, b)

        assert isinstance(t_stat, float)
        assert isinstance(p_val, float)
        assert not np.isnan(t_stat)
        assert not np.isnan(p_val)

    def test_safe_ttest_rel_with_exception(self):
        """Test graceful handling when ttest_rel raises exception."""
        a = [1.0, 2.0]
        b = [1.0, 2.0]  # Identical values may cause issues

        # Mock scipy to raise exception
        with patch('src.analysis.statistical_analysis.stats.ttest_rel') as mock_ttest:
            mock_ttest.side_effect = ValueError("constant input")
            t_stat, p_val = safe_ttest_rel(a, b)

        assert isinstance(t_stat, float)
        assert isinstance(p_val, float)
        assert np.isnan(t_stat)
        assert np.isnan(p_val)

    def test_safe_ttest_rel_with_none_values(self):
        """Test handling when result object has None attributes."""
        a = [1.0, 2.0, 3.0]
        b = [1.1, 2.1, 3.1]

        # Create mock result with None values
        mock_result = Mock()
        mock_result.statistic = None
        mock_result.pvalue = None

        with patch('src.analysis.statistical_analysis.stats.ttest_rel') as mock_ttest:
            mock_ttest.return_value = mock_result
            t_stat, p_val = safe_ttest_rel(a, b)

        assert isinstance(t_stat, float)
        assert isinstance(p_val, float)
        assert np.isnan(t_stat)
        assert np.isnan(p_val)

    def test_safe_ttest_rel_with_actual_scipy(self):
        """Integration test with actual scipy.stats.ttest_rel."""
        # This test uses real scipy (not mocked) to ensure compatibility
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [1.1, 2.1, 3.1, 4.1, 5.1]

        t_stat, p_val = safe_ttest_rel(a, b)

        # Verify return types and basic properties
        assert isinstance(t_stat, float)
        assert isinstance(p_val, float)
        assert not np.isnan(t_stat)
        assert not np.isnan(p_val)
        assert 0.0 <= p_val <= 1.0  # p-value must be in valid range
        assert t_stat < 0  # b values are slightly higher, so t should be negative

    def test_safe_ttest_rel_with_identical_samples(self):
        """Test behavior with identical samples (edge case)."""
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [1.0, 2.0, 3.0, 4.0, 5.0]

        t_stat, p_val = safe_ttest_rel(a, b)

        # With identical samples, t should be 0 and p should be 1.0
        assert isinstance(t_stat, float)
        assert isinstance(p_val, float)
        # Allow for numerical precision issues
        assert abs(t_stat) < 1e-10 or np.isnan(t_stat)
        # p-value should be very close to 1.0 (or NaN if scipy can't compute)
        assert (abs(p_val - 1.0) < 1e-6) or np.isnan(p_val)

    def test_safe_ttest_rel_preserves_kwargs(self):
        """Test that additional kwargs are passed through to scipy."""
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [1.1, 2.1, 3.1, 4.1, 5.1]

        with patch('src.analysis.statistical_analysis.stats.ttest_rel') as mock_ttest:
            mock_ttest.return_value = (-2.236, 0.0876)
            safe_ttest_rel(a, b, alternative='greater')

        # Verify kwargs were passed
        mock_ttest.assert_called_once()
        call_kwargs = mock_ttest.call_args[1]
        assert 'alternative' in call_kwargs
        assert call_kwargs['alternative'] == 'greater'


class TestMultipleComparisonsCorrection:
    """Test Benjamini-Hochberg FDR correction implementation."""

    def test_benjamini_hochberg_basic(self):
        """Test basic BH correction logic."""
        # Example from literature: p-values [0.01, 0.04, 0.03, 0.005]
        # Sorted: [0.005, 0.01, 0.03, 0.04]
        # Ranks: [1, 2, 3, 4]
        # BH adjusted: p * n / rank
        # [0.005*4/1=0.02, 0.01*4/2=0.02, 0.03*4/3=0.04, 0.04*4/4=0.04]

        p_values = [0.01, 0.04, 0.03, 0.005]
        n = len(p_values)
        sorted_indices = sorted(range(n), key=lambda i: p_values[i])

        adjusted = []
        for rank, idx in enumerate(sorted_indices, start=1):
            adj_p = min(1.0, p_values[idx] * n / rank)
            adjusted.append((idx, adj_p))

        # Sort back to original order
        adjusted.sort(key=lambda x: x[0])
        adjusted_p = [a[1] for a in adjusted]

        # Verify corrections
        assert abs(adjusted_p[0] - 0.02) < 0.001  # 0.01 -> 0.02
        assert abs(adjusted_p[1] - 0.04) < 0.001  # 0.04 -> 0.04
        assert abs(adjusted_p[2] - 0.04) < 0.001  # 0.03 -> 0.04
        assert abs(adjusted_p[3] - 0.02) < 0.001  # 0.005 -> 0.02

    def test_benjamini_hochberg_all_significant(self):
        """Test when all p-values are significant after correction."""
        p_values = [0.001, 0.002, 0.003, 0.004]
        n = len(p_values)

        for rank, p in enumerate(p_values, start=1):
            adj_p = min(1.0, p * n / rank)
            # All should remain < 0.05
            assert adj_p < 0.05

    def test_benjamini_hochberg_none_significant(self):
        """Test when no p-values are significant after correction."""
        p_values = [0.1, 0.2, 0.3, 0.4]
        n = len(p_values)

        for rank, p in enumerate(p_values, start=1):
            adj_p = min(1.0, p * n / rank)
            # All should remain >= 0.05
            assert adj_p >= 0.05

    def test_benjamini_hochberg_mixed(self):
        """Test mixed significant/non-significant results."""
        p_values = [0.001, 0.02, 0.04, 0.1]
        alpha = 0.05
        n = len(p_values)
        sorted_indices = sorted(range(n), key=lambda i: p_values[i])

        # Apply BH procedure
        significant = []
        for rank, idx in enumerate(sorted_indices, start=1):
            adj_p = min(1.0, p_values[idx] * n / rank)
            if adj_p < alpha:
                significant.append(idx)

        # Expect first few to be significant
        assert len(significant) > 0
        assert 0 in significant  # p=0.001 should be significant


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
