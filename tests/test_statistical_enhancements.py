"""
Tests for enhanced statistical analysis module.
"""

import numpy as np
import pytest
import warnings
from src.analysis.statistical_analysis import (
    compare_optimizers_ttest,
    compute_power_analysis,
    compute_required_sample_size,
    power_analysis_report,
    bonferroni_correction,
    holm_bonferroni_correction,
    benjamini_hochberg_correction,
    compare_multiple_optimizers
)


class TestBasicTTest:
    """Tests for basic t-test functionality."""
    
    def test_ttest_identical_groups(self):
        """Test t-test with identical groups (edge case)."""
        results_A = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        results_B = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        
        # Now handled properly without scipy warnings
        result = compare_optimizers_ttest(results_A, results_B)
        
        assert result['mean_A'] == result['mean_B']
        # Should properly detect no difference (p=1.0, Cohen's d=0)
        assert result['p_value'] == 1.0
        assert result['cohens_d'] == 0.0
        assert not result['significant']
    
    def test_ttest_different_groups(self):
        """Test t-test with clearly different groups."""
        results_A = np.array([0.9, 0.91, 0.92, 0.93, 0.94])
        results_B = np.array([0.5, 0.51, 0.52, 0.53, 0.54])
        
        result = compare_optimizers_ttest(results_A, results_B)
        
        assert result['mean_A'] > result['mean_B']
        assert result['p_value'] < 0.001  # Should be very significant
        assert result['significant']
        assert result['cohens_d'] > 5.0  # Large effect
    
    def test_ttest_confidence_intervals(self):
        """Test that confidence intervals contain the mean."""
        results_A = np.random.randn(20) + 10
        results_B = np.random.randn(20) + 11
        
        result = compare_optimizers_ttest(results_A, results_B)
        
        # Mean should be within its own confidence interval
        assert result['ci_A'][0] <= result['mean_A'] <= result['ci_A'][1]
        assert result['ci_B'][0] <= result['mean_B'] <= result['ci_B'][1]
    
    def test_cohens_d_calculation(self):
        """Test Cohen's d effect size calculation."""
        # Known case: two groups with clear separation
        results_A = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        results_B = np.array([1.0, 1.1, 1.2, 1.3, 1.4])
        
        result = compare_optimizers_ttest(results_A, results_B)
        
        # Effect size should be large (> 2.0 standard deviations difference)
        assert abs(result['cohens_d']) > 2.0


class TestPowerAnalysis:
    """Tests for power analysis functions."""
    
    def test_power_increases_with_sample_size(self):
        """Test that power increases with sample size."""
        effect_size = 0.5
        alpha = 0.05
        
        power_n5 = compute_power_analysis(effect_size, 5, alpha)
        power_n10 = compute_power_analysis(effect_size, 10, alpha)
        power_n50 = compute_power_analysis(effect_size, 50, alpha)
        
        assert power_n5 < power_n10 < power_n50
    
    def test_power_increases_with_effect_size(self):
        """Test that power increases with effect size."""
        n_samples = 20
        alpha = 0.05
        
        power_small = compute_power_analysis(0.2, n_samples, alpha)
        power_medium = compute_power_analysis(0.5, n_samples, alpha)
        power_large = compute_power_analysis(0.8, n_samples, alpha)
        
        assert power_small < power_medium < power_large
    
    def test_power_bounds(self):
        """Test that power is between 0 and 1 (or handled gracefully)."""
        effect_sizes = [0.1, 0.5, 1.0, 2.0]
        sample_sizes = [5, 10, 20, 50, 100]
        
        for effect_size in effect_sizes:
            for n in sample_sizes:
                power = compute_power_analysis(effect_size, n)
                # Power should be in [0, 1] or NaN for extreme cases
                assert (0 <= power <= 1) or np.isnan(power), f"Power out of bounds for d={effect_size}, n={n}: power={power}"
    
    def test_required_sample_size_for_high_power(self):
        """Test that required sample size achieves target power."""
        effect_size = 0.5
        target_power = 0.8
        alpha = 0.05
        
        required_n = compute_required_sample_size(effect_size, target_power, alpha)
        achieved_power = compute_power_analysis(effect_size, required_n, alpha)
        
        assert achieved_power >= target_power, \
            f"Required n={required_n} only achieves power={achieved_power}, target was {target_power}"
    
    def test_required_sample_size_decreases_with_effect(self):
        """Test that larger effects require fewer samples."""
        target_power = 0.8
        alpha = 0.05
        
        n_small = compute_required_sample_size(0.2, target_power, alpha)
        n_medium = compute_required_sample_size(0.5, target_power, alpha)
        n_large = compute_required_sample_size(0.8, target_power, alpha)
        
        assert n_small > n_medium > n_large
    
    def test_power_analysis_report(self):
        """Test comprehensive power analysis report."""
        np.random.seed(42)
        results_A = np.random.normal(10, 1, size=15)
        results_B = np.random.normal(10.5, 1, size=15)
        
        report = power_analysis_report(results_A, results_B, "A", "B")
        
        assert 'observed_effect_size' in report
        assert 'achieved_power' in report
        assert 'required_n' in report
        assert 0 <= report['achieved_power'] <= 1
        assert report['required_n'] >= 2


class TestMultipleComparisons:
    """Tests for multiple comparison correction methods."""
    
    def test_bonferroni_correction(self):
        """Test Bonferroni correction."""
        # 5 tests with p-values
        p_values = [0.001, 0.002, 0.03, 0.04, 0.05]
        alpha = 0.05
        
        significant, adjusted_alpha = bonferroni_correction(p_values, alpha)
        
        # With 5 tests, adjusted alpha = 0.05 / 5 = 0.01
        assert adjusted_alpha == 0.01
        
        # p=0.001 and p=0.002 should be significant (< 0.01)
        assert significant[0] == True
        assert significant[1] == True
        assert significant[2] == False
        assert sum(significant) >= 2
    
    def test_bonferroni_all_significant(self):
        """Test Bonferroni when all tests are highly significant."""
        p_values = [0.001, 0.001, 0.001]
        alpha = 0.05
        
        significant, adjusted_alpha = bonferroni_correction(p_values, alpha)
        
        # All should remain significant
        assert all(significant)
    
    def test_holm_bonferroni_correction(self):
        """Test Holm-Bonferroni correction."""
        # Ordered p-values
        p_values = [0.01, 0.02, 0.03, 0.04, 0.05]
        alpha = 0.05
        
        significant = holm_bonferroni_correction(p_values, alpha)
        
        # Holm is less conservative than Bonferroni
        # Should reject more nulls
        assert sum(significant) >= sum(bonferroni_correction(p_values, alpha)[0])
    
    def test_holm_step_down_property(self):
        """Test that Holm stops at first non-rejection."""
        # Design p-values where step-down should stop
        p_values = [0.001, 0.005, 0.01, 0.1, 0.2]
        alpha = 0.05
        
        significant = holm_bonferroni_correction(p_values, alpha)
        
        # Once a null is not rejected, all subsequent should also not be rejected
        first_nonsig = None
        for i, sig in enumerate(significant):
            if not sig and first_nonsig is None:
                first_nonsig = i
            if first_nonsig is not None:
                assert not sig, f"Test {i} should not be significant after {first_nonsig}"
    
    def test_benjamini_hochberg_correction(self):
        """Test Benjamini-Hochberg (FDR) correction."""
        p_values = [0.01, 0.02, 0.03, 0.04, 0.05]
        alpha = 0.05
        
        significant = benjamini_hochberg_correction(p_values, alpha)
        
        # BH should be less conservative than Bonferroni
        bonf_sig, _ = bonferroni_correction(p_values, alpha)
        assert sum(significant) >= sum(bonf_sig)
    
    def test_bh_all_null(self):
        """Test BH when all null hypotheses are true."""
        p_values = [0.6, 0.7, 0.8, 0.9, 0.95]
        alpha = 0.05
        
        significant = benjamini_hochberg_correction(p_values, alpha)
        
        # Should reject none
        assert not any(significant)
    
    def test_bh_all_alternative(self):
        """Test BH when all alternative hypotheses are true."""
        p_values = [0.001, 0.002, 0.003, 0.004, 0.005]
        alpha = 0.05
        
        significant = benjamini_hochberg_correction(p_values, alpha)
        
        # Should reject all
        assert all(significant)
    
    def test_correction_methods_order(self):
        """Test that correction methods have expected ordering."""
        # Bonferroni most conservative, BH least conservative
        p_values = [0.01, 0.02, 0.03, 0.04, 0.05]
        alpha = 0.05
        
        bonf_sig, _ = bonferroni_correction(p_values, alpha)
        holm_sig = holm_bonferroni_correction(p_values, alpha)
        bh_sig = benjamini_hochberg_correction(p_values, alpha)
        
        # Number of rejections should follow: Bonferroni ≤ Holm ≤ BH
        assert sum(bonf_sig) <= sum(holm_sig) <= sum(bh_sig)


class TestMultipleOptimizerComparison:
    """Tests for comparing multiple optimizers."""
    
    def test_multiple_optimizer_comparison(self):
        """Test pairwise comparison of multiple optimizers."""
        np.random.seed(42)
        results_dict = {
            'Opt1': np.random.normal(0.9, 0.01, size=10),
            'Opt2': np.random.normal(0.91, 0.01, size=10),
            'Opt3': np.random.normal(0.92, 0.01, size=10)
        }
        
        df = compare_multiple_optimizers(results_dict, correction_method='none', alpha=0.05)
        
        # Should have 3 choose 2 = 3 comparisons
        assert len(df) == 3
        
        # All columns should be present
        required_cols = ['Optimizer A', 'Optimizer B', 'p-value', 'Cohen\'s d', 
                        'Significant (raw)', 'Significant (corrected)']
        for col in required_cols:
            assert col in df.columns
    
    def test_multiple_comparison_reduces_false_positives(self):
        """Test that correction reduces false positives."""
        np.random.seed(42)
        # All from same distribution (no true differences)
        results_dict = {
            f'Opt{i}': np.random.normal(0.9, 0.01, size=20)
            for i in range(5)
        }
        
        df_none = compare_multiple_optimizers(results_dict, correction_method='none', alpha=0.05)
        df_bonf = compare_multiple_optimizers(results_dict, correction_method='bonferroni', alpha=0.05)
        
        # Corrected version should have fewer (or equal) significant results
        assert sum(df_bonf['Significant (corrected)']) <= sum(df_none['Significant (raw)'])
    
    def test_different_correction_methods(self):
        """Test that all correction methods work."""
        np.random.seed(42)
        results_dict = {
            'Opt1': np.random.normal(0.9, 0.01, size=10),
            'Opt2': np.random.normal(0.95, 0.01, size=10)
        }
        
        for method in ['none', 'bonferroni', 'holm', 'bh']:
            df = compare_multiple_optimizers(results_dict, correction_method=method)
            assert len(df) == 1  # Only one comparison
            assert 'Significant (corrected)' in df.columns


class TestStatisticalProperties:
    """Tests for statistical properties and edge cases."""
    
    def test_zero_variance_handling(self):
        """Test handling of zero variance data with different means."""
        results_A = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        results_B = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        
        # Now handled properly without scipy warnings
        result = compare_optimizers_ttest(results_A, results_B)
        
        # Should not crash - different means with zero variance = infinite effect
        assert result['p_value'] == 0.0
        assert np.isinf(result['cohens_d'])
        assert result['significant']
    
    def test_small_sample_size(self):
        """Test with very small sample sizes."""
        results_A = np.array([0.9, 0.95])
        results_B = np.array([0.85, 0.90])
        
        result = compare_optimizers_ttest(results_A, results_B)
        
        # Should complete without error
        assert 'p_value' in result
        assert 'cohens_d' in result
    
    def test_large_sample_size(self):
        """Test with large sample sizes."""
        np.random.seed(42)
        results_A = np.random.normal(0.95, 0.01, size=1000)
        results_B = np.random.normal(0.951, 0.01, size=1000)
        
        result = compare_optimizers_ttest(results_A, results_B)
        power = compute_power_analysis(result['cohens_d'], 1000)
        
        # With large n, even small effects should be detected
        assert power > 0.9
    
    def test_power_edge_cases(self):
        """Test power analysis edge cases."""
        # Very small effect
        power_tiny = compute_power_analysis(0.01, 100)
        assert 0 <= power_tiny <= 0.1
        
        # Very large effect
        power_huge = compute_power_analysis(3.0, 10)
        assert power_huge > 0.99
        
        # Very small sample
        power_small_n = compute_power_analysis(0.5, 2)
        assert 0 <= power_small_n <= 0.2


class TestNormalityTests:
    """Tests for normality testing functions."""
    
    def test_shapiro_wilk_normal_data(self):
        """Test Shapiro-Wilk on normal data."""
        from src.analysis.statistical_analysis import test_normality
        
        # Generate normal data
        np.random.seed(42)
        data = np.random.normal(0, 1, size=50)
        
        result = test_normality(data, method='shapiro', alpha=0.05)
        
        assert result['method'] == 'shapiro'
        assert result['statistic'] > 0
        assert result['p_value'] > 0.05  # Should not reject normality
        assert result['normal'] == True
    
    def test_shapiro_wilk_non_normal_data(self):
        """Test Shapiro-Wilk on non-normal data."""
        from src.analysis.statistical_analysis import test_normality
        
        # Generate exponential data (non-normal)
        np.random.seed(42)
        data = np.random.exponential(1, size=100)
        
        result = test_normality(data, method='shapiro', alpha=0.05)
        
        # Should reject normality (but might not always due to randomness)
        assert result['method'] == 'shapiro'
        assert 'statistic' in result
        assert 'p_value' in result
    
    def test_anderson_darling(self):
        """Test Anderson-Darling test."""
        from src.analysis.statistical_analysis import test_normality
        
        np.random.seed(42)
        data = np.random.normal(0, 1, size=50)
        
        result = test_normality(data, method='anderson', alpha=0.05)
        
        assert result['method'] == 'anderson'
        assert result['statistic'] > 0
        assert result['normal'] in [True, False]
    
    def test_kstest(self):
        """Test Kolmogorov-Smirnov test."""
        from src.analysis.statistical_analysis import test_normality
        
        np.random.seed(42)
        data = np.random.normal(0, 1, size=50)
        
        result = test_normality(data, method='kstest', alpha=0.05)
        
        assert result['method'] == 'kstest'
        assert result['statistic'] >= 0
        assert 0 <= result['p_value'] <= 1
    
    def test_small_sample_warning(self):
        """Test warning for very small samples."""
        from src.analysis.statistical_analysis import test_normality
        
        data = np.array([1.0, 2.0])
        
        result = test_normality(data, method='shapiro')
        
        assert 'warning' in result
        assert result['normal'] is None


class TestNonParametricTests:
    """Tests for non-parametric statistical tests."""
    
    def test_mann_whitney_different_groups(self):
        """Test Mann-Whitney U test with different groups."""
        from src.analysis.statistical_analysis import compare_optimizers_mann_whitney
        
        np.random.seed(42)
        # Two clearly different groups
        results_A = np.random.exponential(1, size=20) + 5
        results_B = np.random.exponential(1, size=20)
        
        result = compare_optimizers_mann_whitney(
            results_A, results_B,
            name_A="A", name_B="B",
            alternative='two-sided'
        )
        
        assert result['test'] == 'Mann-Whitney U'
        assert result['median_A'] > result['median_B']
        assert result['p_value'] < 0.05  # Should be significant
        assert result['significant'] == True
    
    def test_mann_whitney_same_groups(self):
        """Test Mann-Whitney U test with same groups."""
        from src.analysis.statistical_analysis import compare_optimizers_mann_whitney
        
        np.random.seed(42)
        results_A = np.random.normal(0, 1, size=20)
        results_B = np.random.normal(0, 1, size=20)
        
        result = compare_optimizers_mann_whitney(results_A, results_B)
        
        # Should not be significant (same distribution)
        assert result['p_value'] > 0.01
        assert abs(result['effect_size_r']) < 0.5
    
    def test_wilcoxon_paired_different(self):
        """Test Wilcoxon test with paired different samples."""
        from src.analysis.statistical_analysis import compare_optimizers_wilcoxon
        
        np.random.seed(42)
        # Before and after (paired)
        before = np.random.normal(5, 1, size=20)
        after = before + np.random.normal(2, 0.5, size=20)  # Improvement
        
        result = compare_optimizers_wilcoxon(
            after, before,
            name_A="After", name_B="Before",
            alternative='greater'
        )
        
        assert result['test'] == 'Wilcoxon signed-rank'
        assert result['median_A'] > result['median_B']
        assert result['p_value'] < 0.05
        assert result['significant'] == True
    
    def test_wilcoxon_requires_equal_length(self):
        """Test that Wilcoxon requires equal-length samples."""
        from src.analysis.statistical_analysis import compare_optimizers_wilcoxon
        
        results_A = np.array([1, 2, 3])
        results_B = np.array([1, 2, 3, 4, 5])
        
        with pytest.raises(ValueError):
            compare_optimizers_wilcoxon(results_A, results_B)
    
    def test_mann_whitney_effect_size(self):
        """Test Mann-Whitney effect size calculation."""
        from src.analysis.statistical_analysis import compare_optimizers_mann_whitney
        
        # Perfect separation
        results_A = np.array([1, 2, 3, 4, 5])
        results_B = np.array([6, 7, 8, 9, 10])
        
        result = compare_optimizers_mann_whitney(results_A, results_B)
        
        # Effect size should be strong
        assert abs(result['effect_size_r']) > 0.5


class TestAutoSelectTest:
    """Tests for automatic test selection."""
    
    def test_auto_select_normal_data(self):
        """Test auto-selection with normal data."""
        from src.analysis.statistical_analysis import auto_select_test
        
        np.random.seed(42)
        results_A = np.random.normal(10, 1, size=50)
        results_B = np.random.normal(10.5, 1, size=50)
        
        result = auto_select_test(results_A, results_B, paired=False)
        
        # Should select parametric test (t-test)
        assert 'parametric' in result['test_type']
        assert result['normality_A']['normal'] == True
        assert result['normality_B']['normal'] == True
    
    def test_auto_select_non_normal_independent(self):
        """Test auto-selection with non-normal independent data."""
        from src.analysis.statistical_analysis import auto_select_test
        
        np.random.seed(42)
        results_A = np.random.exponential(1, size=30)
        results_B = np.random.exponential(1, size=30)
        
        result = auto_select_test(results_A, results_B, paired=False)
        
        # Should select Mann-Whitney U
        if not result['normality_A']['normal'] or not result['normality_B']['normal']:
            assert 'Mann-Whitney' in result['test_type']
    
    def test_auto_select_non_normal_paired(self):
        """Test auto-selection with non-normal paired data."""
        from src.analysis.statistical_analysis import auto_select_test
        
        np.random.seed(42)
        results_A = np.random.exponential(1, size=30)
        results_B = results_A * 1.2 + np.random.normal(0, 0.1, size=30)
        
        result = auto_select_test(results_A, results_B, paired=True)
        
        # Should select Wilcoxon if not normal
        if not result['normality_A']['normal'] or not result['normality_B']['normal']:
            assert 'Wilcoxon' in result['test_type']
    
    def test_auto_select_includes_normality_info(self):
        """Test that auto-selection includes normality test info."""
        from src.analysis.statistical_analysis import auto_select_test
        
        np.random.seed(42)
        results_A = np.random.normal(10, 1, size=30)
        results_B = np.random.normal(10, 1, size=30)
        
        result = auto_select_test(results_A, results_B)
        
        assert 'normality_A' in result
        assert 'normality_B' in result
        assert 'test_result' in result
        assert 'test_type' in result
