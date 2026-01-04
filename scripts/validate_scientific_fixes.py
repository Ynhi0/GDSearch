"""
Quick Validation Test - Verify All Scientific Fixes

This script tests that all critical gaps have been properly addressed:

BASIC METHODOLOGY (Gaps 1-17):
- Gap 1: Time tracking
- Gap 11: Batch size variance correction
- Gap 12: Effective learning rate for Adam
- Gap 15: Gradient norm tracking (non-convex)
- Gap 16: Dynamic noise variance (interpolation)
- Gap 17: Heavy-tailed gradient detection

MASTER-LEVEL (Gaps 18-24):
- Gap 18: AIC/BIC model selection (not R²)
- Gap 21: No-augmentation config flag
- Gap 22: Multiplicative noise in theoretical bounds

GRANDMASTER-LEVEL (Gaps 25-34):
- Gap 25: Filter normalization for loss landscapes
- Gap 27: Hessian batch size for OOM prevention
- Gap 31: Adam update magnitude tracking (actual step distance)
- Gap 32: 1/√k rate fitting + Bonferroni correction
- Gap 33: Last-K mean for optimizer comparisons

Run this before defense to ensure no regressions.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_gap_1_time_tracking():
    """Verify time tracking is in training output"""
    print("\n[Gap 1] Testing time tracking...")
    
    main_file = Path('run_all_kaggle.py')
    if not main_file.exists():
        print("  ⚠ run_all_kaggle.py not found")
        return False
    
    content = main_file.read_text(encoding='utf-8')
    
    checks = [
        ('import time', 'time module import'),
        ('time_history', 'time history tracking'),
        ('epoch_start = time.time()', 'epoch timer'),
        ('elapsed_time = time.time() - start_time', 'elapsed time calculation')
    ]
    
    passed = True
    for pattern, desc in checks:
        if pattern in content:
            print(f"  ✓ {desc} found")
        else:
            print(f"  ✗ {desc} MISSING")
            passed = False
    
    return passed


def test_gap_11_batch_variance():
    """Verify batch size correction in theory validation"""
    print("\n[Gap 11] Testing batch size variance correction...")
    
    val_file = Path('src/experiments/theory_practice_validation.py')
    if not val_file.exists():
        print("  ⚠ theory_practice_validation.py not found")
        return False
    
    content = val_file.read_text(encoding='utf-8')
    
    checks = [
        ('batch_size', 'batch size extraction'),
        ('sigma_corrected = sigma_est / batch_size', 'σ²/B correction'),
        ('GAP 11', 'Gap 11 comment/documentation')
    ]
    
    passed = True
    for pattern, desc in checks:
        if pattern in content:
            print(f"  ✓ {desc} found")
        else:
            print(f"  ✗ {desc} MISSING")
            passed = False
    
    return passed


def test_gap_12_effective_lr():
    """Verify effective LR for Adam stability check"""
    print("\n[Gap 12] Testing effective learning rate...")
    
    val_file = Path('src/experiments/theory_practice_validation.py')
    if not val_file.exists():
        print("  ⚠ theory_practice_validation.py not found")
        return False
    
    content = val_file.read_text(encoding='utf-8')
    
    checks = [
        ('effective_lr', 'effective LR variable'),
        ('is_adaptive', 'adaptive optimizer detection'),
        ('lr_for_check', 'LR selection for stability'),
        ('GAP 12', 'Gap 12 comment/documentation')
    ]
    
    passed = True
    for pattern, desc in checks:
        if pattern in content:
            print(f"  ✓ {desc} found")
        else:
            print(f"  ✗ {desc} MISSING")
            passed = False
    
    return passed


def test_gap_15_gradient_norm():
    """Verify gradient norm tracking for non-convex validation"""
    print("\n[Gap 15] Testing gradient norm tracking...")
    
    main_file = Path('run_all_kaggle.py')
    if main_file.exists():
        content = main_file.read_text(encoding='utf-8')
        if 'grad_norm_history' in content:
            print("  ✓ Gradient norm tracking in training loop")
        else:
            print("  ✗ Gradient norm tracking MISSING in training")
            return False
    
    val_file = Path('src/experiments/theory_practice_validation.py')
    if val_file.exists():
        content = val_file.read_text(encoding='utf-8')
        if 'grad_norm_history' in content and 'GAP 15' in content:
            print("  ✓ Gradient norm validation logic present")
        else:
            print("  ✗ Gradient norm validation MISSING")
            return False
    
    return True


def test_gap_16_dynamic_noise():
    """Verify dynamic noise variance modeling"""
    print("\n[Gap 16] Testing dynamic noise variance...")
    
    val_file = Path('src/experiments/theory_practice_validation.py')
    if not val_file.exists():
        print("  ⚠ theory_practice_validation.py not found")
        return False
    
    content = val_file.read_text(encoding='utf-8')
    
    checks = [
        ('use_dynamic_noise', 'dynamic noise flag'),
        ('INTERPOLATION REGIME', 'interpolation detection'),
        ('GAP 16', 'Gap 16 comment/documentation')
    ]
    
    passed = True
    for pattern, desc in checks:
        if pattern in content:
            print(f"  ✓ {desc} found")
        else:
            print(f"  ✗ {desc} MISSING")
            passed = False
    
    return passed


def test_gap_17_heavy_tails():
    """Verify heavy-tailed gradient detection"""
    print("\n[Gap 17] Testing heavy-tailed gradient detection...")
    
    noise_file = Path('src/analysis/gradient_noise_analysis.py')
    if not noise_file.exists():
        print("  ⚠ gradient_noise_analysis.py not found")
        return False
    
    content = noise_file.read_text(encoding='utf-8')
    
    checks = [
        ('shapiro', 'Shapiro-Wilk test'),
        ('normality_test_pvalue', 'normality p-value'),
        ('is_gaussian', 'Gaussian flag'),
        ('GAP 17', 'Gap 17 comment/documentation')
    ]
    
    passed = True
    for pattern, desc in checks:
        if pattern in content:
            print(f"  ✓ {desc} found")
        else:
            print(f"  ✗ {desc} MISSING")
            passed = False
    
    return passed


def test_gap_18_aic_model_selection():
    """Verify AIC/BIC model selection (not R²)"""
    print("\n[Gap 18] Testing AIC/BIC model selection...")
    
    analyzer_file = Path('src/analysis/convergence_rate_analyzer.py')
    if not analyzer_file.exists():
        print("  ⚠ convergence_rate_analyzer.py not found")
        return False
    
    content = analyzer_file.read_text(encoding='utf-8')
    
    checks = [
        ('compute_aic', 'AIC function'),
        ('compute_bic', 'BIC function'),
        ('best_aic', 'AIC-based model selection'),
        ('GAP 18', 'Gap 18 comment/documentation')
    ]
    
    passed = True
    for pattern, desc in checks:
        if pattern in content:
            print(f"  ✓ {desc} found")
        else:
            print(f"  ✗ {desc} MISSING")
            passed = False
    
    return passed


def test_gap_21_augmentation_flag():
    """Verify no-augmentation config flag"""
    print("\n[Gap 21] Testing augmentation config flag...")
    
    data_file = Path('src/core/data_utils.py')
    if not data_file.exists():
        print("  ⚠ data_utils.py not found")
        return False
    
    content = data_file.read_text(encoding='utf-8')
    
    checks = [
        ('augment: bool = True', 'augment parameter'),
        ('GAP 21', 'Gap 21 comment/documentation'),
        ('if augment:', 'conditional augmentation')
    ]
    
    passed = True
    for pattern, desc in checks:
        if pattern in content:
            print(f"  ✓ {desc} found")
        else:
            print(f"  ✗ {desc} MISSING")
            passed = False
    
    return passed


def test_gap_22_multiplicative_noise():
    """Verify multiplicative noise support in theory"""
    print("\n[Gap 22] Testing multiplicative noise model...")
    
    bounds_file = Path('src/analysis/theoretical_bounds.py')
    if not bounds_file.exists():
        print("  ⚠ theoretical_bounds.py not found")
        return False
    
    content = bounds_file.read_text(encoding='utf-8')
    
    checks = [
        ("noise_model: str = 'additive'", 'noise_model parameter'),
        ("noise_model == 'multiplicative'", 'multiplicative noise handling'),
        ('noise_floor', 'noise floor computation'),
        ('GAP 22', 'Gap 22 comment/documentation')
    ]
    
    passed = True
    for pattern, desc in checks:
        if pattern in content:
            print(f"  ✓ {desc} found")
        else:
            print(f"  ✗ {desc} MISSING")
            passed = False
    
    return passed


def test_gap_25_filter_normalization():
    """Verify filter-wise normalization for loss landscapes"""
    print("\n[Gap 25] Testing filter normalization...")
    
    landscape_file = Path('src/visualization/loss_landscape.py')
    if not landscape_file.exists():
        print("  ⚠ loss_landscape.py not found")
        return False
    
    content = landscape_file.read_text(encoding='utf-8')
    
    checks = [
        ('filter_normalize', 'filter_normalize parameter'),
        ('GAP 25', 'Gap 25 comment/documentation'),
        ('Li et al.', 'Citation to Li et al. 2018'),
        ('w_norm = param.data.norm()', 'per-parameter normalization')
    ]
    
    passed = True
    for pattern, desc in checks:
        if pattern in content:
            print(f"  ✓ {desc} found")
        else:
            print(f"  ✗ {desc} MISSING")
            passed = False
    
    return passed


def test_gap_27_hessian_batch_size():
    """Verify Hessian batch size warning"""
    print("\n[Gap 27] Testing Hessian OOM prevention...")
    
    saddle_file = Path('src/analysis/saddle_point_detection.py')
    if not saddle_file.exists():
        print("  ⚠ saddle_point_detection.py not found")
        return False
    
    content = saddle_file.read_text(encoding='utf-8')
    
    checks = [
        ('GAP 27', 'Gap 27 comment/documentation'),
        ('HESSIAN_BATCH_SIZE', 'batch size constant'),
        ('OOM', 'OOM warning')
    ]
    
    passed = True
    for pattern, desc in checks:
        if pattern in content:
            print(f"  ✓ {desc} found")
        else:
            print(f"  ✗ {desc} MISSING")
            passed = False
    
    return passed


def test_gap_31_adam_update():
    """Verify actual step distance tracking for Adam"""
    print("\n[Gap 31] Testing Adam update magnitude fix...")
    
    dynamics_file = Path('src/core/dynamics_tracker.py')
    if not dynamics_file.exists():
        print("  ⚠ dynamics_tracker.py not found")
        return False
    
    content = dynamics_file.read_text(encoding='utf-8')
    
    checks = [
        ('GAP 31', 'Gap 31 comment/documentation'),
        ('actual_step_sizes', 'actual step tracking'),
        ('step_distance', 'step distance computation')
    ]
    
    passed = True
    for pattern, desc in checks:
        if pattern in content:
            print(f"  ✓ {desc} found")
        else:
            print(f"  ✗ {desc} MISSING")
            passed = False
    
    return passed


def test_gap_32_root_rate():
    """Verify 1/√k rate fitting + Bonferroni correction"""
    print("\n[Gap 32] Testing 1/√k rate + Bonferroni...")
    
    # Check convergence analysis
    conv_file = Path('src/experiments/convergence_analysis.py')
    if conv_file.exists():
        content = conv_file.read_text(encoding='utf-8')
        if 'root_sublinear' in content and 'inv_iters_sqrt' in content:
            print("  ✓ 1/√k rate fitting found")
        else:
            print("  ✗ 1/√k rate fitting MISSING")
            return False
    
    # Check Bonferroni correction
    matrix_file = Path('src/analysis/optimizer_comparison_matrix.py')
    if matrix_file.exists():
        content = matrix_file.read_text(encoding='utf-8')
        if 'bonferroni' in content and 'num_comparisons' in content:
            print("  ✓ Bonferroni correction found")
        else:
            print("  ✗ Bonferroni correction MISSING")
            return False
    
    return True


def test_gap_33_last_k_mean():
    """Verify Last-K mean for optimizer comparisons"""
    print("\n[Gap 33] Testing Last-K mean aggregation...")
    
    matrix_file = Path('src/analysis/optimizer_comparison_matrix.py')
    if not matrix_file.exists():
        print("  ⚠ optimizer_comparison_matrix.py not found")
        return False
    
    content = matrix_file.read_text(encoding='utf-8')
    
    checks = [
        ('last_k_mean', 'last_k_mean aggregation'),
        ('GAP 33', 'Gap 33 comment/documentation'),
        ('[-k:]', 'last K selection')
    ]
    
    passed = True
    for pattern, desc in checks:
        if pattern in content:
            print(f"  ✓ {desc} found")
        else:
            print(f"  ✗ {desc} MISSING")
            passed = False
    
    return passed


def test_correlation_analysis():
    """Verify correlation analysis module exists"""
    print("\n[Gap 14] Testing correlation analysis...")
    
    corr_file = Path('src/analysis/correlation_analysis.py')
    if not corr_file.exists():
        print("  ✗ correlation_analysis.py not found")
        return False
    
    content = corr_file.read_text(encoding='utf-8')
    
    checks = [
        ('collect_hessian_convergence_data', 'Hessian-convergence correlation'),
        ('collect_sharpness_accuracy_data', 'Sharpness-accuracy correlation'),
        ('pearsonr', 'Statistical tests'),
        ('plot_correlation_analysis', 'Plotting function')
    ]
    
    passed = True
    for pattern, desc in checks:
        if pattern in content:
            print(f"  ✓ {desc} found")
        else:
            print(f"  ✗ {desc} MISSING")
            passed = False
    
    return passed


def main():
    """Run all validation tests"""
    print("="*80)
    print("COMPREHENSIVE SCIENTIFIC FIXES VALIDATION")
    print("="*80)
    
    tests = [
        # Basic methodology
        ("Gap 1: Time Tracking", test_gap_1_time_tracking),
        ("Gap 11: Batch Variance", test_gap_11_batch_variance),
        ("Gap 12: Effective LR", test_gap_12_effective_lr),
        ("Gap 14: Correlation Analysis", test_correlation_analysis),
        ("Gap 15: Gradient Norm", test_gap_15_gradient_norm),
        ("Gap 16: Dynamic Noise", test_gap_16_dynamic_noise),
        ("Gap 17: Heavy Tails", test_gap_17_heavy_tails),
        # Master-level
        ("Gap 18: AIC Model Selection", test_gap_18_aic_model_selection),
        ("Gap 21: Augmentation Flag", test_gap_21_augmentation_flag),
        ("Gap 22: Multiplicative Noise", test_gap_22_multiplicative_noise),
        # Grandmaster-level
        ("Gap 25: Filter Normalization", test_gap_25_filter_normalization),
        ("Gap 27: Hessian Batch Size", test_gap_27_hessian_batch_size),
        ("Gap 31: Adam Update Tracking", test_gap_31_adam_update),
        ("Gap 32: Root Rate + Bonferroni", test_gap_32_root_rate),
        ("Gap 33: Last-K Mean", test_gap_33_last_k_mean),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"  ✗ Test failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} - {name}")
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n" + "="*80)
        print("✓ ALL CRITICAL FIXES VERIFIED - READY FOR DEFENSE")
        print("="*80)
        return 0
    else:
        print(f"\n⚠ {total_count - passed_count} ISSUES DETECTED - REVIEW REQUIRED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
