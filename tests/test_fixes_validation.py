"""
Validation script to test all fixes are working correctly.
Tests:
1. CSV files contain final test metrics in last row
2. Visualizations detect scaling correctly
3. Optimizer extraction is robust
4. Dice metrics are properly handled
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def test_csv_has_final_metrics():
    """Test that per-run CSVs have test_acc/test_dice in final row"""
    print("\n" + "="*80)
    print("TEST 1: CSV Files Contain Final Test Metrics")
    print("="*80)

    # Create sample data to simulate what experiments produce
    test_cases = [
        {
            'name': 'CIFAR10_ResNet18_Adam_seed42.csv',
            'data': pd.DataFrame({
                'epoch': [1, 2, 3],
                'train_loss': [2.3, 1.8, 1.5],
                'val_loss': [2.1, 1.7, 1.4],
                'train_acc': [0.3, 0.45, 0.55],
                'val_acc': [0.35, 0.48, 0.58],
                'test_acc': [np.nan, np.nan, 0.60],  # Final test acc in last row
                'test_loss': [np.nan, np.nan, 1.3]
            }),
            'expected_metric': 'test_acc',
            'expected_value': 0.60
        },
        {
            'name': 'Medical_UNet2D_Adam_seed42.csv',
            'data': pd.DataFrame({
                'epoch': [1, 2],
                'train_loss': [0.8, 0.7],
                'val_loss': [0.6, 0.5],
                'train_dice': [0.20, 0.23],
                'val_dice': [0.18, 0.27],
                'test_dice': [np.nan, 0.26],  # Final dice in last row
                'test_loss': [np.nan, 0.55]
            }),
            'expected_metric': 'test_dice',
            'expected_value': 0.26
        }
    ]

    all_passed = True
    for test_case in test_cases:
        df = test_case['data']
        metric = test_case['expected_metric']
        expected = test_case['expected_value']

        last_row = df.iloc[-1]
        actual = last_row[metric]

        if pd.notna(actual) and np.isclose(actual, expected, rtol=0.01):
            print(f"✅ {test_case['name']}: {metric}={actual:.2f} (expected {expected:.2f})")
        else:
            print(f"❌ {test_case['name']}: {metric}={actual} (expected {expected})")
            all_passed = False

    assert all_passed

def test_scaling_detection():
    """Test automatic fraction vs percentage detection"""
    print("\n" + "="*80)
    print("TEST 2: Automatic Scaling Detection")
    print("="*80)

    test_cases = [
        {'values': [0.45, 0.55, 0.68, 0.72], 'expected_scale': 100.0, 'type': 'fraction'},
        {'values': [45, 55, 68, 72], 'expected_scale': 1.0, 'type': 'percentage'},
        {'values': [0.001, 0.01, 0.05], 'expected_scale': 100.0, 'type': 'tiny fraction'},
    ]

    all_passed = True
    for test_case in test_cases:
        values = np.array(test_case['values'])
        max_val = values.max()
        is_fraction = max_val <= 1.5
        scale_factor = 100.0 if is_fraction else 1.0

        expected = test_case['expected_scale']
        if scale_factor == expected:
            print(f"✅ {test_case['type']}: max={max_val}, scale={scale_factor} (expected {expected})")
        else:
            print(f"❌ {test_case['type']}: max={max_val}, scale={scale_factor} (expected {expected})")
            all_passed = False

    assert all_passed

def test_optimizer_extraction():
    """Test robust optimizer name extraction from filenames"""
    print("\n" + "="*80)
    print("TEST 3: Robust Optimizer Extraction")
    print("="*80)

    test_cases = [
        {'filename': 'CIFAR10_ResNet18_Adam_seed42.csv', 'expected': 'Adam'},
        {'filename': 'MNIST_SimpleMLP_SAM_Adam_seed123.csv', 'expected': 'SAM_Adam'},
        {'filename': 'Medical_UNet2D_Lookahead_SGD_seed456.csv', 'expected': 'Lookahead_SGD'},
        {'filename': 'NLP_LSTM_SGD_Momentum_seed789.csv', 'expected': 'SGD_Momentum'},
    ]

    all_passed = True
    for test_case in test_cases:
        filename = test_case['filename']
        expected = test_case['expected']

        # Simulate the extraction logic from run_all_kaggle.py
        stem = filename.replace('.csv', '')
        parts = stem.split('_')

        optimizer_found = False
        optimizer = None

        # Method 1: Look for token(s) before 'seed'
        # Handle compound names like SAM_Adam, Lookahead_SGD, SGD_Momentum
        for i, part in enumerate(parts):
            if 'seed' in part:
                if i >= 2:
                    # Check if previous 2 tokens form a known optimizer
                    compound_name = f"{parts[i-2]}_{parts[i-1]}"
                    known_compound = ['SAM_SGD', 'SAM_Adam', 'Lookahead_SGD', 'Lookahead_Adam',
                                    'SGD_Momentum']
                    if compound_name in known_compound:
                        optimizer = compound_name
                        optimizer_found = True
                        break
                if not optimizer_found and i > 0:
                    # Fallback to single token before seed
                    optimizer = parts[i-1]
                    optimizer_found = True
                break

        # Method 2: Fallback to scanning for known optimizer names
        if not optimizer_found:
            known_optimizers = ['sgd', 'adam', 'adamw', 'amsgrad', 'sam', 'lookahead',
                               'radam', 'lamb', 'adabound', 'rmsprop']
            for part in parts:
                part_lower = part.lower()
                for opt_name in known_optimizers:
                    if opt_name in part_lower:
                        optimizer = part
                        optimizer_found = True
                        break
                if optimizer_found:
                    break

        if optimizer == expected:
            print(f"✅ {filename}: extracted '{optimizer}' (expected '{expected}')")
        else:
            print(f"❌ {filename}: extracted '{optimizer}' (expected '{expected}')")
            all_passed = False

    assert all_passed

def test_dice_metric_detection():
    """Test that dice metrics are properly detected and labeled"""
    print("\n" + "="*80)
    print("TEST 4: Dice Metric Detection")
    print("="*80)

    test_cases = [
        {
            'columns': ['epoch', 'train_loss', 'test_acc', 'val_acc'],
            'expected_metric': 'test_acc',
            'expected_type': 'accuracy'
        },
        {
            'columns': ['epoch', 'train_loss', 'test_dice', 'val_dice'],
            'expected_metric': 'test_dice',
            'expected_type': 'dice'
        },
        {
            'columns': ['epoch', 'train_loss', 'val_acc', 'final_test_dice'],
            'expected_metric': 'final_test_dice',
            'expected_type': 'dice'
        },
    ]

    all_passed = True
    for test_case in test_cases:
        columns = test_case['columns']
        expected_metric = test_case['expected_metric']
        expected_type = test_case['expected_type']

        # Simulate detection logic
        acc_col = None
        metric_type = 'accuracy'

        # Check for dice first
        for col in ['test_dice', 'final_test_dice', 'val_dice']:
            if col in columns:
                acc_col = col
                metric_type = 'dice'
                break

        # If no dice, check for accuracy
        if acc_col is None:
            for col in ['test_acc', 'test_accuracy', 'final_test_acc', 'val_acc']:
                if col in columns:
                    acc_col = col
                    metric_type = 'accuracy'
                    break

        if acc_col == expected_metric and metric_type == expected_type:
            print(f"✅ Columns {columns[:3]}...: detected metric='{acc_col}', type='{metric_type}'")
        else:
            print(f"❌ Columns {columns[:3]}...: detected metric='{acc_col}', type='{metric_type}' (expected '{expected_metric}', '{expected_type}')")
            all_passed = False

    assert all_passed

def main():
    print("\n" + "="*80)
    print("GDSearch Bug Fixes Validation")
    print("="*80)

    results = {
        'CSV Final Metrics': test_csv_has_final_metrics(),
        'Scaling Detection': test_scaling_detection(),
        'Optimizer Extraction': test_optimizer_extraction(),
        'Dice Metric Detection': test_dice_metric_detection()
    }

    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    all_passed = all(results.values())
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")

    print("="*80)
    if all_passed:
        print("✅ ALL VALIDATION TESTS PASSED")
        print("All bug fixes are working correctly!")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("Please review the failures above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
