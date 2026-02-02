"""
Test critical bug fixes for file handle leaks and type coercion.
"""
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.csv_utils import safe_read_csv
from src.utils.metric_normalization import extract_metric


def test_csv_file_handle_leak():
    """Test that file handles are properly closed even on error paths."""
    print("Testing BUG #1: File handle leak prevention...")
    
    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("col1,col2\n")
        f.write("1,2\n")
        f.write("3,4\n")
        temp_path = f.name
    
    try:
        # Normal read
        df = safe_read_csv(temp_path)
        assert df is not None
        assert df.shape == (2, 2)
        print("  ✓ Normal read successful")
        
        # Test that file can be deleted (no open handles)
        Path(temp_path).unlink()
        print("  ✓ File handle properly released")
        
        # Test with non-existent file (should raise CSVReadError)
        try:
            safe_read_csv("nonexistent.csv")
            assert False, "Should have raised CSVReadError"
        except Exception as e:
            assert "does not exist" in str(e)
            print("  ✓ Non-existent file handling correct")
        
        # Test with empty file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_empty = f.name
        
        result = safe_read_csv(temp_empty)
        assert result is None
        Path(temp_empty).unlink()
        print("  ✓ Empty file handling correct")
        
        print("✓ BUG #1 FIX VERIFIED: File handles properly managed\n")
        return True
    except Exception as e:
        print(f"✗ BUG #1 FIX FAILED: {e}\n")
        return False


def test_type_coercion_precision():
    """Test that extract_metric always returns scalar, never Series."""
    print("Testing BUG #2: Type coercion precision...")
    
    # Test single-row DataFrame
    df1 = pd.DataFrame({'test_accuracy': [0.92]})
    result1 = extract_metric(df1, 'test_accuracy')
    assert isinstance(result1, float), f"Expected float, got {type(result1)}"
    assert result1 == 0.92
    print("  ✓ Single-row returns scalar")
    
    # Test multi-row DataFrame with default aggregation (last)
    df2 = pd.DataFrame({'test_accuracy': [0.85, 0.90, 0.92]})
    result2 = extract_metric(df2, 'test_accuracy')
    assert isinstance(result2, float), f"Expected float, got {type(result2)}"
    assert result2 == 0.92  # Should return last value
    print("  ✓ Multi-row with default aggregation returns scalar (last)")
    
    # Test multi-row with different aggregations
    result_first = extract_metric(df2, 'test_accuracy', aggregation='first')
    assert result_first == 0.85
    print("  ✓ Multi-row with 'first' aggregation returns scalar")
    
    result_mean = extract_metric(df2, 'test_accuracy', aggregation='mean')
    assert abs(result_mean - 0.89) < 0.01  # (0.85 + 0.90 + 0.92) / 3
    print("  ✓ Multi-row with 'mean' aggregation returns scalar")
    
    result_min = extract_metric(df2, 'test_accuracy', aggregation='min')
    assert result_min == 0.85
    print("  ✓ Multi-row with 'min' aggregation returns scalar")
    
    result_max = extract_metric(df2, 'test_accuracy', aggregation='max')
    assert result_max == 0.92
    print("  ✓ Multi-row with 'max' aggregation returns scalar")
    
    # Test with alias
    df3 = pd.DataFrame({'test_acc': [0.88]})
    result3 = extract_metric(df3, 'test_accuracy')  # Using standard name
    assert isinstance(result3, float), f"Expected float, got {type(result3)}"
    assert result3 == 0.88
    print("  ✓ Alias resolution works correctly")
    
    # Test with missing metric
    result4 = extract_metric(df1, 'nonexistent_metric', default=0.0)
    assert result4 == 0.0
    print("  ✓ Missing metric returns default")
    
    # Test with numpy scalars
    df5 = pd.DataFrame({'test_accuracy': [np.float64(0.95)]})
    result5 = extract_metric(df5, 'test_accuracy')
    assert isinstance(result5, float), f"Expected float, got {type(result5)}"
    assert result5 == 0.95
    print("  ✓ Numpy scalar coercion works")
    
    # Test with NaN values
    df6 = pd.DataFrame({'test_accuracy': [np.nan, 0.88, np.nan, 0.92]})
    result6 = extract_metric(df6, 'test_accuracy', aggregation='last')
    assert isinstance(result6, float), f"Expected float, got {type(result6)}"
    assert result6 == 0.92
    print("  ✓ NaN handling works correctly")
    
    print("✓ BUG #2 FIX VERIFIED: Always returns scalar, never Series\n")
    return True


if __name__ == "__main__":
    print("=" * 70)
    print("CRITICAL BUG FIX VALIDATION")
    print("=" * 70)
    print()
    
    results = []
    results.append(test_csv_file_handle_leak())
    results.append(test_type_coercion_precision())
    
    print("=" * 70)
    if all(results):
        print("✓ ALL CRITICAL BUG FIXES VERIFIED")
        print("=" * 70)
        exit(0)
    else:
        print("✗ SOME BUG FIXES FAILED")
        print("=" * 70)
        exit(1)
