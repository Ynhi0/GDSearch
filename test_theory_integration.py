"""
Quick test script to verify theory analysis integration in run_all_kaggle.py
"""
import sys
from pathlib import Path
from typing import Dict, Any

# Import the theory analysis pipeline function
sys.path.insert(0, str(Path(__file__).parent))

def test_theory_pipeline_import():
    """Test that we can import the theory analysis pipeline function."""
    try:
        from run_all_kaggle import run_theory_analysis_pipeline
        print("✓ Successfully imported run_theory_analysis_pipeline")
        return True
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        return False

def test_theory_pipeline_dry_run():
    """Test the theory analysis pipeline in dry-run mode."""
    try:
        from run_all_kaggle import run_theory_analysis_pipeline
        
        # Create mock inputs
        results_dir = Path('results')
        experiment_results = {}
        
        print("\nRunning theory analysis pipeline (dry-run mode)...")
        result = run_theory_analysis_pipeline(
            results_dir=results_dir,
            experiment_results=experiment_results,
            dry_run=True  # Don't actually execute, just show what would happen
        )
        
        print(f"\n✓ Dry-run completed successfully")
        print(f"   Result keys: {list(result.keys())}")
        return True
        
    except Exception as e:
        print(f"\n✗ Dry-run failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_advanced_bounds_import():
    """Test that we can import the advanced bounds module."""
    try:
        from src.analysis.advanced_bounds import (
            saddle_escape_time_bound,
            adam_nonconvex_full_bound,
            hessian_based_tighter_bound,
            variance_reduction_bound
        )
        print("✓ Successfully imported advanced_bounds module")
        return True
    except ImportError as e:
        print(f"⚠ Advanced bounds module not available: {e}")
        return False

if __name__ == '__main__':
    print("="*80)
    print("THEORY INTEGRATION TEST")
    print("="*80)
    
    results = []
    
    print("\n[Test 1] Import theory pipeline function...")
    results.append(test_theory_pipeline_import())
    
    print("\n[Test 2] Import advanced bounds module...")
    results.append(test_advanced_bounds_import())
    
    print("\n[Test 3] Run theory pipeline (dry-run)...")
    results.append(test_theory_pipeline_dry_run())
    
    print("\n" + "="*80)
    passed = sum(results)
    total = len(results)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("="*80)
    
    sys.exit(0 if all(results) else 1)
