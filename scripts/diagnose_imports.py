#!/usr/bin/env python3
"""
Diagnostic script to check all module imports and dependencies.
Helps troubleshoot "module not available" warnings.
"""
import sys
import importlib
from pathlib import Path

def check_module(module_path, import_names):
    """Check if a module and its exports are available."""
    print(f"\n{'='*60}")
    print(f"Checking: {module_path}")
    print(f"{'='*60}")
    
    try:
        # Try to import the module
        module = importlib.import_module(module_path)
        print(f"✅ Module imported successfully")
        
        # Check specific imports
        for name in import_names:
            if hasattr(module, name):
                print(f"   ✅ {name}: Available")
            else:
                print(f"   ❌ {name}: Not found in module")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        
        # Try to diagnose the issue
        module_file = module_path.replace('.', '/') + '.py'
        if Path(module_file).exists():
            print(f"   📄 File exists: {module_file}")
            print(f"   💡 Check for syntax errors in the file")
        else:
            print(f"   📄 File not found: {module_file}")
            print(f"   💡 Check file path and __init__.py files")
        
        return False
    
    except Exception as e:
        print(f"❌ Unexpected error: {type(e).__name__}: {e}")
        return False

def main():
    # Add parent directory to path for imports
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    sys.path.insert(0, str(repo_root))
    
    print("🔍 GDSearch Import Diagnostic Tool")
    print(f"Repository root: {repo_root}")
    print(f"Python path: {sys.path[:3]}")
    
    # Check core modules
    modules_to_check = {
        'src.analysis.statistical_analysis': [
            'compare_multiple_optimizers',
            'perform_ttest_comparison',
            'calculate_cohens_d',
            'run_power_analysis'
        ],
        'src.visualization.interactive_plots': [
            'plot_multi_optimizer_comparison'
        ],
        'src.experiments.convergence_analysis': [
            'ConvergenceAnalyzer'
        ],
        'src.visualization.loss_landscape': [
            'probe_loss_2d',
            'create_loss_landscape'
        ]
    }
    
    results = {}
    for module_path, imports in modules_to_check.items():
        results[module_path] = check_module(module_path, imports)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    success_count = sum(results.values())
    total_count = len(results)
    
    for module_path, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {module_path}")
    
    print(f"\n{success_count}/{total_count} modules available")
    
    if success_count < total_count:
        print("\n💡 To fix import issues:")
        print("   1. Check syntax errors: python -m py_compile <file>")
        print("   2. Verify dependencies: pip install -r requirements.txt")
        print("   3. Check __init__.py files exist in all src/ subdirectories")

if __name__ == '__main__':
    main()
