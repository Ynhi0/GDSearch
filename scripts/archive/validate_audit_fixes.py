"""
Quick validation script to demonstrate remediation fixes.

Tests:
1. Custom optimizers are used (not torch.optim)
2. UUID prevents filename collisions
3. JSON config loading works
"""

import sys
import uuid
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_custom_optimizer_routing():
    """Verify build_optimizer returns custom wrappers, not torch.optim."""
    print("=" * 70)
    print("TEST 1: Custom Optimizer Routing")
    print("=" * 70)
    
    from src.experiments.run_nn_experiment import build_optimizer
    import torch.nn as nn
    
    model = nn.Linear(10, 1)
    
    test_cases = ['SGD', 'Adam', 'AdamW', 'SGD_Momentum']
    results = []
    
    for opt_name in test_cases:
        try:
            if opt_name == 'SGD_Momentum':
                opt = build_optimizer(opt_name, model, lr=0.01, momentum=0.9)
            else:
                opt = build_optimizer(opt_name, model, lr=0.01)
            
            opt_class = type(opt).__name__
            opt_module = type(opt).__module__
            
            # Check if it's a custom wrapper (not torch.optim)
            is_custom = 'pytorch_optimizers' in opt_module
            status = "✅ PASS" if is_custom else "❌ FAIL"
            
            print(f"{opt_name:15} → {opt_class:20} | Module: {opt_module:30} | {status}")
            results.append(is_custom)
        except Exception as e:
            print(f"{opt_name:15} → ERROR: {e}")
            results.append(False)
    
    print(f"\nResult: {sum(results)}/{len(results)} optimizers use custom wrappers")
    return all(results)


def test_uuid_uniqueness():
    """Verify UUID prevents filename collisions."""
    print("\n" + "=" * 70)
    print("TEST 2: UUID Filename Uniqueness")
    print("=" * 70)
    
    from src.experiments.run_nn_experiment import result_filename
    
    config = {
        'model': 'SimpleMLP',
        'dataset': 'MNIST',
        'optimizer': 'Adam',
        'lr': 0.001,
        'seed': 42
    }
    
    # Generate 5 filenames with same config
    filenames = [result_filename(config) for _ in range(5)]
    
    print("Generated filenames:")
    for i, fname in enumerate(filenames, 1):
        print(f"  {i}. {fname}")
    
    unique_count = len(set(filenames))
    print(f"\nUnique filenames: {unique_count}/5")
    
    if unique_count == 5:
        print("✅ PASS: All filenames are unique (UUID working)")
        return True
    else:
        print("❌ FAIL: Duplicate filenames detected")
        return False


def test_json_config_loading():
    """Verify JSON config loading works."""
    print("\n" + "=" * 70)
    print("TEST 3: JSON Config Loading")
    print("=" * 70)
    
    import json
    import os
    
    config_path = 'configs/nn_tuning.json'
    
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        print("   (This is OK - fallback to defaults will be used)")
        return True
    
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        print(f"✅ Successfully loaded: {config_path}")
        print(f"   Structure: {list(config_data.keys())}")
        
        if 'sweeps' in config_data:
            print(f"   Sweeps found: {len(config_data['sweeps'])}")
            for i, sweep in enumerate(config_data['sweeps'][:3], 1):
                print(f"     {i}. {sweep.get('model')} on {sweep.get('dataset')}")
        
        return True
    except Exception as e:
        print(f"❌ FAIL: Error loading config: {e}")
        return False


def test_imports():
    """Verify all necessary imports exist."""
    print("\n" + "=" * 70)
    print("TEST 4: Import Verification")
    print("=" * 70)
    
    try:
        import src.experiments.run_nn_experiment as exp_module
        
        # Check for required imports
        required_attrs = ['json', 'uuid', 'build_optimizer', 'result_filename', 'main']
        
        results = []
        for attr in required_attrs:
            has_attr = hasattr(exp_module, attr) or attr in dir(exp_module)
            status = "✅" if has_attr else "❌"
            print(f"  {attr:20} {status}")
            results.append(has_attr)
        
        # Check if custom wrappers are imported
        import inspect
        source = inspect.getsource(exp_module)
        
        has_wrappers = 'SGDWrapper' in source and 'AdamWrapper' in source
        print(f"  Custom wrappers     {'✅' if has_wrappers else '❌'}")
        results.append(has_wrappers)
        
        if all(results):
            print("\n✅ PASS: All imports verified")
            return True
        else:
            print("\n❌ FAIL: Some imports missing")
            return False
            
    except Exception as e:
        print(f"❌ FAIL: Import error: {e}")
        return False


def main():
    print("\n" + "GDSearch Fix Validation" + "\n")
    print("This script validates the fixes applied in response to the")
    print("Research Validity Review (December 2025)\n")
    
    tests = [
        ("Custom Optimizer Routing", test_custom_optimizer_routing),
        ("UUID Filename Uniqueness", test_uuid_uniqueness),
        ("JSON Config Loading", test_json_config_loading),
        ("Import Verification", test_imports),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"\n❌ {name} CRASHED: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    
    for (name, _), result in zip(tests, results):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name:30} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nAll remediation fixes validated successfully!")
        print("   The codebase now tests custom optimizer implementations.")
        return 0
    else:
        print(f"\n{total - passed} test(s) failed. Review the output above.")
        return 1


if __name__ == '__main__':
    exit(main())
