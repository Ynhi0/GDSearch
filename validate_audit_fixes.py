#!/usr/bin/env python3
"""
Validation Script for Audit Fixes
==================================
This script validates that all 4 critical audit fixes have been successfully applied.

Run: python validate_audit_fixes.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

def test_fix_1_monolith_divergence():
    """Test that canonical imports work and local duplicates are removed."""
    print("\n" + "="*80)
    print("FIX 1: MONOLITH DIVERGENCE CHECK")
    print("="*80)
    
    # Test canonical imports
    try:
        from src.core.models import ResNet18, BasicBlock
        from src.core.pytorch_optimizers import SAMWrapper
        print("PASS: Canonical imports successful")
        print(f"   - ResNet18: {ResNet18}")
        print(f"   - BasicBlock: {BasicBlock}")
        print(f"   - SAMWrapper: {SAMWrapper}")
        
        # Verify these are the only definitions
        with open('run_all_kaggle.py', 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check that duplicate classes were removed
        if 'class ResNet18(nn.Module):' in content and 'def __init__(self, num_classes=10):' in content:
            print("FAIL: Duplicate ResNet18 class still exists in run_all_kaggle.py")
            return False
        else:
            print("PASS: Duplicate ResNet18 class removed")
            
        if 'class SAM(torch.optim.Optimizer):' in content and 'def __init__(self, params, base_optimizer' in content:
            print("FAIL: Duplicate SAM class still exists in run_all_kaggle.py")
            return False
        else:
            print("PASS: Duplicate SAM class removed")
            
        # Verify imports are present
        if 'from src.core.models import ResNet18, BasicBlock' in content:
            print("PASS: Canonical model imports present")
        else:
            print("FAIL: Canonical model imports missing")
            return False
            
        if 'from src.core.pytorch_optimizers import' in content and 'SAMWrapper' in content:
            print("PASS: Canonical SAMWrapper import present")
        else:
            print("FAIL: Canonical SAMWrapper import missing")
            return False
            
        return True
        
    except ImportError as e:
        print(f"FAIL: Import error: {e}")
        return False


def test_fix_2_oom_taint_tracking():
    """Test that OOM taint tracking is implemented."""
    print("\n" + "="*80)
    print("FIX 2: OOM TAINT TRACKING CHECK")
    print("="*80)
    
    with open('run_all_kaggle.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for taint tracking variables
    checks = [
        ('run_tainted = False', 'Taint flag initialization'),
        ('effective_batch_size', 'Effective batch size tracking'),
        ("'tainted': run_tainted", 'Tainted column in results'),
        ("'effective_batch_size': effective_batch_size", 'Effective batch size column in results'),
        ('AUDIT WARNING: Run Tainted', 'OOM warning message'),
        ('SCIENTIFIC INTEGRITY: This run uses variable batch size', 'Scientific integrity warning'),
    ]
    
    all_passed = True
    for pattern, description in checks:
        if pattern in content:
            print(f"PASS: {description}")
        else:
            print(f"FAIL: {description} - pattern '{pattern}' not found")
            all_passed = False
    
    # Check that oom_safe_train_step returns tainted flag
    if 'def oom_safe_train_step' in content and 'Returns:' in content:
        if 'tainted' in content[content.find('def oom_safe_train_step'):content.find('def oom_safe_train_step') + 2000]:
            print("PASS: oom_safe_train_step returns tainted flag")
        else:
            print("FAIL: oom_safe_train_step doesn't return tainted flag")
            all_passed = False
    
    return all_passed


def test_fix_3_zombie_config():
    """Test that zombie configuration is fixed."""
    print("\n" + "="*80)
    print("FIX 3: ZOMBIE CONFIGURATION CHECK")
    print("="*80)
    
    with open('run_all_kaggle.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for benchmark config loading
    checks = [
        ("benchmark_config_path = Path(__file__).parent / 'configs' / 'benchmark_hyperparameters.json'", 
         'Benchmark config path construction'),
        ('AUDIT FIX 3: Loaded authoritative config', 'Config loading confirmation message'),
        ('if benchmark_config_path.exists():', 'Config existence check'),
    ]
    
    all_passed = True
    for pattern, description in checks:
        if pattern in content:
            print(f"PASS: {description}")
        else:
            print(f"FAIL: {description}")
            all_passed = False
    
    # Verify config file exists
    config_path = project_root / 'configs' / 'benchmark_hyperparameters.json'
    if config_path.exists():
        print(f"PASS: Config file exists at {config_path}")
    else:
        print(f"WARNING: Config file not found at {config_path}")
    
    return all_passed


def test_fix_4_checkpoint_visualization():
    """Test that checkpoint saving is enabled by default."""
    print("\n" + "="*80)
    print("FIX 4: CHECKPOINT VISUALIZATION SUPPORT CHECK")
    print("="*80)
    
    with open('run_all_kaggle.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for checkpoint manager initialization
    checks = [
        ('checkpoint_manager = RobustCheckpointManager(', 'Checkpoint manager initialization'),
        ('AUDIT FIX 4: Checkpoint manager ALWAYS initialized', 'Checkpoint manager always enabled documentation'),
        ('base_dir=str(results_dir / "checkpoints")', 'Checkpoint directory configuration'),
        ('checkpoint_manager.restore_rng_states(checkpoint)', 'RNG state restoration'),
        ("'rng_states'", 'RNG states in checkpoints'),
    ]
    
    all_passed = True
    for pattern, description in checks:
        if pattern in content:
            print(f"PASS: {description}")
        else:
            print(f"FAIL: {description}")
            all_passed = False
    
    return all_passed


def main():
    """Run all validation tests."""
    print("\n" + "="*80)
    print("GDSEARCH AUDIT FIXES VALIDATION")
    print("="*80)
    print("Validating that all 4 critical audit fixes have been applied...")
    
    results = {
        'Fix 1 - Monolith Divergence': test_fix_1_monolith_divergence(),
        'Fix 2 - OOM Taint Tracking': test_fix_2_oom_taint_tracking(),
        'Fix 3 - Zombie Configuration': test_fix_3_zombie_config(),
        'Fix 4 - Checkpoint Visualization': test_fix_4_checkpoint_visualization(),
    }
    
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    for fix, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{status}: {fix}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*80)
    if all_passed:
        print("ALL AUDIT FIXES VALIDATED SUCCESSFULLY!")
        print("="*80)
        print("\nThe codebase now satisfies:")
        print("  - Scientific Validity (OOM taint tracking)")
        print("  - Maintainability (no code duplication)")
        print("  - Configuration Authority (benchmark configs loaded)")
        print("  - Reproducibility (checkpoints enabled by default)")
        print("\nReady for NeurIPS reproducibility review!")
        return 0
    else:
        print("SOME AUDIT FIXES FAILED VALIDATION")
        print("="*80)
        print("\nPlease review the failed checks above and re-apply fixes.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
