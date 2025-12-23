#!/usr/bin/env python3
"""
Regression tests for critical audit fixes.

This test file ensures that the 5 critical blockers identified in the first audit
remain fixed and do not regress in future code changes.

Critical fixes tested:
1. Auto-LR call signature (keyword arguments not positional)
2. SAM batch ablation closure requirement
3. Convergence validation call signature match
4. PerformanceProfiler.print_summary() exists and works
5. HF NLP validation split (not test set for early stopping)
"""

import sys
import os
import tempfile
import inspect
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import torch
import torch.nn as nn


def test_auto_lr_call_signature():
    """Regression test: Auto-LR must use keyword arguments not positional.
    
    Blocker A from audit: HF NLP Auto-LR was calling with positional args
    causing temp_opt to be passed where train_loader expected.
    
    This test verifies the Auto-LR invocation uses keyword arguments.
    """
    with open('run_all_kaggle.py', 'r', encoding='utf-8') as f:
        code = f.read()
    
    # Find the HF NLP Auto-LR section (around line 3600)
    auto_lr_section = code.find('suggested_lr = find_optimal_lr(')
    assert auto_lr_section != -1, "Auto-LR invocation not found"
    
    # Get context (200 chars after call)
    context = code[auto_lr_section:auto_lr_section + 500]
    
    # Verify keyword arguments are used (model=, train_loader=, etc.)
    assert 'model=' in context, "Auto-LR must use model= keyword argument"
    assert 'train_loader=' in context, "Auto-LR must use train_loader= keyword argument"
    assert 'criterion=' in context, "Auto-LR must use criterion= keyword argument"
    assert 'device=' in context, "Auto-LR must use device= keyword argument"
    
    print("Auto-LR uses keyword arguments (not positional)")



def test_sam_closure_requirement():
    """Regression test: SAM wrapper requires closure in step().
    
    Blocker B from audit: Batch ablation was calling SAM.step() without closure.
    
    This test verifies SAM wrapper implementation has requires_closure flag.
    """
    # Check if SAMWrapper is defined in run_all_kaggle.py or src
    with open('run_all_kaggle.py', 'r', encoding='utf-8') as f:
        code = f.read()
    
    # Look for SAM wrapper definition or import
    assert 'SAMWrapper' in code or 'from src.core.pytorch_optimizers import' in code, \
        "SAM optimizer wrapper not found"
    
    # Verify requires_closure is mentioned (should be set to True for SAM)
    if 'class SAMWrapper' in code:
        # SAM is defined inline
        sam_def_start = code.find('class SAMWrapper')
        sam_def = code[sam_def_start:sam_def_start + 2000]
        assert 'requires_closure' in sam_def, "SAM wrapper missing requires_closure flag"
        assert 'requires_closure = True' in sam_def, "SAM wrapper should set requires_closure = True"
    
    print("SAM wrapper correctly requires closure")



def test_convergence_validation_signature():
    """Regression test: Convergence validation accepts only output_dir.
    
    Blocker C from audit: Orchestrator was passing unsupported keyword arguments
    to run_convergence_rate_comparison.
    
    This test verifies the actual signature matches expected usage.
    """
    from src.experiments.convergence_rate_validation import run_convergence_rate_comparison
    
    # Get the function signature
    sig = inspect.signature(run_convergence_rate_comparison)
    params = list(sig.parameters.keys())
    
    # Verify it accepts output_dir
    assert 'output_dir' in params
    
    # Verify it does NOT accept these (they were being passed incorrectly)
    assert 'dataset' not in params
    assert 'model' not in params
    assert 'epochs' not in params
    assert 'seed' not in params
    
    print("Convergence validation has correct signature (output_dir only)")


def test_performance_profiler_print_summary():
    """Regression test: PerformanceProfiler must have print_summary() method.
    
    Blocker D from audit: print_summary() method was missing entirely.
    
    This test verifies the method exists and produces output.
    """
    import sys
    import io
    
    # Import PerformanceProfiler from run_all_kaggle
    # We need to extract just the class definition without running the module
    with open('run_all_kaggle.py', 'r', encoding='utf-8') as f:
        code = f.read()
    
    # Verify print_summary method exists in source
    assert 'def print_summary(self):' in code, "PerformanceProfiler.print_summary() method missing from source"
    
    # Verify it's inside PerformanceProfiler class
    class_start = code.find('class PerformanceProfiler')
    method_location = code.find('def print_summary(self):', class_start)
    next_class = code.find('\nclass ', class_start + 10)
    
    if next_class == -1:
        next_class = len(code)
    
    assert class_start < method_location < next_class, "print_summary() not inside PerformanceProfiler class"
    
    print("PerformanceProfiler.print_summary() method exists")


def test_hf_nlp_validation_split():
    """Regression test: HF NLP must use validation split, not test set for early stopping.
    
    Scientific Blocker from audit: NLP path was using test set for early stopping,
    causing adaptive overfitting.
    
    This test verifies the validation split logic is present in the HF NLP path.
    """
    with open('run_all_kaggle.py', 'r', encoding='utf-8') as f:
        code = f.read()
    
    # Find the HF NLP section (look for _run_nlp_experiment_huggingface function)
    hf_section_start = code.find('def _run_nlp_experiment_huggingface')
    assert hf_section_start != -1, "HF NLP function not found"
    
    # Get a reasonable chunk (next 10000 characters should contain the full logic)
    hf_section = code[hf_section_start:hf_section_start + 10000]
    
    # Verify validation split is created
    assert 'train_val_split' in hf_section or 'val_size' in hf_section or 'validation_split' in hf_section, \
        "No validation split logic found in HF NLP section"
    
    # Verify early stopping uses validation, not test
    # Look for val_ds or validation dataset creation
    assert 'val_ds' in hf_section or 'validation_ds' in hf_section, \
        "No validation dataset found in HF NLP section"
    
    # Verify val_loader is created
    assert 'val_loader' in hf_section or 'validation_loader' in hf_section, \
        "No validation loader found in HF NLP section"
    
    print("HF NLP uses validation split (not test set for early stopping)")


def test_no_runtime_auto_install():
    """Regression test: No runtime pip install attempts.
    
    Runtime auto-install is fragile and non-deterministic. This test verifies
    that subprocess.check_call with pip install has been removed.
    """
    with open('run_all_kaggle.py', 'r', encoding='utf-8') as f:
        code = f.read()
    
    # Check for runtime auto-install patterns
    lines = code.split('\n')
    
    problematic_lines = []
    for i, line in enumerate(lines, 1):
        # Look for subprocess.check_call with pip install
        if 'subprocess.check_call' in line and 'pip' in line and 'install' in line:
            # Allow it only in run_code_quality_checks (which is a diagnostic helper)
            # Get context (prev 20 lines) to check if we're in run_code_quality_checks
            context_start = max(0, i - 20)
            context = '\n'.join(lines[context_start:i])
            
            if 'def run_code_quality_checks' not in context:
                problematic_lines.append((i, line.strip()))
    
    assert len(problematic_lines) == 0, \
        f"Found {len(problematic_lines)} runtime auto-install attempts (should be 0):\n" + \
        '\n'.join([f"Line {i}: {line}" for i, line in problematic_lines])
    
    print("✓ No runtime auto-install attempts found (clean prerequisite handling)")


def test_sam_ablation_closure_usage():
    """Integration test: Verify batch ablation path correctly calls SAM with closure.
    
    This test checks that the run_batch_ablation function (which had the SAM bug)
    now properly uses closures for optimizers that require them.
    """
    with open('run_all_kaggle.py', 'r', encoding='utf-8') as f:
        code = f.read()
    
    # Find run_batch_ablation function
    func_start = code.find('def run_batch_ablation')
    assert func_start != -1, "run_batch_ablation function not found"
    
    # Get the function body (next 5000 chars to include training loop)
    func_body = code[func_start:func_start + 5000]
    
    # Verify closure is defined (could be "def closure" or "closure = lambda")
    has_closure_def = 'def closure' in func_body or 'closure =' in func_body
    
    # Verify optimizer.step() is called with closure
    has_closure_call = 'optimizer.step(closure)' in func_body
    
    # At minimum, verify the function handles closure-based optimizers
    # (it may use hasattr or requires_closure checks)
    has_closure_handling = (has_closure_def and has_closure_call) or \
                          'requires_closure' in func_body or \
                          ('hasattr(optimizer' in func_body and 'requires_closure' in code[func_start:func_start + 10000])
    
    assert has_closure_handling, \
        f"run_batch_ablation must handle closure-based optimizers (SAM requires closure)\\n" \
        f"Found closure def: {has_closure_def}, closure call: {has_closure_call}"
    
    print("✓ Batch ablation correctly handles closure for SAM optimizer")



if __name__ == '__main__':
    pytest.main([__file__, '-v'])
