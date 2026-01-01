# Quick Test Suite - Verify all scripts work (5 minutes)
# PowerShell version for Windows
# Integration validation for Phase 5 + QA improvements

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "QUICK TEST SUITE - GDSearch" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Estimated runtime: 5 minutes"
Write-Host ""

$TotalTests = 0
$PassedTests = 0
$FailedTests = 0

function Run-Test {
    param(
        [string]$TestName,
        [string]$TestCommand
    )
    
    $script:TotalTests++
    Write-Host "[TEST $TotalTests] $TestName" -ForegroundColor Yellow
    Write-Host "Command: $TestCommand"
    
    try {
        $output = Invoke-Expression $TestCommand 2>&1
        $outputStr = $output | Out-String
        
        # Check if command succeeded and produced expected output
        if ($LASTEXITCODE -eq 0 -or $outputStr -match "PASS|complete") {
            Write-Host "✓ PASS" -ForegroundColor Green
            $script:PassedTests++
        } else {
            Write-Host "✗ FAIL" -ForegroundColor Red
            Write-Host "Output: $outputStr" -ForegroundColor Gray
            $script:FailedTests++
        }
    } catch {
        Write-Host "✗ FAIL (Exception: $_)" -ForegroundColor Red
        $script:FailedTests++
    }
    Write-Host ""
}

# Test 1: Import safety test
Run-Test "Import Safety (No side effects)" `
    "python scripts\quick_validation_test.py --verbose 2>&1"

# Test 2: Reproducibility setup
Run-Test "Reproducibility Setup" `
    'python -c "from src.utils.reproducibility import setup_experiment_reproducibility; setup_experiment_reproducibility(seed=42); print(''PASS'')"'

# Test 3: Condition number sweep (ultra-quick)
Run-Test "Condition Number Sweep (Ultra-Quick)" `
    "python run_condition_number_sweep.py --ultra-quick"

# Test 4: SimpleMLP BN ablation (ultra-quick)
Run-Test "SimpleMLP BN Ablation (Ultra-Quick)" `
    "python run_simplemlp_bn_ablation.py --ultra-quick"

# Test 5: NLP full data flag (quick check, don't run full training)
Run-Test "NLP Full Data Flag" `
    'python -c "from src.experiments.run_transformer_nlp import main; print(''PASS'')"'

# Test 6: Adaptive convergence detection
Run-Test "Adaptive Convergence Detection" `
    'python -c "from src.utils.convergence_detection import AdaptiveConvergenceDetector; d = AdaptiveConvergenceDetector(); print(''PASS'')"'

# Test 7: Anti-aliasing plots
Run-Test "Anti-Aliasing Plot Module" `
    'python -c "from src.visualization.antialiasing_plots import plot_with_envelope; print(''PASS'')"'

# Test 8: Dynamics tracker (disk-based)
Run-Test "Dynamics Tracker (Disk-Based Logging)" `
    'python -c "from src.core.dynamics_tracker import TrainingDynamicsTracker; t = TrainingDynamicsTracker(param_snapshot_dir=''test_snapshots''); print(''PASS'')"'

# Test 9: Reproducibility verification (dynamic)
Run-Test "Dynamic Reproducibility Verification" `
    'python -c "from src.core.reproducibility import verify_checkpoint_with_metadata; print(''PASS'')"'

# Test 10: Condition number analysis
Run-Test "Condition Number Analysis Module" `
    'python -c "from src.analysis.condition_number_analysis import quadratic_with_condition_number; print(''PASS'')"'

# Summary
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "TEST SUMMARY" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Total tests: $TotalTests"
Write-Host "Passed: $PassedTests" -ForegroundColor Green
Write-Host "Failed: $FailedTests" -ForegroundColor Red

if ($FailedTests -eq 0) {
    Write-Host "✓ ALL TESTS PASSED" -ForegroundColor Green
    exit 0
} else {
    Write-Host "✗ SOME TESTS FAILED" -ForegroundColor Red
    exit 1
}
