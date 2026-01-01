#!/bin/bash
# Quick Test Suite - Verify all scripts work (5 minutes)
# Integration validation for Phase 5 + QA improvements

set -e  # Exit on first error

echo "=========================================="
echo "QUICK TEST SUITE - GDSearch"
echo "=========================================="
echo "Estimated runtime: 5 minutes"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

run_test() {
    local test_name="$1"
    local test_cmd="$2"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo -e "${YELLOW}[TEST $TOTAL_TESTS]${NC} $test_name"
    echo "Command: $test_cmd"
    
    if eval "$test_cmd"; then
        echo -e "${GREEN}✓ PASS${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${RED}✗ FAIL${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
    echo ""
}

# Test 1: Import safety test
run_test "Import Safety (No side effects)" \
    "python scripts/quick_validation_test.py --verbose 2>&1 | grep -q 'PASS'"

# Test 2: Reproducibility setup
run_test "Reproducibility Setup" \
    "python -c 'from src.utils.reproducibility import setup_experiment_reproducibility; setup_experiment_reproducibility(seed=42); print(\"PASS\")' | grep -q 'PASS'"

# Test 3: Condition number sweep (ultra-quick)
run_test "Condition Number Sweep (Ultra-Quick)" \
    "python run_condition_number_sweep.py --ultra-quick 2>&1 | grep -q 'Condition number sweep complete'"

# Test 4: SimpleMLP BN ablation (ultra-quick)
run_test "SimpleMLP BN Ablation (Ultra-Quick)" \
    "python run_simplemlp_bn_ablation.py --ultra-quick 2>&1 | grep -q 'SimpleMLP BN ablation complete'"

# Test 5: NLP full data flag
run_test "NLP Full Data Flag" \
    "python src/experiments/run_transformer_nlp.py --full-data --seeds 42 --epochs 1 2>&1 | grep -q 'FULL DATA MODE'"

# Test 6: Main experiment runner (ultra-quick)
run_test "Main Runner (Ultra-Quick Mode)" \
    "python run_all_kaggle.py --ultra-quick --seeds 42 --deterministic --no-mlflow 2>&1 | grep -q 'Experiment complete'"

# Test 7: Adaptive convergence detection
run_test "Adaptive Convergence Detection" \
    "python -c 'from src.utils.convergence_detection import AdaptiveConvergenceDetector; d = AdaptiveConvergenceDetector(); print(\"PASS\")' | grep -q 'PASS'"

# Test 8: Anti-aliasing plots
run_test "Anti-Aliasing Plot Module" \
    "python -c 'from src.visualization.antialiasing_plots import plot_with_envelope; print(\"PASS\")' | grep -q 'PASS'"

# Test 9: Dynamics tracker (disk-based)
run_test "Dynamics Tracker (Disk-Based Logging)" \
    "python -c 'from src.core.dynamics_tracker import TrainingDynamicsTracker; t = TrainingDynamicsTracker(param_snapshot_dir=\"test_snapshots\"); print(\"PASS\")' | grep -q 'PASS'"

# Test 10: Reproducibility verification (dynamic)
run_test "Dynamic Reproducibility Verification" \
    "python -c 'from src.core.reproducibility import verify_checkpoint_with_metadata; print(\"PASS\")' | grep -q 'PASS'"

# Summary
echo "=========================================="
echo "TEST SUMMARY"
echo "=========================================="
echo "Total tests: $TOTAL_TESTS"
echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"
echo -e "Failed: ${RED}$FAILED_TESTS${NC}"

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}✓ ALL TESTS PASSED${NC}"
    exit 0
else
    echo -e "${RED}✗ SOME TESTS FAILED${NC}"
    exit 1
fi
