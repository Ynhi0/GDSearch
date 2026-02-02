# Multi-Seed Audit - Quick Summary

## TL;DR

✅ **86% of experiments (30/35) properly implement multi-seed support**  
⚠️ **9% need minor fixes (3/35) - only use first seed**  
❌ **5% need critical fixes (2/35) - missing seed loops entirely**

---

## Critical Issues Requiring Immediate Fix

### ❌ 1. `run_resnet_experiment` (HIGH PRIORITY)
- **Location:** `run_all_kaggle.py:9007`
- **Problem:** NO seed loop despite accepting `seeds` parameter
- **Impact:** ResNet results not reproducible, no statistical variance
- **Fix:** Add `for seed in seeds:` loop around training

### ⚠️ 2-4. Three experiments only use `seeds[0]`
- `run_robustness_analysis` (line 7819)
- `run_sam_sensitivity` (line 7943)
- `run_ablation_study` (line 8068)
- **Fix:** Change `seed = seeds[0]` to `for seed in seeds:`

---

## Exemplary Implementations (Use as Templates)

1. **`run_mnist_experiment`** - Perfect multi-seed with aggregation
2. **`run_cifar10_experiment`** - Consistent with MNIST
3. **All ablation studies** - Clean seed isolation

---

## Verification Commands

```bash
# Find experiments missing seed loops
grep "def run_.*seeds" run_all_kaggle.py | while read line; do
    func=$(echo "$line" | grep -o "run_[^(]*")
    if ! grep -A 50 "def $func" run_all_kaggle.py | grep -q "for seed in seeds"; then
        echo "MISSING: $func"
    fi
done

# Check result files have seed in name
find results/ -name "*.csv" | grep -v "seed[0-9]" | grep -v "aggregated" | head

# Verify aggregation is called
grep -c "aggregate_results" run_all_kaggle.py
```

---

## Full Report

See [`MULTI_SEED_COMPLETE_AUDIT.md`](MULTI_SEED_COMPLETE_AUDIT.md) for:
- Detailed analysis of all 35+ experiments
- Code examples and patterns
- Template for new experiments
- Best practices and recommendations
