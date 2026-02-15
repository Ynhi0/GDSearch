# COMPREHENSIVE NAMING & MULTI-SEED AUDIT - COMPLETE FINDINGS

**Date:** February 2, 2026  
**Agents:** 6 specialized audit agents  
**Scope:** Full codebase - naming consistency, multi-seed support, hardcoded values

---

## EXECUTIVE SUMMARY

**Total Issues Found:** 200+ across all categories

**CRITICAL FINDINGS:**
1. ❌ **BLOCKER:** No seed aggregation after experiments complete - results never combined into mean±std
2. ❌ **12 Naming Inconsistencies** - SGD_Momentum vs SGDMomentum, CIFAR10 naming inconsistency, etc.
3. ❌ **150+ Hardcoded Values** - Seeds, batch sizes, LRs prevent flexible experimentation
4. ✅ **Multi-Seed Support:** 97% of experiments (27/30) properly support multiple seeds
5. ⚠️ **70+ Documentation Files** - Need cleanup and reorganization

---

## AGENT 1: NAMING INCONSISTENCIES (12 Critical Issues)

### Issue #1: SGD_Momentum vs SGDMomentum ⚠️⚠️⚠️
**Locations:**
- src/core/optimizers.py: `class SGDMomentum` (no underscore)
- configs/nn_tuning.json: `"SGD_Momentum"` (with underscore)
- run_all_kaggle.py: Uses `"SGD_Momentum"` consistently
**Fix:** Standardize to `SGD_Momentum` everywhere

### Issue #2: CIFAR10 Naming Consistency
**Locations:**
- Code uses: `CIFAR10` (no hyphen)
- Docs previously used: `CIFAR-10` (with hyphen)
- Directory: `cifar-10-batches-py/` (hyphenated - official dataset name, keep as-is)
**Fix:** Use `CIFAR10` consistently in code and docs

### Issue #3: Result File Naming Pattern Missing
**Problem:** No centralized filename formatter
**Impact:** Scripts expect `NN_*` pattern but files don't match
**Fix:** Create `format_result_filename()` helper

[... 9 more naming issues documented ...]

**Total Naming Fixes Needed:** 12 issues across 30+ files

---

## AGENT 2: MULTI-SEED SUPPORT (2 Fixes Required)

### ✅ EXCELLENT: 27/30 Experiments Have Full Multi-Seed Support

**Working Perfectly:**
- MNIST experiments (SimpleMLP, +BN, ResNet18, ViT)
- CIFAR-10 experiments  
- NLP/Transformer experiments
- Medical segmentation
- 2D optimization
- All ablation studies (except 2)

**Each Has:**
- ✅ `for seed in seeds:` loop
- ✅ `set_seed(seed)` calls
- ✅ Seed-specific result files
- ✅ Proper isolation

### ❌ NEEDS FIX:

**1. Batch Size Ablation (line 1939)**
- Missing `seeds` parameter
- Fix: Add seed loop

**2. Scheduler Ablation (line 2234)**
- Hardcoded `seed=42`
- Fix: Use `seeds` from args

---

## AGENT 3: CROSS-FILE MISMATCHES (5 Critical)

### Mismatch #1: Config vs Class Names
- benchmark_hyperparameters.json uses `"SGDMomentum"`
- nn_tuning.json uses `"SGD_Momentum"`
- Factory expects normalized names
**Fix:** Standardize all configs to `"SGD_Momentum"`

### Mismatch #2: Schema vs Actual Optimizer Names
- Schema enum has lowercase: `'sgd_momentum'`
- Configs use PascalCase: `"SGD_Momentum"`
**Fix:** Update schema to match actual usage

[... 3 more mismatches ...]

---

## AGENT 4: HARDCODED VALUES (150+ Issues)

### Critical Category: Hardcoded Seeds (15 instances)
**Locations:**
- src/utils/transformed_subset.py:13 - `seed=42`
- src/experiments/beta_sensitivity_training.py:114 - `seed=42`
- src/experiments/initialization_ablation.py:360 - `seed=42`
- [... 12 more ...]

**Impact:** Breaks multi-seed experiments, not reproducible
**Fix:** Use parameter or config value

### Critical Category: Hardcoded Learning Rates (20+ instances)
**Locations:**
- src/experiments/stochastic_2d_integrity_fix.py:114-131
- src/experiments/saddle_point_escape_experiment.py:69-71
- [... 18 more ...]

**Impact:** Cannot run fair comparisons without code edits
**Fix:** Load from config or use fair defaults

### High Priority: Dataset Normalization (50+ duplicates)
**Code:** `(0.1307,)` and `(0.3081,)` repeated 50+ times
**Fix:** Create constants in `src/utils/constants.py`

### High Priority: Hardcoded Batch Sizes (30+ instances)
**Code:** `batch_size=128` scattered everywhere
**Fix:** Use config consistently

[... 110 more hardcoded values documented ...]

---

## AGENT 5: AGGREGATION MISSING ⚠️⚠️⚠️ PUBLICATION BLOCKER

### CRITICAL ISSUE: Results Never Aggregated Across Seeds

**Current Behavior:**
- Each seed creates separate CSV: `*_seed42.csv`, `*_seed123.csv`, etc.
- ❌ NO aggregation step after seeds complete
- ❌ NO mean±std summary files created
- ❌ Users cannot easily see "AdamW: 92.3±0.4%"

**Evidence:**
- `aggregate_results()` function EXISTS in `src/experiments/run_multi_seed.py:66-124`
- ❌ NEVER CALLED from `run_all_kaggle.py` main experiments
- Scripts re-aggregate manually every time (fragile)

**What's Missing:**
```python
# After all seeds complete in run_mnist_experiment:
from src.experiments.run_multi_seed import aggregate_results
agg = aggregate_results(seed_csvs, 'test_acc')
# Save: results/MNIST_SimpleMLP_AdamW_aggregated.csv
```

**Impact:** **BLOCKS PUBLICATION** - Cannot report "Method X: 92.3±0.4%" without manual aggregation

**Fix Required:** Add aggregation step to ALL experiment functions

---

## AGENT 6: DOCS CLEANUP (70+ Files Need Organization)

### Current State: 70+ Markdown Files in /docs

**Categories:**
- 15 Audit Reports (many duplicate/superseded)
- 12 Implementation Summaries
- 8 Guides
- 6 References
- 29 Miscellaneous

**Issues:**
- 20+ outdated files (issues already fixed)
- 12+ duplicate audits (same content, different names)
- No organization (flat directory)
- Inconsistent naming (CAPS vs snake_case)

**Recommended Cleanup:**
1. **Remove:** 20 outdated/superseded files
2. **Merge:** 12 duplicate audits → 3 master reports
3. **Organize:** Create subdirectories (audits/, guides/, reference/, implementation/)
4. **Rename:** Standardize to `CATEGORY_descriptor.md`
5. **Index:** Create `docs/README.md` with navigation

**Result:** 70 files → 35 essential files, properly organized

---

## IMPLEMENTATION PRIORITY

### Phase 1: Critical Blockers (Implement NOW)
1. ⚠️⚠️⚠️ **Add seed aggregation to all experiments** (PUBLICATION BLOCKER)
2. ⚠️⚠️ **Fix hardcoded seeds breaking multi-seed experiments** (15 locations)
3. ⚠️⚠️ **Standardize optimizer naming** (SGD_Momentum vs SGDMomentum)
4. ⚠️ **Add seeds parameter to batch/scheduler ablations**

### Phase 2: High Priority (This Week)
5. **Create centralized result filename formatter**
6. **Fix config schema mismatches**
7. **Move normalization constants to constants.py** (50+ duplicates)
8. **Ensure CIFAR10 naming consistency across all files**

### Phase 3: Medium Priority (Next Week)
9. **Fix hardcoded learning rates** (20+ locations)
10. **Fix hardcoded batch sizes** (30+ locations)
11. **Create optimizer name constants/enum**
12. **Cleanup docs directory**

### Phase 4: Low Priority (Future)
13. Fix remaining hardcoded values
14. Add validation for naming consistency
15. Create migration guide for config updates

---

## FILES REQUIRING CHANGES

**Critical Fixes (25 files):**
- run_all_kaggle.py (add aggregation + fix 2 ablations)
- src/experiments/*.py (15 files with hardcoded seeds)
- configs/*.json (3 files with naming mismatches)
- src/utils/constants.py (create with normalization constants)
- src/core/optimizer_registry.py (add name enum)
- [... 5 more ...]

**Documentation (70 files):**
- docs/*.md (cleanup and reorganize all)

---

## VALIDATION PLAN

After implementing fixes:

```bash
# Validate naming consistency
python scripts/validate_naming_consistency.py

# Test multi-seed aggregation
python run_all_kaggle.py --experiments mnist --seeds 42,123,456
ls results/experiments/mnist/*_aggregated.csv  # Should exist

# Verify no hardcoded seeds
grep -r "seed=42" src/ --exclude-dir=tests  # Should be empty

# Check config schema
python scripts/validate_config_schema.py
```

---

## ESTIMATED EFFORT

- **Phase 1 (Critical):** 12-16 hours
- **Phase 2 (High):** 8-10 hours
- **Phase 3 (Medium):** 6-8 hours
- **Phase 4 (Low):** 4-6 hours
- **Total:** 30-40 hours

---

**Status:** Ready for implementation - all issues documented with specific fixes
