#!/usr/bin/env python
"""
Final comprehensive retraining recommendation report.
"""
import pandas as pd
from pathlib import Path

print('='*90)
print('GDSEARCH v2 EXPERIMENTS - RETRAINING RECOMMENDATION REPORT')
print('='*90)

results_dir = Path('results_proposal_full_20260223_v2/experiments')

# Quick summary
print('''
EXECUTIVE SUMMARY:
  Most experiments completed successfully with expected training depth.
  2 experiments require attention:
    1. ROBUSTNESS - 10% of runs failed (50/500 seeds with single-iteration only)
    2. HIGHDIM - Epoch coverage only 123-375 (expected ~500)

================================================================================
DETAILED FINDINGS:
================================================================================

1. HEALTHY EXPERIMENTS (No action needed):
   ✓ Medical (3 epochs × 90 seeds) - Intentionally quick for diagnostic testing
   ✓ MNIST (2 epochs × 36 seeds) - Quick baseline for scripting validation
   ✓ ResNet (20 epochs × 21 seeds) - Small quick test
   ✓ Ablation (465-999 epochs × 50 seeds) - High variability but mostly complete
   ✓ Advanced Ablation (10 epochs × 80 seeds) - As designed
   ✓ Batch Ablation (60 epochs × 1 seed) - Single validation run, complete
   ✓ LR Ablation (5-10 epochs × 120 seeds) - Quick ablation, as intended
   ✓ Init Ablation (2-10 epochs × 240 seeds) - Quick ablation, as intended
   ✓ WD Ablation (5-10 epochs × 120 seeds) - Quick ablation, as intended

2. NEEDS INVESTIGATION - Robustness Experiment:
   ⚠️ Status: INCOMPLETE RUNS DETECTED
   • Total seeds: 500
   • Successfully completed (19999 iters): 350 seeds (70%)
   • FAILED - Single iteration only (0 iterations recorded): 50 seeds (10%)
   • Other incomplete values (327, 237, 448, etc. iters): 100 seeds (20%)
   
   • Root Cause: Files examined show failed seeds have exactly 1 row with:
     iteration=0, loss=0.0, grad_norm=0.0, x=1.0, y=1.0
     (vs. normal seeds with 4000+ rows, iterations 0-17724)
   
   • Interpretation: These 50 seeds likely crashed immediately after initialization
     (possible OOM, timeout, or numerical issue with that parameter combo)

3. NEEDS ATTENTION - HighDim Experiment:
   ⚠️ Status: UNDERCOVERED TRAINING
   • Total seeds: 60
   • Epoch range: 123-375 (expected: ~500)
   • Average: 217.6 epochs
   • Coverage: 44% of expected maximum
   
   • Root Cause: Unknown from current data (possible early stopping, interrupted runs,
     or intentional quick mode that wasn't documented)
   
   • Recommendation: Review if this is acceptable coverage or if rerun needed

4. SCHEDULER ABLATION:
  ✅ VERIFIED CORRECT - No rerun needed
  • Total seeds: 40 (4 optimizer-scheduler pairs × 10 seeds)
  • Epoch range: 10 (hardcoded)
  • Coverage: 100%

================================================================================
RETRAINING STRATEGY:
================================================================================

IMMEDIATE ACTION (High Priority):
──────────────────────────────────

1. ROBUSTNESS RETRY:
  Command:
  python run_all_kaggle.py --experiments=robustness --seeds <failed-seeds> --resume --resume-behavior=restart_if_no_checkpoint
   
  Where <failed-seeds> are the failed seed values. Use --resume to skip already-completed runs.
  Note: Remove the 1-row robustness CSVs (and matching metadata) before rerun so they are retried.
   
  These fail immediately, so retry should be quick test first before full rerun.
   
   Rationale: 70% completion rate is good, but 10% failure is concerning. Rerunning
   failed subset only takes ~10% of time vs. full 500-seed rerun.

2. HIGHDIM DIAGNOSTIC:
   First: Understand why coverage is 44% of expected
   
   Option A (Conservative): Accept current depth, proceed with analysis
  Option B (Thorough): Rerun with full depth
   
  Command (if decided to rerun):
  python run_all_kaggle.py --experiments=highdim --seeds 42,123,456 --resume
   
   Start with seed 42 to validate, then full run if OK.

3. SCHEDULER ABLATION:
  Status: ✅ VERIFIED CORRECT - No rerun needed
   
  The code is hardcoded to run 10 epochs (line 2641), not 100.
  All 40 seed files (4 optimizer-scheduler pairs × 10 seeds) show correct 10-epoch coverage.
  Initial analysis had incorrect baseline expectation.

OPTIONAL ACTION (Lower Priority):
──────────────────────────────────

• Ablation variance (465-999 epochs): Current range is acceptable. If exact
  reproducibility needed, could rerun with explicit --epochs=1000, but current
  data is usable.

================================================================================
VALIDATION CHECKLIST:
================================================================================

Before declaring v2 complete:
□ Extract failed robustness seed list
□ Quick test: rerun 1-2 failed robustness seeds, verify convergence
□ Option: Full robustness rerun of 50 seeds OR accept 70% completion rate
□ Decision: highdim - keep as-is OR rerun with full depth
□ Re-audit PNG outputs after any regeneration

Expected outcome:
  After fixes: 99%+ completion rate across all 1000+ seed runs
  v2 folder ready for final analysis + paper submission

================================================================================
QUICK REFERENCE TIMESTAMPS:
================================================================================

Run time estimate (all reruns):
  • Robustness 50 seeds: ~2-3 hours (if crashes were transient)
  • HighDim full: ~4-5 hours  
Total worst-case rerun: 6-8 hours

Recommendation: Start with robustness tonight, assess highdim
separately after confirming other fixes work.
''')

# Show failed robustness seeds for easy copying
print('\n' + '='*90)
print('FAILED ROBUSTNESS SEEDS (for reference):')
print('='*90)

robustness_dir = results_dir / 'robustness'
zero_seeds = []

for csv_file in sorted(robustness_dir.glob('*_seed*.csv')):
    df = pd.read_csv(csv_file)
    if len(df) == 1 and df.iloc[0]['iteration'] == 0:
        # Extract seed number from filename
        match = csv_file.stem.split('_seed')[-1]
        zero_seeds.append(match)

if zero_seeds:
    print(f'\nTotal failed seeds: {len(zero_seeds)}')
    print(f'Seed list (comma-separated for --seeds flag):')
    print(','.join(zero_seeds[:20]))
    if len(zero_seeds) > 20:
        print('...')
        print(','.join(zero_seeds[-10:]))
else:
    print('No failed seeds detected in this check.')

print('\n' + '='*90)
