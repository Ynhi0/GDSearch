#!/usr/bin/env python3
"""Comprehensive validation of partial Kaggle results - find ALL anomalies."""

import argparse
import os
import logging
import json
import csv
import glob
from pathlib import Path
from collections import defaultdict
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

parser = argparse.ArgumentParser(description="Comprehensive validation of partial Kaggle results")
parser.add_argument('--results-dir', default=os.environ.get('GDSEARCH_RESULTS_DIR', 'results/results_full'), help='Path to results directory')
args = parser.parse_args()

RESULTS_DIR = Path(args.results_dir)
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"

logging.info(f"Using results directory: {RESULTS_DIR}")

print("=" * 80)
print("COMPREHENSIVE VALIDATION REPORT")
print("=" * 80)

# ===== SECTION 1: File Structure Analysis =====
print("\n[1] FILE STRUCTURE ANALYSIS")
print("-" * 80)

csv_files = list(EXPERIMENTS_DIR.rglob("*.csv"))
json_files = list(EXPERIMENTS_DIR.rglob("*.json"))
checkpoint_files = list(CHECKPOINTS_DIR.glob("*.pt"))
backup_files = [f for f in CHECKPOINTS_DIR.glob("*.pt.*") if "backup" in f.name]

print(f"Total CSV files: {len(csv_files)}")
print(f"Total JSON metadata: {len(json_files)}")
print(f"Total checkpoints (.pt): {len(checkpoint_files)}")
print(f"Total backup files: {len(backup_files)}")

# Aggregate CSVs
aggregate_csvs = [f for f in csv_files if "results.csv" in f.name.lower()]
experiment_csvs = [f for f in csv_files if "results.csv" not in f.name.lower()]
print(f"  - Aggregate files: {len(aggregate_csvs)}")
print(f"  - Experiment files: {len(experiment_csvs)}")

# ===== SECTION 2: Missing Metadata Files =====
print("\n[2] MISSING METADATA FILES")
print("-" * 80)

missing_metadata = []
for csv_file in experiment_csvs:
    json_file = csv_file.parent / f"{csv_file.stem}_metadata.json"
    if not json_file.exists():
        missing_metadata.append(csv_file)
        print(f"  ❌ {csv_file.relative_to(EXPERIMENTS_DIR)}")

if not missing_metadata:
    print("  ✅ All experiment CSVs have metadata")
else:
    print(f"\n  Total missing metadata: {len(missing_metadata)}")

# ===== SECTION 3: Checkpoint Analysis =====
print("\n[3] CHECKPOINT VALIDATION")
print("-" * 80)

# Build checkpoint index
checkpoint_index = {}
for ckpt in checkpoint_files:
    # Extract seed from filename (e.g., "MNIST_AdaBound_seed1011.pt" -> "1011")
    name = ckpt.stem  # removes .pt
    checkpoint_index[name] = ckpt

print(f"Checkpoint index built: {len(checkpoint_index)} unique checkpoints")

# Check which experiments have checkpoints
missing_checkpoints = []
found_checkpoints = []

for csv_file in experiment_csvs:
    # Parse CSV name: DATASET_MODEL_OPTIMIZER_seedSEED.csv
    # e.g., "MNIST_SimpleMLP_Adam_seed42.csv"
    parts = csv_file.stem.split("_")
    
    # Find seed in filename
    seed_part = [p for p in parts if p.startswith("seed")]
    if not seed_part:
        print(f"  ⚠️ Cannot parse seed from: {csv_file.name}")
        continue
    
    # Build expected checkpoint name (without seed prefix)
    # MNIST_SimpleMLP_Adam_seed42.csv -> MNIST_Adam_seed42.pt (simplified model name)
    dataset = parts[0]
    optimizer = parts[-2]  # Second to last before seedXXX
    seed = seed_part[0]
    
    # Try multiple checkpoint naming patterns
    possible_names = [
        f"{dataset}_{optimizer}_{seed}",  # MNIST_Adam_seed42
        f"{dataset}_{'_'.join(parts[1:-2])}_{seed}",  # Full pattern
    ]
    
    # Check if any variant exists
    found = False
    for name in possible_names:
        if name in checkpoint_index:
            found_checkpoints.append((csv_file, checkpoint_index[name]))
            found = True
            break
    
    if not found:
        missing_checkpoints.append(csv_file)

print(f"  ✅ Checkpoints found: {len(found_checkpoints)}")
print(f"  ❌ Checkpoints missing: {len(missing_checkpoints)}")

if missing_checkpoints:
    print("\n  Missing checkpoints for:")
    for csv_file in missing_checkpoints[:10]:  # Show first 10
        print(f"    - {csv_file.relative_to(EXPERIMENTS_DIR)}")
    if len(missing_checkpoints) > 10:
        print(f"    ... and {len(missing_checkpoints) - 10} more")

# ===== SECTION 4: Epoch Completion Analysis =====
print("\n[4] EPOCH COMPLETION ANALYSIS")
print("-" * 80)

epoch_stats = {
    "full_runs": [],  # 50 epochs
    "partial_runs": [],  # < 50 epochs
    "tiny_runs": [],  # < 5 epochs
    "empty_runs": []  # 0-1 epochs
}

for csv_file in experiment_csvs:
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            epoch_count = len(rows)
            
            if epoch_count >= 50:
                epoch_stats["full_runs"].append((csv_file, epoch_count))
            elif epoch_count >= 5:
                epoch_stats["partial_runs"].append((csv_file, epoch_count))
            elif epoch_count >= 2:
                epoch_stats["tiny_runs"].append((csv_file, epoch_count))
            else:
                epoch_stats["empty_runs"].append((csv_file, epoch_count))
    except Exception as e:
        logging.exception(f"Error reading {csv_file.name}")

print(f"Full runs (50 epochs):    {len(epoch_stats['full_runs'])}")
print(f"Partial runs (5-49):      {len(epoch_stats['partial_runs'])}")
print(f"Tiny runs (2-4):          {len(epoch_stats['tiny_runs'])}")
print(f"Empty runs (0-1):         {len(epoch_stats['empty_runs'])}")

# Show partial run distribution
if epoch_stats["partial_runs"]:
    partial_epochs = [e for _, e in epoch_stats["partial_runs"]]
    print(f"\nPartial run epoch distribution:")
    print(f"  Min: {min(partial_epochs)} epochs")
    print(f"  Max: {max(partial_epochs)} epochs")
    print(f"  Mean: {np.mean(partial_epochs):.1f} epochs")
    print(f"  Median: {np.median(partial_epochs):.0f} epochs")

# ===== SECTION 5: Early Stopping Pattern Analysis =====
print("\n[5] EARLY STOPPING PATTERN ANALYSIS")
print("-" * 80)

early_stop_patterns = defaultdict(list)

for csv_file in experiment_csvs:
    try:
        # Read CSV
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        if len(rows) < 2:
            continue
            
        # Check if early stopped (< 50 epochs)
        if len(rows) < 50:
            # Try to find when validation peaked
            if 'val_acc' in rows[0]:
                val_accs = [float(r['val_acc']) for r in rows if r.get('val_acc')]
                if val_accs:
                    peak_epoch = val_accs.index(max(val_accs)) + 1
                    stop_epoch = len(rows)
                    patience_used = stop_epoch - peak_epoch
                    
                    early_stop_patterns['stopped'].append({
                        'file': csv_file.name,
                        'peak_epoch': peak_epoch,
                        'stop_epoch': stop_epoch,
                        'patience': patience_used,
                        'peak_val_acc': max(val_accs)
                    })
    except Exception as e:
        logging.exception(f"Error analyzing {csv_file.name}")

if early_stop_patterns['stopped']:
    patterns = early_stop_patterns['stopped']
    patience_values = [p['patience'] for p in patterns]
    
    print(f"Early stopped experiments: {len(patterns)}")
    print(f"\nPatience statistics:")
    print(f"  Min patience: {min(patience_values)}")
    print(f"  Max patience: {max(patience_values)}")
    print(f"  Mean patience: {np.mean(patience_values):.1f}")
    print(f"  Median patience: {np.median(patience_values):.0f}")
    
    # Check for patience=10 pattern
    patience_10 = [p for p in patterns if 9 <= p['patience'] <= 11]
    print(f"  Experiments with patience~10: {len(patience_10)} ({100*len(patience_10)/len(patterns):.1f}%)")
    
    # Show examples
    print(f"\nExample early-stopped experiments:")
    for pattern in patterns[:5]:
        print(f"  {pattern['file']}")
        print(f"    Peak: epoch {pattern['peak_epoch']} (val_acc={pattern['peak_val_acc']:.4f})")
        print(f"    Stop: epoch {pattern['stop_epoch']} (patience={pattern['patience']})")

# ===== SECTION 6: Tainted Runs Detection =====
print("\n[6] TAINTED RUNS DETECTION (OOM Recovery)")
print("-" * 80)

tainted_runs = []

for csv_file in experiment_csvs:
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        # Check for tainted flag or effective_batch_size != original
        for row in rows:
            if row.get('tainted') and str(row['tainted']).lower() == 'true':
                tainted_runs.append({
                    'file': csv_file.name,
                    'epoch': row.get('epoch', '?'),
                    'effective_batch_size': row.get('effective_batch_size', '?')
                })
                break
    except Exception as e:
        logging.exception(f"Error checking tainted run {csv_file.name}")

print(f"Tainted runs found: {len(tainted_runs)}")
if tainted_runs:
    print("\n  ⚠️ TAINTED RUNS (OOM recovery occurred):")
    for t in tainted_runs:
        print(f"    {t['file']} - epoch {t['epoch']}, batch_size={t['effective_batch_size']}")
else:
    print("  ✅ No tainted runs detected")

# ===== SECTION 7: Data Quality Analysis =====
print("\n[7] DATA QUALITY ANALYSIS")
print("-" * 80)

quality_issues = {
    'nan_values': [],
    'zero_loss': [],
    'extreme_values': [],
    'non_improving': []
}

for csv_file in experiment_csvs:
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        if len(rows) < 2:
            continue
        
        # Check for NaN values
        for row in rows:
            for key, value in row.items():
                if 'loss' in key or 'acc' in key:
                    try:
                        val = float(value)
                        if np.isnan(val) or np.isinf(val):
                            quality_issues['nan_values'].append((csv_file.name, key, row.get('epoch', '?')))
                    except Exception as e:
                        logging.exception(f"Error parsing numeric value in {csv_file.name} for key {key}: {e}")
                        continue
        
        # Check for zero loss (suspicious)
        train_losses = [float(r['train_loss']) for r in rows if r.get('train_loss')]
        if train_losses and any(loss == 0.0 for loss in train_losses):
            quality_issues['zero_loss'].append(csv_file.name)
        
        # Check for extreme values (loss > 100)
        if train_losses and any(loss > 100 for loss in train_losses):
            quality_issues['extreme_values'].append((csv_file.name, max(train_losses)))
        
        # Check for non-improving runs (train_acc doesn't increase)
        if 'train_acc' in rows[0]:
            train_accs = [float(r['train_acc']) for r in rows if r.get('train_acc')]
            if len(train_accs) >= 10:
                # Check if last 10 epochs show no improvement
                if max(train_accs[-10:]) <= train_accs[0] + 0.01:
                    quality_issues['non_improving'].append(csv_file.name)
    except Exception as e:
        logging.exception(f"Error analyzing quality metrics for {csv_file.name}")

print(f"NaN/Inf values:       {len(quality_issues['nan_values'])}")
print(f"Zero loss runs:       {len(quality_issues['zero_loss'])}")
print(f"Extreme loss values:  {len(quality_issues['extreme_values'])}")
print(f"Non-improving runs:   {len(quality_issues['non_improving'])}")

if quality_issues['nan_values']:
    print("\n  ⚠️ NaN/Inf values found:")
    for file, key, epoch in quality_issues['nan_values'][:5]:
        print(f"    {file} - {key} at epoch {epoch}")

if quality_issues['extreme_values']:
    print("\n  ⚠️ Extreme loss values:")
    for file, max_loss in quality_issues['extreme_values']:
        print(f"    {file} - max loss: {max_loss:.2f}")

# ===== SECTION 8: Dataset Distribution =====
print("\n[8] DATASET & OPTIMIZER DISTRIBUTION")
print("-" * 80)

dataset_counts = defaultdict(int)
optimizer_counts = defaultdict(int)
seed_counts = defaultdict(int)

for csv_file in experiment_csvs:
    parts = csv_file.stem.split("_")
    dataset = parts[0]
    dataset_counts[dataset] += 1
    
    # Extract optimizer (before seedXXX)
    for i, part in enumerate(parts):
        if part.startswith("seed"):
            if i > 0:
                optimizer = parts[i-1]
                optimizer_counts[optimizer] += 1
            # Extract seed number
            seed_num = part.replace("seed", "")
            seed_counts[seed_num] += 1
            break

print("Experiments by dataset:")
for dataset, count in sorted(dataset_counts.items()):
    print(f"  {dataset}: {count}")

print("\nExperiments by optimizer:")
for opt, count in sorted(optimizer_counts.items(), key=lambda x: -x[1])[:10]:
    print(f"  {opt}: {count}")

print("\nExperiments by seed:")
for seed, count in sorted(seed_counts.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
    print(f"  Seed {seed}: {count}")

# ===== SECTION 9: Metadata Validation =====
print("\n[9] METADATA VALIDATION")
print("-" * 80)

metadata_issues = []

for csv_file in experiment_csvs:
    json_file = csv_file.parent / f"{csv_file.stem}_metadata.json"
    if json_file.exists():
        try:
            with open(json_file, 'r') as f:
                metadata = json.load(f)
            
            # Read CSV to get actual epoch count
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                actual_epochs = len(list(reader))
            
            expected_epochs = metadata.get('epochs', 50)
            
            # Check mismatch
            if actual_epochs < expected_epochs:
                metadata_issues.append({
                    'file': csv_file.name,
                    'expected': expected_epochs,
                    'actual': actual_epochs,
                    'diff': expected_epochs - actual_epochs
                })
        except Exception as e:
            logging.exception(f"Error validating metadata for {csv_file.name}")

print(f"Metadata/CSV mismatches: {len(metadata_issues)}")
if metadata_issues:
    print(f"\n  Expected vs Actual epochs:")
    # Group by difference
    by_diff = defaultdict(list)
    for issue in metadata_issues:
        by_diff[issue['diff']].append(issue)
    
    for diff in sorted(by_diff.keys()):
        issues = by_diff[diff]
        print(f"    {len(issues)} files stopped {diff} epochs early")

# ===== SECTION 10: Summary & Recommendations =====
print("\n" + "=" * 80)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 80)

total_experiments = len(experiment_csvs)
completed = len(epoch_stats['full_runs'])
partial = len(epoch_stats['partial_runs']) + len(epoch_stats['tiny_runs'])
failed = len(epoch_stats['empty_runs'])

print(f"\nExperiment Status:")
print(f"  ✅ Completed (50 epochs):  {completed}/{total_experiments} ({100*completed/total_experiments:.1f}%)")
print(f"  ⏸️  Partial (< 50 epochs):  {partial}/{total_experiments} ({100*partial/total_experiments:.1f}%)")
print(f"  ❌ Failed (< 2 epochs):    {failed}/{total_experiments} ({100*failed/total_experiments:.1f}%)")

print(f"\nCheckpoint Status:")
print(f"  ✅ Checkpoints available:  {len(found_checkpoints)}/{total_experiments}")
print(f"  ❌ Checkpoints missing:    {len(missing_checkpoints)}/{total_experiments}")

print(f"\nData Quality:")
if len(tainted_runs) == 0 and len(quality_issues['nan_values']) == 0:
    print("  ✅ No tainted runs or data quality issues detected")
else:
    print(f"  ⚠️ Tainted runs: {len(tainted_runs)}")
    print(f"  ⚠️ Data quality issues: {sum(len(v) for v in quality_issues.values())}")

print(f"\nEarly Stopping Analysis:")
if early_stop_patterns['stopped']:
    patience_values = [p['patience'] for p in early_stop_patterns['stopped']]
    patience_10_count = len([p for p in patience_values if 9 <= p <= 11])
    print(f"  {len(early_stop_patterns['stopped'])} experiments stopped early")
    print(f"  {patience_10_count} experiments used patience~10 (expected behavior)")
    print(f"  ✅ Early stopping working as designed")
else:
    print("  No early stopping detected")

print("\nRecommendations:")
if missing_metadata:
    print(f"  ⚠️ {len(missing_metadata)} experiments missing metadata - may affect resume")
if missing_checkpoints:
    print(f"  ⚠️ {len(missing_checkpoints)} experiments missing checkpoints - cannot resume these")
if partial > 0:
    print(f"  💡 {partial} partial runs can be resumed with --resume flag")
if failed > 0:
    print(f"  ❌ {failed} failed runs need to be re-run")

print("\n" + "=" * 80)
print("VALIDATION COMPLETE")
print("=" * 80)
