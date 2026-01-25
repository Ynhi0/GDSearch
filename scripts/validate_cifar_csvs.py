#!/usr/bin/env python3
"""Scan experiments/cifar10 CSVs for corrupted or empty files.

Usage:
    python scripts/validate_cifar_csvs.py --path experiments/cifar10

Exits with non-zero code if problems found.
"""
import argparse
from pathlib import Path
import sys
import pandas as pd


def check_csv(p: Path):
    try:
        df = pd.read_csv(p)
        if df.empty:
            return False, 'empty'
        # basic columns check
        if not any(c.lower() in ('epoch', 'train_loss', 'test_acc', 'test_accuracy') for c in df.columns):
            return False, 'missing_metrics'
        return True, ''
    except Exception as e:
        return False, f'corrupt: {e}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='experiments/cifar10')
    args = parser.parse_args()

    p = Path(args.path)
    if not p.exists():
        print(f'Path not found: {p}')
        sys.exit(2)

    problems = []
    for f in sorted(p.glob('*.csv')):
        ok, reason = check_csv(f)
        if not ok:
            problems.append((f, reason))
            print(f'PROBLEM: {f} -> {reason}')

    if problems:
        print(f'Found {len(problems)} problematic files')
        sys.exit(1)
    else:
        print('All CIFAR10 CSVs appear healthy')
        sys.exit(0)

if __name__ == '__main__':
    main()
