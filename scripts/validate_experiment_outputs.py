"""Validate experiment CSV outputs and emit a small JSON report.

Usage: python scripts/validate_experiment_outputs.py <results_dir> [--report report.json]

Exits with non-zero code if issues are found.
"""
import json
import sys
from pathlib import Path
import pandas as pd


def validate_results_dir(results_dir: Path):
    issues = {'empty_files': [], 'missing_epoch': [], 'missing_final_metrics': []}

    metric_keywords = ['test_acc', 'test_accuracy', 'final_test_acc', 'final_test_accuracy', 'test_dice', 'final_test_dice']

    for csv_file in results_dir.glob('**/*.csv'):
        try:
            # Fast check for empty
            if csv_file.stat().st_size == 0:
                issues['empty_files'].append(str(csv_file))
                continue

            df = pd.read_csv(csv_file)
            if df.shape[0] == 0:
                issues['empty_files'].append(str(csv_file))
                continue

            if 'epoch' not in df.columns:
                issues['missing_epoch'].append(str(csv_file))

            # Check final metrics existence
            found_metric = False
            for k in metric_keywords:
                if k in df.columns:
                    found_metric = True
                    break
            if not found_metric:
                issues['missing_final_metrics'].append(str(csv_file))

        except Exception as e:
            issues['empty_files'].append(str(csv_file))
            continue

    return issues


def main(argv):
    if len(argv) < 2:
        print("Usage: validate_experiment_outputs.py <results_dir> [--report report.json]")
        return 2

    results_dir = Path(argv[1])
    report_path = None
    if '--report' in argv:
        idx = argv.index('--report')
        if idx + 1 < len(argv):
            report_path = Path(argv[idx+1])

    issues = validate_results_dir(results_dir)

    if report_path:
        report_path.write_text(json.dumps(issues, indent=2))

    any_issues = any(len(v) > 0 for v in issues.values())
    if any_issues:
        print("Validation found issues. See report for details.")
        return 1

    print("Validation OK - no issues found")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
