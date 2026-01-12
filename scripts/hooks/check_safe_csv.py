#!/usr/bin/env python3
"""
Pre-commit hook to check for unsafe CSV write patterns.

Detects usage of df.to_csv() without corresponding safe_to_csv or mkdir calls.
"""

import sys
import re
from pathlib import Path


def check_file(filepath: Path) -> list[str]:
    """Check a single file for unsafe CSV patterns."""
    issues = []

    try:
        content = filepath.read_text(encoding='utf-8')
    except Exception as e:
        return [f"{filepath}: Could not read file: {e}"]

    lines = content.split('\n')

    # Check for df.to_csv without safe_to_csv import
    has_unsafe_to_csv = False
    has_safe_import = 'from src.utils.file_safety import' in content and 'safe_to_csv' in content
    has_mkdir = '.mkdir(' in content and 'exist_ok=True' in content

    for i, line in enumerate(lines, 1):
        # Skip comments
        if line.strip().startswith('#'):
            continue

        # Detect .to_csv usage
        if '.to_csv(' in line and 'safe_to_csv' not in line:
            # Check if this file has safety measures
            if not (has_safe_import or has_mkdir):
                has_unsafe_to_csv = True
                issues.append(
                    f"{filepath}:{i}: Unsafe CSV write detected. "
                    f"Use safe_to_csv() or add mkdir(parents=True, exist_ok=True)"
                )

    return issues


def main():
    """Check all staged Python files."""
    files_to_check = [Path(f) for f in sys.argv[1:] if f.endswith('.py')]

    all_issues = []
    for filepath in files_to_check:
        issues = check_file(filepath)
        all_issues.extend(issues)

    if all_issues:
        print("❌ Unsafe CSV write patterns detected:")
        print()
        for issue in all_issues:
            print(f"  {issue}")
        print()
        print("Fix: Import and use safe_to_csv() from src.utils.file_safety")
        print("  from src.utils.file_safety import safe_to_csv")
        print("  safe_to_csv(df, 'path/to/output.csv', index=False)")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
