#!/usr/bin/env python3
"""
Pre-commit hook to check for incomplete seeding patterns.

Detects usage of np.random.seed() without corresponding set_seed() call.
"""

import sys
import re
from pathlib import Path


def check_file(filepath: Path) -> list[str]:
    """Check a single file for incomplete seeding."""
    issues = []

    try:
        content = filepath.read_text(encoding='utf-8')
    except Exception as e:
        return [f"{filepath}: Could not read file: {e}"]

    lines = content.split('\n')

    # Check for incomplete seeding
    has_np_random_seed = False
    has_set_seed_import = 'from src.core.training_utils import' in content and 'set_seed' in content
    has_set_seed_call = 'set_seed(' in content

    for i, line in enumerate(lines, 1):
        # Skip comments
        if line.strip().startswith('#'):
            continue

        # Detect np.random.seed usage
        if 'np.random.seed(' in line and 'set_seed(' not in line:
            has_np_random_seed = True

            # Allow in tests directory
            if '/tests/' in str(filepath) or '\\tests\\' in str(filepath):
                continue

            # Only flag if no set_seed is used anywhere in file
            if not (has_set_seed_import and has_set_seed_call):
                issues.append(
                    f"{filepath}:{i}: Incomplete seeding detected. "
                    f"Use set_seed() instead of np.random.seed() for reproducibility"
                )
                break  # Only report once per file

    return issues


def main():
    """Check all staged Python files in experiments/analysis."""
    files_to_check = [Path(f) for f in sys.argv[1:] if f.endswith('.py')]

    all_issues = []
    for filepath in files_to_check:
        issues = check_file(filepath)
        all_issues.extend(issues)

    if all_issues:
        print("⚠️  Incomplete seeding patterns detected:")
        print()
        for issue in all_issues:
            print(f"  {issue}")
        print()
        print("Fix: Import and use set_seed() from src.core.training_utils")
        print("  from src.core.training_utils import set_seed")
        print("  set_seed(42)  # Seeds numpy, torch, and random module")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
