#!/usr/bin/env python3
"""
Pre-commit hook to detect top-level 'seed' entries in JSON config files and fail the commit.
This enforces using 'seeds' (array of ints) for reproducible multi-seed experiments.
"""
import sys
import json
from pathlib import Path


def check_file(path: Path) -> list[str]:
    issues = []
    if not path.exists():
        return issues
    if path.suffix.lower() != '.json':
        return issues
    try:
        j = json.loads(path.read_text(encoding='utf-8'))
    except Exception as e:
        return [f"{path}: Could not parse JSON: {e}"]

    if isinstance(j, dict) and 'seed' in j and 'seeds' not in j:
        issues.append(f"{path}: top-level 'seed' found; use 'seeds' (array) instead")
    return issues


def main():
    paths = [Path(p) for p in sys.argv[1:]]
    # If no paths provided, check all configs
    if not paths:
        paths = list(Path('configs').glob('*.json'))

    all_issues = []
    for p in paths:
        all_issues.extend(check_file(p))

    if all_issues:
        print("⚠️  Config seeding issues detected:")
        for issue in all_issues:
            print("  ", issue)
        print()
        print("Fix: Replace top-level 'seed': <int> with 'seeds': [42,123,456] or similar multi-seed list.")
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())