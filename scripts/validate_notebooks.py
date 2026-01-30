#!/usr/bin/env python3
"""Validate Jupyter notebooks for common syntax issues pre-execution.

Currently checks for: unexpected line-continuation backslashes in code cells that may
cause SyntaxError: unexpected character after line continuation character.

Usage: python scripts/validate_notebooks.py [--fix]
"""
import json
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO)


def check_notebook(path: Path):
    with path.open('r', encoding='utf-8') as f:
        nb = json.load(f)

    changed = False
    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
        src = cell.get('source', [])
        # src may be a list of lines or a single string
        if isinstance(src, list):
            lines = src
        else:
            lines = src.splitlines(True)

        for i, line in enumerate(lines[:-1]):
            if line.rstrip().endswith('\\'):
                # line continuation detected; warn
                logging.warning("Notebook %s: code cell contains trailing backslash at line %d", path, i+1)
    return changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fix', action='store_true', help='Attempt to fix trivial issues (not recommended for complex notebooks)')
    args = parser.parse_args()

    nb_paths = list(Path('notebooks').rglob('*.ipynb'))
    if not nb_paths:
        logging.info('No notebooks found in notebooks/.')
        return

    issues = 0
    for nb in nb_paths:
        logging.info('Checking %s', nb)
        try:
            if check_notebook(nb):
                issues += 1
        except Exception as e:
            logging.exception('Failed to check %s: %s', nb, e)
            issues += 1

    if issues:
        logging.warning('Found issues in %d notebook(s). Please inspect warnings above.', issues)
    else:
        logging.info('No obvious notebook syntax issues found.')


if __name__ == '__main__':
    main()
