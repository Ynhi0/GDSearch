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


def check_notebook(path: Path, fix: bool = False):
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
            # Detect trailing backslash line-continuation (common source of SyntaxError)
            if line.rstrip().endswith('\\'):
                logging.warning("Notebook %s: code cell contains trailing backslash at line %d", path, i+1)
                changed = True

        # Detect literal '\\n' sequences in code cells which often indicate escaped newlines
        # that should be real newlines. Optionally fix when --fix is provided.
        for idx, line in enumerate(lines):
            if '\\n' in line:
                logging.warning("Notebook %s: code cell contains literal '\\n' at line %d; this can cause SyntaxError.", path, idx+1)
                changed = True
                if fix:
                    # Replace literal '\\n' with actual newlines and re-split the source
                    new_src = ''.join(lines).replace('\\n', '\n')
                    if isinstance(src, list):
                        cell['source'] = new_src.splitlines(True)
                    else:
                        cell['source'] = new_src
                    # Update local lines to reflect the change for further checks
                    if isinstance(cell['source'], list):
                        lines = cell['source']
                    else:
                        lines = cell['source'].splitlines(True)

    # If fix requested and changes made, write back the notebook
    if fix and changed:
        try:
            with path.open('w', encoding='utf-8') as f:
                json.dump(nb, f, ensure_ascii=False, indent=1)
        except Exception as e:
            logging.exception('Failed to write fixed notebook %s: %s', path, e)
            raise

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
            if check_notebook(nb, fix=args.fix):
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
