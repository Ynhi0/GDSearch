#!/usr/bin/env python3
"""Fix and validate notebooks before execution.

This script runs the notebook validator with fix enabled and then uses nbformat to
validate notebook structure. Run this before executing notebooks with papermill
or nbconvert to avoid SyntaxErrors caused by malformed code cells.
"""
from pathlib import Path
import logging
import json
import nbformat
# Import check_notebook from scripts. Use importlib fallback when running as a script (scripts/ not a package)
try:
    from scripts.validate_notebooks import check_notebook
except Exception:
    import importlib.util
    spec = importlib.util.spec_from_file_location('validate_notebooks', str(Path(__file__).resolve().parent / 'validate_notebooks.py'))
    validate_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validate_mod)
    check_notebook = validate_mod.check_notebook

logging.basicConfig(level=logging.INFO)


def validate_notebook_structure(path: Path) -> bool:
    try:
        nb = nbformat.read(str(path), as_version=nbformat.NO_CONVERT)
        # Will raise if invalid
        nbformat.validate(nb)
        return True
    except Exception as e:
        logging.exception('Notebook structure validation failed for %s: %s', path, e)
        return False


def main():
    notebooks = list(Path('notebooks').rglob('*.ipynb'))
    if not notebooks:
        logging.info('No notebooks found.')
        return 0

    failures = 0
    for nb in notebooks:
        logging.info('Fixing & validating %s', nb)
        try:
            changed = check_notebook(nb, fix=True)
            if changed:
                logging.info('Fixed issues in %s', nb)
            ok = validate_notebook_structure(nb)
            if not ok:
                failures += 1
        except Exception as e:
            logging.exception('Error processing %s: %s', nb, e)
            failures += 1

    if failures:
        logging.error('Notebook validation failed for %d notebook(s).', failures)
        return 2

    logging.info('All notebooks fixed and validated successfully.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())