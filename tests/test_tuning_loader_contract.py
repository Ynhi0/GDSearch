"""
Tests to enforce tuning safety contract:
- Any function whose name contains 'tune' or 'tuning' MUST NOT accept a parameter named 'test_loader' unless
  its docstring explicitly documents that the parameter is validation data (contains 'validation' or 'val').
This helps prevent accidental test-set leakage during hyperparameter tuning.
"""

import ast
import inspect
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _iter_py_files():
    for p in REPO_ROOT.rglob('*.py'):
        # skip tests themselves
        if 'tests' in p.parts:
            continue
        yield p


def _get_functions_from_file(path: Path):
    # Read file safely (skip files that are not UTF-8 to avoid decode errors)
    try:
        src = path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        # Skip non-UTF8 files (e.g., binary or other encodings that are not code sources we care about)
        return
    try:
        tree = ast.parse(src)
    except SyntaxError:
        # Skip files that fail to parse as Python - may contain non-executable snippets or be incompatible
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            yield node, path


def test_tuning_functions_do_not_accept_test_loader_without_doc():
    offenders = []

    for file_path in _iter_py_files():
        for node, path in _get_functions_from_file(file_path):
            name = node.name
            if 'tune' in name.lower() or 'tuning' in name.lower():
                # Check parameters
                params = [arg.arg for arg in node.args.args]
                if 'test_loader' in params:
                    # Check docstring for validation wording
                    doc = ast.get_docstring(node) or ''
                    doc_l = doc.lower()
                    if not ('validation' in doc_l or 'val_' in doc_l or 'val ' in doc_l or 'not test' in doc_l):
                        offenders.append((path, name))

    assert not offenders, (
        "Error: Tuning functions must not accept 'test_loader' without documenting that it is actually validation data. "
        "Offenders: " + ', '.join(f"{p}:{n}" for p, n in offenders)
    )
