import re
from pathlib import Path
import pytest


ALLOWLIST_COMMENT = "broad catch intentional"  # match phrase in comment; allow flexible prefixes


def _find_broad_catches(path: Path):
    bad = []
    pattern = re.compile(r"^\s*except\s+Exception\b")
    lines = path.read_text(encoding='utf-8', errors='ignore').splitlines()
    for i, line in enumerate(lines):
        if pattern.match(line):
            # Check same line or previous two lines for allowlist comment (case-insensitive)
            context = "\n".join(lines[max(0, i-2):i+1]).lower()
            if ALLOWLIST_COMMENT not in context:
                bad.append((i+1, line.strip(), context))
    return bad


def test_no_unannotated_broad_except_in_core_and_entrypoints():
    # Only check core modules and the main entrypoint as specified by CI guard
    base = Path('src/core')
    paths = list(base.rglob('*.py'))
    # Also include the top-level run_all_kaggle.py
    paths.append(Path('run_all_kaggle.py'))

    failures = []
    for p in paths:
        # If the file contains a file-level allowlist comment near the top, skip checking
        top_content = p.read_text(encoding='utf-8', errors='ignore').splitlines()[:80]
        if any(ALLOWLIST_COMMENT in l.lower() for l in top_content):
            continue
        bad = _find_broad_catches(p)
        if bad:
            failures.append((p, bad))

    if failures:
        msgs = []
        for p, items in failures:
            for lineno, line, ctx in items:
                msgs.append(f"{p}:{lineno}: {line}\nContext:\n{ctx}\n")
        pytest.fail("Found unannotated broad 'except Exception:' occurrences:\n" + "\n".join(msgs))
