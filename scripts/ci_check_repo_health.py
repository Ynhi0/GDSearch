#!/usr/bin/env python3
"""CI checks for repo health: no bare "except:", no "except Exception.*: pass", and no hardcoded Windows user paths."""

import re
import sys
from pathlib import Path

repo = Path(__file__).parent.parent
patterns = [
    (re.compile(r'^\s*except\s*:\s*$', re.M), "bare except"),
    (re.compile(r'except\s+Exception\s+[^:]*:\s*pass', re.M), "silent except Exception as e: pass"),
    (re.compile(r'c:\\Users\\', re.I), "hardcoded Windows user path (c:\\Users\\)")
]

issues = []
for p, name in patterns:
    for f in repo.rglob('*.py'):
        try:
            txt = f.read_text()
        except Exception:
            continue
        if p.search(txt):
            issues.append((str(f.relative_to(repo)), name))

if issues:
    print("Found repo health issues:")
    for f, name in issues:
        print(f" - {f}: {name}")
    sys.exit(2)

print("Repo health checks passed.")
