"""
Simple import scan: detect top-level imports from `src/` and ensure they can be imported in the current environment.
Exits with non-zero code and prints missing imports.
"""
import re
import sys
from pathlib import Path
import importlib

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
REQS = ROOT / 'requirements.txt'

IMPORT_RE = re.compile(r'^\s*(?:from\s+([A-Za-z0-9_\.]+)\s+import|import\s+([A-Za-z0-9_\.]+))')

# Collect required packages from requirements.txt (simple parse)
req_names = set()
if REQS.exists():
    for line in REQS.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        # take portion before any comparison operators
        name = re.split(r'[=<>!]', line)[0].strip().lower()
        # handle extras like package[extra]
        name = re.split(r'\[', name)[0]
        req_names.add(name)

# Provide common alias mappings where import root differs from PyPI package name
ALIAS_MAP = {
    'sklearn': 'scikit-learn',
    'PIL': 'pillow',
    'cv2': 'opencv-python',
    'yaml': 'pyyaml',
    'skimage': 'scikit-image',
    'torchvision': 'torchvision',
    'torch': 'torch',
    'tensorflow': 'tensorflow',
}

imports = set()
for py in SRC.rglob('*.py'):
    text = py.read_text(encoding='utf-8')
    for m in IMPORT_RE.finditer(text):
        mod = m.group(1) or m.group(2)
        if not mod:
            continue
        root = mod.split('.')[0]
        # ignore local package imports
        if root == 'src':
            continue
        imports.add(root)

missing = []
for mod in sorted(imports):
    mod_lower = mod.lower()
    # Map common alias to expected PyPI package name
    mapped = ALIAS_MAP.get(mod, None)
    candidates = {mod_lower}
    if mapped:
        candidates.add(mapped.lower())
    # If requirements includes any candidate, consider it covered
    if any(c in req_names for c in candidates):
        continue
    try:
        importlib.import_module(mod)
    except Exception:
        missing.append(mod)

if missing:
    print("Missing imports detected (ensure packages are in requirements.txt and installed):")
    for m in missing:
        print("  -", m)
    sys.exit(2)
else:
    print("Import scan passed: all discovered imports are importable in the current environment or listed in requirements.txt")
    sys.exit(0)
