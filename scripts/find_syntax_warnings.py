import warnings, ast
from pathlib import Path
root=Path('.').resolve()
files=list(root.rglob('*.py'))
for p in files:
    if 'tests' in p.parts:
        continue
    try:
        src=p.read_text(encoding='utf-8')
    except Exception:
        continue
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        try:
            ast.parse(src)
        except Exception:
            continue
        if w:
            print(p)
            for ww in w:
                print('  ', ww.category.__name__, ww.message)