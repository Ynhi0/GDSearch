"""Inspect plotly JSON embedded in an interactive HTML file and summarize metrics present."""
import json
from pathlib import Path
p = Path(r'C:/Users/MPhuc/Downloads/results/results_full/visualizations/interactive/cifar10_interactive_comparison.html')
s = p.read_text(encoding='utf-8')
start_token = 'Plotly.newPlot('
idx = s.find(start_token)
if idx == -1:
    print('Plotly.newPlot not found')
    raise SystemExit(1)
# Use regex to find the data array argument robustly: Plotly.newPlot('id', [data], layout, config)
import re
m = re.search(r"Plotly\.newPlot\(\s*['\"][^'\"]+['\"]\s*,\s*(\[)", s)
if not m:
    # fallback: try to find the first '[' after the call start
    open_br = s.find('[', idx)
    if open_br == -1:
        print('Could not locate data array start via regex or fallback')
        raise SystemExit(1)
else:
    open_br = m.start(1)
# find matching closing bracket while handling quotes
level = 0
in_single = False
in_double = False
escape = False
end = None
for i, ch in enumerate(s[open_br:], start=open_br):
    if escape:
        escape = False
        continue
    if ch == '\\':
        escape = True
        continue
    if ch == '"' and not in_single:
        in_double = not in_double
        continue
    if ch == "'" and not in_double:
        in_single = not in_single
        continue
    if in_single or in_double:
        continue
    if ch == '[':
        level += 1
    elif ch == ']':
        level -= 1
        if level == 0:
            end = i
            break
if end is None:
    print('Could not locate matching closing bracket for data array')
    raise SystemExit(1)
data_json = s[open_br:end+1]
try:
    # debug: show snippet to inspect why JSON fails
    print('DATA SNIPPET:', data_json[:200])
    data = json.loads(data_json)
except Exception as e:
    print('JSON parse failed:', e)
    print('Full data snippet (first 500 chars):')
    print(data_json[:500])
    raise
# Summarize traces per subplot (yaxis mapping) and flag test-related traces
summary = {}
for t in data:
    name = t.get('name')
    y = t.get('y', [])
    # count non-null numbers
    nonnull = sum(1 for v in y if v is not None)
    is_test = False
    if name:
        low = str(name).lower()
        if 'test' in low or 'acc' in low or 'loss' in low:
            is_test = True
    summary.setdefault(t.get('yaxis','y'), []).append({'name': name, 'nonnull': nonnull, 'len': len(y), 'is_test': is_test})

from pprint import pprint
pprint(summary)
print('\nTotal traces:', len(data))
