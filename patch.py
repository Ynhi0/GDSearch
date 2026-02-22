import re
import pandas as pd
import numpy as np

with open('src/visualization/ablation_plots.py', 'r', encoding='utf-8') as f:
    text = f.read()

def repl(m):
    return '''        
        improvement_str = "N/A"
        try:
            if not pd.isna(baseline_value):
                improvement_str = f"{float(mean) - float(baseline_value):+.2f}"
        except: pass
        std_str = f" ± {std:.2f}" if not pd.isna(std) else ""

        row = [
            config,
            f"{mean:.2f}{std_str}",
            improvement_str,
            f"{int(count)} seeds"
        ]'''

# Match robustly without worrying about ± encoding
text = re.sub(
    r'        row = \[\s*config,\s*f"\{mean:\.2f\}.{1,5}\{std:\.2f\}",\s*f"\{improvement:\+\.2f\}",\s*f"\{int\(count\)\} seeds"\s*]',
    repl, text, flags=re.MULTILINE
)

with open('src/visualization/ablation_plots.py', 'w', encoding='utf-8') as f:
    f.write(text)
print('Patch applied')
