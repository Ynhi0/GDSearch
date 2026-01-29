from pathlib import Path
import pandas as pd
import sys
sys.path.insert(0, r'C:/Users/MPhuc/Desktop/GDSearch')
from src.utils.filename import parse_opt_seed_from_stem
p=Path(r'C:/Users/MPhuc/Downloads/results/results_full/experiments/cifar10')
files=sorted([f for f in p.glob('*.csv') if 'summary' not in f.name.lower() and 'results' not in f.name.lower()])
for f in files:
    stem=f.stem
    opt, seed = parse_opt_seed_from_stem(stem)
    try:
        df=pd.read_csv(f)
        # find final acc
        acc=None
        for c in ['final_test_acc','test_acc','test_accuracy','val_acc']:
            if c in df.columns and df[c].dropna().size>0:
                acc=df[c].dropna().iloc[-1]
                break
    except Exception as e:
        acc='ERROR'
    print(f.name,'->', 'opt=',opt,'seed=',seed,'last_acc=',acc)
