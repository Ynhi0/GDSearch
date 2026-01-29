from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.metric_normalization import to_percent
import pandas as pd

p = Path(r'C:/Users/MPhuc/Downloads/results/results_full/experiments/cifar10/NN_ResNet18_CIFAR10_Adam_lr0.001_seed42.csv')
d = pd.read_csv(p)
print('last test_accuracy raw:', d['test_accuracy'].dropna().iloc[-1])
print('normalized %:', to_percent(d['test_accuracy'].dropna().iloc[-1]))
print('\nSummary (per-seed) head:')
print(pd.read_csv(Path(r'C:/Users/MPhuc/Downloads/results/results_full/experiments/cifar10/CIFAR10_summary_per_seed.csv')).head(10))
print('\nSummary (means):')
print(pd.read_csv(Path(r'C:/Users/MPhuc/Downloads/results/results_full/experiments/cifar10/CIFAR10_summary.csv')))
