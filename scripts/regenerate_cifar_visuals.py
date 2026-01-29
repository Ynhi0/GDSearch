"""Regenerate only CIFAR10 visuals using existing CSVs and summaries."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runners.run_all_kaggle import create_experiment_visualizations

results_root = Path(r"C:/Users/MPhuc/Downloads/results/results_full")
csvs = list(results_root.glob('experiments/cifar10/*.csv'))
create_experiment_visualizations('CIFAR10', str(results_root), csvs)
print('Regenerated CIFAR visuals')