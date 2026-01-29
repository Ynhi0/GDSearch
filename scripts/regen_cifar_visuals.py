from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runners.run_all_kaggle import create_experiment_visualizations

csvs = list(Path(r'C:/Users/MPhuc/Downloads/results/results_full/experiments/cifar10').glob('*.csv'))
create_experiment_visualizations('CIFAR10', r'C:/Users/MPhuc/Downloads/results/results_full', csvs)
print('Done')