import os
import pandas as pd
import numpy as np
from pathlib import Path
import shutil

from scripts.generate_experiment_plots import generate_all_plots


def _make_sample_csv(path: Path, epochs=3):
    df = pd.DataFrame({
        'epoch': list(range(1, epochs+1)),
        'train_loss': np.linspace(1.0, 0.1, epochs),
        'test_accuracy': np.linspace(0.5, 0.8, epochs)
    })
    df.to_csv(path, index=False)


def test_generate_plots_groups_various_filenames(tmp_path):
    results_dir = tmp_path / 'results'
    exp_dir = results_dir / 'experiments' / 'cifar10' / 'experiments' / 'cifar10'
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Create CSVs with various naming styles
    names = [
        'NN_SimpleMLP_CIFAR-10_Adam_lr0.001_seed42.csv',
        'NN_ResNet18_CIFAR10_SGD_lr0.01_seed42.csv',
        'NN_SimpleMLP_MNIST_Adam_lr0.001_seed43.csv'
    ]

    for n in names:
        _make_sample_csv(exp_dir / n)

    # Run generator
    generate_all_plots(str(results_dir))

    viz_dir = results_dir / 'visualizations'
    assert viz_dir.exists()

    files = list(viz_dir.glob('*.png'))
    assert len(files) > 0, 'No visualization PNGs produced'

    # Ensure files are non-empty
    for f in files:
        assert f.stat().st_size > 100, f'Generated plot {f} looks empty'

    # Cleanup
    shutil.rmtree(str(results_dir))
