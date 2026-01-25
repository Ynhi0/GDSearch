import tempfile
import shutil
from pathlib import Path
import pandas as pd
import json
import warnings
from src.visualization.cifar_viz import create_cifar10_visualizations


def _write_csv(path: Path, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def test_create_cifar10_visualizations(tmp_path):
    results_dir = tmp_path
    exp_dir = tmp_path / 'experiments' / 'cifar10'
    exp_dir.mkdir(parents=True)

    # Good time-series CSVs
    df1 = pd.DataFrame({
        'epoch': [1, 2, 3],
        'test_acc': [0.78, '0.80', '82%'],
        'train_loss': [1.2, 1.0, 0.9]
    })
    _write_csv(exp_dir / 'CIFAR10_ResNet18_Adam_seed42.csv', df1)

    df2 = pd.DataFrame({
        'epoch': [1, 2, 3],
        'test_acc': ['92%', 0.93, 0.935],
        'train_loss': [1.4, 1.1, 0.95],
        'seed': [1011, 1011, 1011]
    })
    _write_csv(exp_dir / 'CIFAR10_ResNet18_AdaBound_seed1011.csv', df2)

    # Corrupt CSV (empty)
    (exp_dir / 'empty_corrupt.csv').write_text('')

    # Summary CSV (with mean/std)
    summary = pd.DataFrame({'optimizer': ['Adam', 'AdaBound'], 'mean': ['92.8%', '0.928'], 'std': [1.2, 0.5]})
    _write_csv(exp_dir / 'CIFAR10_summary.csv', summary)

    # Malformed CSV with non-finite entries
    df_bad = pd.DataFrame({'epoch': [1, 2], 'test_acc': [float('nan'), 'not_a_number'], 'seed': [42, 42]})
    _write_csv(exp_dir / 'CIFAR10_ResNet18_Bad_seed42.csv', df_bad)

    csv_files = list(exp_dir.glob('*.csv'))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        outputs = create_cifar10_visualizations(results_dir, csv_files)

        # Ensure no matplotlib posx/posy warnings
        pos_warnings = [str(x.message).lower() for x in w if 'posx' in str(x.message).lower() or 'posy' in str(x.message).lower()]
        assert not pos_warnings, f"Found positional warnings: {pos_warnings}"

    # Assert output files exist
    assert 'train_loss' in outputs and outputs['train_loss'].exists()
    assert 'test_accuracy' in outputs and outputs['test_accuracy'].exists()
    assert 'final_comparison' in outputs and outputs['final_comparison'].exists()
    # Meta JSON exists and y_max <= 100
    meta = json.loads(outputs['final_meta'].read_text())
    assert meta['y_max'] <= 100
    assert 'Adam' in meta['means']
