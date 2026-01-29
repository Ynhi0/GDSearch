import os
import tempfile
import pandas as pd
from pathlib import Path
import logging

from run_all_kaggle import create_experiment_visualizations
from src.visualization.plotting_utils import normalize_final_results


def test_cifar_visual_pipeline(tmp_path, caplog):
    # Create a small synthetic CIFAR10 results folder
    cifar_dir = tmp_path / 'CIFAR10'
    cifar_dir.mkdir()

    # Corrupt CSV (empty)
    corrupt = cifar_dir / 'corrupt.csv'
    corrupt.write_text('')

    # Time series 1 - fraction encoding
    ts1 = cifar_dir / 'CIFAR10_ResNet18_Adam_seed42.csv'
    ts1.write_text('epoch,val_acc\n1,0.78\n2,0.85\n')

    # Time series 2 - percentage encoding as strings
    ts2 = cifar_dir / 'CIFAR10_ResNet18_SGD_seed1011.csv'
    ts2.write_text('epoch,val_acc\n1,78%\n2,86%\n')

    # Summary CSV (final test accs)
    summary = cifar_dir / 'summary.csv'
    summary.write_text('optimizer,mean,std\nAdam,0.85,0.02\nSGD,86,1.5\n')

    csv_files = [corrupt, ts1, ts2, summary]

    caplog.set_level(logging.WARNING)
    # Run visualization creation
    create_experiment_visualizations('CIFAR10', str(tmp_path), csv_files)

    static_dir = tmp_path / 'visualizations' / 'static' / 'cifar10'
    acc_img = static_dir / 'cifar10_val_accuracy.png'
    final_img = static_dir / 'cifar10_final_comparison.png'

    assert acc_img.exists()
    assert final_img.exists()

    # Ensure no matplotlib text warnings captured
    messages = '\n'.join([r.message for r in caplog.records])
    assert 'posx' not in messages and 'posy' not in messages

    # Validate final results normalized are <=100
    df = pd.read_csv(summary)
    norm = normalize_final_results(df.set_index('optimizer')['mean'])
    assert len(norm) > 0
    assert norm['mean'].max() <= 100.0
