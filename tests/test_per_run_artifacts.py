import os
import json
import tempfile

import pytest

from run_all_kaggle import save_run_artifacts


def test_save_run_artifacts_creates_files(tmp_path):
    base = tmp_path / "results"
    history = [{'epoch': 1, 'train_loss': 0.5, 'test_acc': 90.0}, {'epoch': 2, 'train_loss': 0.4, 'test_acc': 91.0}]
    params = {'lr': 0.01, 'batch_size': 32}

    csv_path, meta_path = save_run_artifacts(str(base), 'MNIST', 'SimpleMLP', 'SGD', 999, history, params)

    assert csv_path is not None and meta_path is not None
    assert os.path.exists(csv_path)
    assert os.path.exists(meta_path)

    # Validate CSV content (simple check)
    with open(csv_path, 'r', encoding='utf-8') as f:
        txt = f.read()
    assert 'epoch' in txt and 'train_loss' in txt

    # Validate metadata
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    assert meta['dataset'] == 'MNIST'
    assert meta['model'] == 'SimpleMLP'
    assert meta['optimizer'] == 'SGD'
    assert meta['seed'] == 999
    assert 'system' in meta


def test_save_run_artifacts_handles_empty_history(tmp_path):
    base = tmp_path / "results"
    history = []
    params = {'lr': 0.01}

    csv_path, meta_path = save_run_artifacts(str(base), 'MNIST', 'SimpleMLP', 'SGD', 1001, history, params)

    assert csv_path is not None and meta_path is not None
    # CSV should exist and not be zero-byte
    assert os.path.exists(csv_path)
    assert os.path.getsize(csv_path) > 0

    # Metadata should mark run as not completed and tainted
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    assert meta['completed'] is False
    assert meta.get('tainted', False) is True
    assert meta['rows'] == 0
