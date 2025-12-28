import os
from pathlib import Path
import pytest
from PIL import Image
import numpy as np

from src.core.medical_data_utils import load_kaggle_medical_dataset
from src.utils.safe_len import len_sized


def _make_image(path, size=(32, 32), color=(128, 128, 128)):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new('RGB', size, color)
    img.save(path)


def test_load_kaggle_with_train_val(tmp_path):
    base = tmp_path / 'kaggle_med'
    train_a = base / 'train' / 'A'
    val_a = base / 'val' / 'A'
    _make_image(train_a / 'img1.jpg')
    _make_image(val_a / 'img2.jpg')

    result = load_kaggle_medical_dataset(str(base), img_size=32, split_seed=0)
    assert result is not None, "Expected datasets to be returned"
    train_ds, val_ds = result
    assert len_sized(train_ds) == 1
    assert len_sized(val_ds) == 1


def test_load_kaggle_train_only_splitting(tmp_path):
    base = tmp_path / 'kaggle_med2'
    train_a = base / 'train' / 'A'
    # Create 5 samples
    for i in range(5):
        _make_image(train_a / f'img{i}.jpg')

    result = load_kaggle_medical_dataset(str(base), img_size=32, split_seed=42)
    assert result is not None
    train_ds, val_ds = result
    # Default split is 80/20 -> with 5 samples val_size=1
    assert len_sized(train_ds) == 4
    assert len_sized(val_ds) == 1
