import os
import pytest
from src.core.reproducibility import load_metadata, verify_checkpoint_with_metadata

META_PATH = 'artifacts/resnet18_cifar10_seed1011_meta.json'


def test_metadata_exists_and_fields():
    assert os.path.exists(META_PATH), f"Metadata file missing: {META_PATH}"
    meta = load_metadata(META_PATH)
    # Minimal schema checks
    assert 'checkpoint' in meta
    assert 'seed' in meta
    assert 'dataset' in meta and meta['dataset'] == 'CIFAR-10'
    assert 'model' in meta and 'ResNet' in meta['model']
    assert 'accuracy' in meta


@pytest.mark.skipif(not os.path.exists('artifacts/checkpoints/CIFAR10_ResNet18_Adam_seed1011.pt'), reason="Checkpoint not present, add artifacts/checkpoints/CIFAR10_ResNet18_Adam_seed1011.pt to enable verification")
def test_verify_checkpoint_matches_claim():
    res = verify_checkpoint_with_metadata(META_PATH, tolerance=0.02)
    assert res['status'] == 'verified', f"Reproducibility verification failed: {res}"