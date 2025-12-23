"""Test IMDB dataset loading with different methods."""
import pytest
import sys

# Skip on Python 3.13 only (known fsspec incompatibility)
skip_python313 = pytest.mark.skipif(
    sys.version_info >= (3, 13),
    reason="Python 3.13 has known fsspec glob pattern incompatibility with HuggingFace datasets"
)


@skip_python313
def test_imdb_standard_load():
    """Test standard IMDB dataset loading."""
    from datasets import load_dataset
    data = load_dataset('imdb', split='train[:10]')
    assert len(data) == 10


@skip_python313
def test_imdb_force_redownload():
    """Test IMDB loading with force_redownload."""
    from datasets import load_dataset
    data = load_dataset('imdb', split='train[:10]', download_mode='force_redownload')
    assert len(data) == 10


@skip_python313
def test_imdb_stanfordnlp_version():
    """Test IMDB loading using stanfordnlp namespace."""
    from datasets import load_dataset
    data = load_dataset('stanfordnlp/imdb', split='train[:10]')
    assert len(data) == 10


@skip_python313
def test_imdb_trust_remote_code():
    """Test IMDB loading with trust_remote_code=True."""
    from datasets import load_dataset
    data = load_dataset('imdb', split='train[:10]', trust_remote_code=True)
    assert len(data) == 10

