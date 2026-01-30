import pytest
import numpy as np

from src.utils.num_utils import safe_to_float, safe_len


def test_safe_to_float_basic():
    assert safe_to_float(3) == 3.0
    assert safe_to_float(3.5) == 3.5
    assert pytest.approx(safe_to_float(np.array(5))) == 5.0


def test_safe_to_float_list_and_none():
    assert safe_to_float([1, 2, 3]) == 1.0
    assert str(safe_to_float(None)) == 'nan'


def test_safe_len_various():
    assert safe_len(None) == 0
    assert safe_len([]) == 0
    assert safe_len([1,2,3]) == 3
    arr = np.arange(12).reshape(3,4)
    assert safe_len(arr) == 12

    try:
        import torch
        t = torch.arange(5)
        assert safe_len(t) == 5
    except Exception:
        # torch may not be available in all test environments
        pass

class DummyNoLen:
    pass

class DummyLen:
    def __len__(self):
        return 7


def test_safe_len_object_fallbacks():
    assert safe_len(DummyNoLen()) == 0
    assert safe_len(DummyLen()) == 7
