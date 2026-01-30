from src.utils.reproducibility import set_reproducibility_mode, check_reproducibility_status
import numpy as np
import random


def test_reproducibility_consistent_cpu():
    set_reproducibility_mode(123, deterministic=True, benchmark=False)

    # Two sequences should be identical when seed is re-applied
    a1 = [random.random() for _ in range(5)]
    b1 = np.random.rand(5)

    set_reproducibility_mode(123, deterministic=True, benchmark=False)

    a2 = [random.random() for _ in range(5)]
    b2 = np.random.rand(5)

    assert a1 == a2
    assert (b1 == b2).all()


def test_check_repro_status_runs():
    status = check_reproducibility_status()
    assert 'cublas_workspace_config' in status