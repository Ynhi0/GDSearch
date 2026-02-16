import numpy as np
import pytest
from src.core.optimizers import Adam, AdamW


def test_adam_and_adamw_weight_decay_produce_different_updates():
    params = (0.5, -0.25)
    grads = (0.1, -0.2)

    adam = Adam(lr=0.01, weight_decay=0.1)
    adamw = AdamW(lr=0.01, weight_decay=0.1)

    updated_adam = adam.step(params, grads)
    updated_adamw = adamw.step(params, grads)

    arr_adam = np.asarray(updated_adam)
    arr_adamw = np.asarray(updated_adamw)

    # With decoupled vs coupled weight decay we expect different parameter updates
    assert not np.allclose(arr_adam, arr_adamw)

    # Sanity: both optimizers should reduce the magnitude of a positive parameter when weight decay > 0
    assert abs(arr_adam[0]) < abs(params[0])
    assert abs(arr_adamw[0]) < abs(params[0])
