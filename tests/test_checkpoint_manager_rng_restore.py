import random
import numpy as np
import torch
from src.core.checkpoint_manager import RobustCheckpointManager


def test_restore_rng_states_roundtrip(tmp_path):
    manager = RobustCheckpointManager(base_dir=str(tmp_path))

    # Set RNGs to a known state and capture state BEFORE drawing random numbers
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    ckpt = {'rng_states': {
        'torch_cpu_rng_state': torch.get_rng_state(),
        'numpy_random_state': np.random.get_state(),
        'python_random_state': random.getstate(),
        'torch_cuda_rng_state_all': None
    }}

    # Now draw random numbers (these should be reproducible after restore)
    r_torch_1 = torch.rand(4)
    r_np_1 = np.random.rand(4)
    r_py_1 = random.random()

    # Change RNGs
    torch.manual_seed(999)
    np.random.seed(999)
    random.seed(999)

    # Restore
    manager.restore_rng_states(ckpt)

    r_torch_2 = torch.rand(4)
    r_np_2 = np.random.rand(4)
    r_py_2 = random.random()

    assert torch.allclose(r_torch_1, r_torch_2), "PyTorch RNG not restored"
    assert np.allclose(r_np_1, r_np_2), "NumPy RNG not restored"
    assert r_py_1 == r_py_2, "Python RNG not restored"
