import random
import numpy as np
import torch
from src.core.checkpoint_manager import RobustCheckpointManager


def test_restore_rng_states_roundtrip(tmp_path):
    manager = RobustCheckpointManager(base_dir=str(tmp_path))

    # Set RNGs to a known state and draw random numbers FIRST
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Draw first batch of random numbers
    r_torch_1 = torch.rand(4)
    r_np_1 = np.random.rand(4)
    r_py_1 = random.random()

    # NOW capture the RNG state AFTER the draws (this is the state we want to restore to)
    ckpt = {'rng_states': {
        'torch_cpu_rng_state': torch.get_rng_state(),
        'numpy_random_state': np.random.get_state(),
        'python_random_state': random.getstate(),
        'torch_cuda_rng_state_all': None
    }}

    # Draw second batch to advance state further
    r_torch_intermediate = torch.rand(4)
    r_np_intermediate = np.random.rand(4)
    r_py_intermediate = random.random()

    # Change RNGs to a completely different state
    torch.manual_seed(999)
    np.random.seed(999)
    random.seed(999)

    # Restore to the checkpoint (after first draw, before second)
    manager.restore_rng_states(ckpt)

    # Draw again - should match the INTERMEDIATE values (not the first ones)
    r_torch_2 = torch.rand(4)
    r_np_2 = np.random.rand(4)
    r_py_2 = random.random()

    assert torch.allclose(r_torch_intermediate, r_torch_2), "PyTorch RNG not restored correctly"
    assert np.allclose(r_np_intermediate, r_np_2), "NumPy RNG not restored correctly"
    assert r_py_intermediate == r_py_2, "Python RNG not restored correctly"
