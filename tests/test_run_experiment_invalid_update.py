import numpy as np
from src.experiments.run_experiment import run_single_experiment

class DummyOptimizer:
    def __init__(self):
        pass
    def reset(self):
        return
    def step(self, params, grads):
        # Return non-finite values to simulate failure in optimizer
        return (np.nan, np.nan)


def test_run_single_experiment_handles_invalid_update():
    optimizer_config = {'type': 'Dummy', 'params': {}}
    function_config = {'type': 'Rosenbrock', 'params': {'a':1, 'b':100}}
    # Monkeypatch factory
    from src.core.optimizers import create_optimizer_instance as create_orig
    import src.core.optimizers as core_opt

    def create_dummy(name, **kwargs):
        return DummyOptimizer()

    core_opt.create_optimizer_instance = create_dummy

    df = run_single_experiment(optimizer_config, function_config, initial_point=(0.0,0.0), num_iterations=3, seed=42)

    # Expect returned DataFrame and finite columns present
    assert not df.empty
    assert 'x' in df.columns and 'y' in df.columns

    # Clean up: restore original factory
    core_opt.create_optimizer_instance = create_orig
