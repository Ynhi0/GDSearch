import pandas as pd
from src.experiments.run_experiment import run_single_experiment


def test_run_single_experiment_sgdmomentum():
    optimizer_config = {'type': 'SGDMomentum', 'params': {'lr': 0.01, 'beta': 0.9}}
    function_config = {'type': 'Rosenbrock', 'params': {'a': 1, 'b': 100}}
    df = run_single_experiment(optimizer_config, function_config, initial_point=(-1.5, 2.0), num_iterations=10, seed=0)
    assert isinstance(df, pd.DataFrame)
    assert 'loss' in df.columns
    assert len(df) == 10
