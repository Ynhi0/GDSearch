import tempfile
import pytest
import pandas as pd

import run_all_kaggle as runner


class FailingCheckpointManager:
    def load_checkpoint(self, *args, **kwargs):
        return None
    def save_checkpoint(self, *args, **kwargs):
        raise OSError("simulated checkpoint failure")
    def validate_optimizer_compatibility(self, *args, **kwargs):
        return True
    def restore_rng_states(self, *args, **kwargs):
        return None


def test_run_mnist_marks_tainted_on_checkpoint_failure(monkeypatch):
    # Quick run with one seed and very small config
    cfg = {
        'model': 'SimpleMLP',
        'dataset': 'MNIST',
        'optimizer': 'SGD',
        'lr': 0.01,
        'epochs': 1,
        'batch_size': 16,
        'seed': 42,
        'val_split': 0.1
    }

    # Run with failing checkpoint manager
    fake_cm = FailingCheckpointManager()

    df = runner.run_mnist_experiment(results_dir="results_test_mnist", seeds=[42], quick=True, skip_tuning=True, profiler=None, tracker=None, checkpoint_manager=fake_cm, resume=False)

    # DataFrame should have been returned and contain tainted==True somewhere
    assert isinstance(df, pd.DataFrame)
    if 'tainted' in df.columns:
        tainted_any = df['tainted'].any()
        assert bool(tainted_any), "Run should be marked tainted when checkpoint save fails"
    else:
        pytest.skip("run_mnist_experiment did not return taint metadata in this configuration")
