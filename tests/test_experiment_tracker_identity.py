from src.core.experiment_tracker import ExperimentTracker as Canonical
import importlib.util
from pathlib import Path
import sys

import run_all_kaggle as root_run


def _import_runners_module():
    runners_path = Path(__file__).resolve().parents[1] / 'runners' / 'run_all_kaggle.py'
    if runners_path.exists():
        spec = importlib.util.spec_from_file_location('runners_run_all_kaggle', str(runners_path))
        module = importlib.util.module_from_spec(spec)
        sys.modules['runners_run_all_kaggle'] = module
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        return module
    return None


def test_experiment_tracker_identity():
    # Root runner should reference the canonical ExperimentTracker
    assert Canonical is root_run.ExperimentTracker

    runners_mod = _import_runners_module()
    if runners_mod is not None:
        assert Canonical is runners_mod.ExperimentTracker


def test_experiment_tracker_basic_api_no_mlflow():
    # Basic API calls should be safe whether or not mlflow is available
    t = Canonical()
    run_id = t.start_run()
    # If mlflow is present we expect a string run id, otherwise None
    assert run_id is None or isinstance(run_id, str)
    # These should not raise
    t.log_params({'a': 1})
    t.log_metrics({'loss': 0.5})
    # End run if started
    try:
        t.end_run()
    except Exception:
        pytest.skip("mlflow end_run failed in environment")
