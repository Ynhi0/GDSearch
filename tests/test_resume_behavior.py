import json
from pathlib import Path
import pandas as pd
import pytest

from src.core import resume_utils


def test_skip_if_results_exist(tmp_path):
    sig = resume_utils.compute_run_signature({'dataset': 'X', 'model': 'M', 'optimizer': 'O', 'seed': 1})
    # create results dir and a summary file containing the signature with a final metric
    results_dir = tmp_path / 'results'
    results_dir.mkdir()
    summary = pd.DataFrame([
        {'dataset': 'X', 'model': 'M', 'optimizer': 'O', 'seed': 1, 'final_test_acc': 0.5, 'run_signature': sig, 'completed': True}
    ])
    summary.to_csv(results_dir / 'summary_quantitative.csv', index=False)

    action = resume_utils.decide_resume_action(None, results_dir, sig, 'skip_if_results_exist')
    assert action == 'skip'


def test_error_if_no_checkpoint_and_no_results(tmp_path):
    sig = resume_utils.compute_run_signature({'dataset': 'X', 'model': 'M', 'optimizer': 'O', 'seed': 2})
    results_dir = tmp_path / 'results'
    results_dir.mkdir()
    # No summary file, no checkpoint -> should raise
    with pytest.raises(RuntimeError):
        resume_utils.decide_resume_action(None, results_dir, sig, 'error_if_no_checkpoint')


def test_restart_if_no_checkpoint(tmp_path):
    sig = resume_utils.compute_run_signature({'dataset': 'X', 'model': 'M', 'optimizer': 'O', 'seed': 3})
    results_dir = tmp_path / 'results'
    results_dir.mkdir()

    action = resume_utils.decide_resume_action(None, results_dir, sig, 'restart_if_no_checkpoint')
    assert action == 'restart'
