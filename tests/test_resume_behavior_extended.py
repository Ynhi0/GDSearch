import numpy as np
import pandas as pd
from src.core import resume_utils


def test_completed_string_false_not_considered_complete(tmp_path):
    sig = resume_utils.compute_run_signature({'dataset': 'X', 'model': 'M', 'optimizer': 'O', 'seed': 10})
    results_dir = tmp_path / 'results'
    results_dir.mkdir()
    # completed as string 'False'
    summary = pd.DataFrame([
        {'dataset': 'X', 'model': 'M', 'optimizer': 'O', 'seed': 10, 'final_test_acc': np.nan, 'run_signature': sig, 'completed': 'False'}
    ])
    summary.to_csv(results_dir / 'summary_quantitative.csv', index=False)
    assert not resume_utils.results_exist(results_dir, sig)


def test_completed_string_true_considered_complete(tmp_path):
    sig = resume_utils.compute_run_signature({'dataset': 'X', 'model': 'M', 'optimizer': 'O', 'seed': 11})
    results_dir = tmp_path / 'results'
    results_dir.mkdir()
    summary = pd.DataFrame([
        {'dataset': 'X', 'model': 'M', 'optimizer': 'O', 'seed': 11, 'final_test_acc': np.nan, 'run_signature': sig, 'completed': 'True'}
    ])
    summary.to_csv(results_dir / 'summary_quantitative.csv', index=False)
    assert resume_utils.results_exist(results_dir, sig)


def test_final_metric_string_nan_not_considered_complete(tmp_path):
    sig = resume_utils.compute_run_signature({'dataset': 'X', 'model': 'M', 'optimizer': 'O', 'seed': 12})
    results_dir = tmp_path / 'results'
    results_dir.mkdir()
    # final_test_acc as string 'nan' should not be considered
    summary = pd.DataFrame([
        {'dataset': 'X', 'model': 'M', 'optimizer': 'O', 'seed': 12, 'final_test_acc': 'nan', 'run_signature': sig}
    ])
    summary.to_csv(results_dir / 'summary_quantitative.csv', index=False)
    assert not resume_utils.results_exist(results_dir, sig)
