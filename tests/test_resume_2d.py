import os
from pathlib import Path

import pytest

import run_all_kaggle as runner
from src.core.checkpoint_manager import RobustCheckpointManager


def test_run_2d_experiments_respects_completed_checkpoint(tmp_path):
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    mgr = RobustCheckpointManager(str(ckpt_dir))

    # Create a completed checkpoint for Rosenbrock + SAM_SGD seed=1
    ckpt_name = "2D_Rosenbrock_SAM_SGD_seed1.pt"
    ckpt = {
        'opt_name': 'SAM_SGD',
        'optimizer': {'param_groups': []},
        'x': [0.0, 0.0],
        'iteration': 10,
        'history': [{'iteration': 10, 'loss': 1e-8}],
        'metadata': {'experiment': '2D', 'function': 'Rosenbrock', 'seed': 1, 'completed': True}
    }
    assert mgr.save_checkpoint(ckpt, ckpt_name, "2D_Rosenbrock_SAM_SGD_seed1")

    results_dir = tmp_path / "results_2d"
    df = runner.run_2d_experiments(results_dir=str(results_dir), seeds=[1], quick=True, checkpoint_manager=mgr, resume=True)

    # SAM_SGD seed 1 should be skipped because checkpoint marked completed
    mask = (df['optimizer'] == 'SAM_SGD') & (df['seed'] == 1) & (df['function'] == 'Rosenbrock')
    assert not mask.any(), "SAM_SGD run with completed checkpoint should be skipped"


def test_run_robustness_analysis_marks_checkpoint_completed(tmp_path):
    ckpt_dir = tmp_path / "ckpts2"
    ckpt_dir.mkdir()
    mgr = RobustCheckpointManager(str(ckpt_dir))

    # Partial checkpoint (not completed) for SAM on Rosenbrock start index 0
    ckpt_name = "Robustness_Rosenbrock_SAM_SGD_seed42_start0.pt"
    ckpt = {
        'opt_name': 'SAM_SGD',
        'optimizer': {'param_groups': []},
        'x': [0.5, 0.5],
        'iteration': 2,
        'history': [],
        'metadata': {'experiment': 'Robustness', 'function': 'Rosenbrock', 'seed': 42, 'start_point': (0.5, 0.5), 'completed': False}
    }
    assert mgr.save_checkpoint(ckpt, ckpt_name, "Robustness_Rosenbrock_SAM_SGD_seed42_start0")

    results_dir = tmp_path / "results_robust"
    df = runner.run_robustness_analysis(results_dir=str(results_dir), seeds=[42], quick=True, checkpoint_manager=mgr, resume=True)

    # After run, checkpoint for that start should be marked completed
    final_ckpt = mgr.load_checkpoint(ckpt_name)
    assert final_ckpt is not None
    assert final_ckpt.get('metadata', {}).get('completed', False) is True
    # And results must contain at least one row for SAM_SGD seed 42
    assert any((df['optimizer'] == 'SAM_SGD') & (df['seed'] == 42))
