import os
import sys
import pandas as pd

import run_all_kaggle as rag


def _call_main_with_args(argv, monkeypatch, tmp_path, env=None, stubs=None):
    # helper: set sys.argv, optional env var mapping and stub experiment functions
    old_argv = sys.argv.copy()
    try:
        sys.argv = [sys.argv[0]] + argv
        # set results-dir to tmp_path to avoid repo writes
        if '--results-dir' not in sys.argv:
            sys.argv += ['--results-dir', str(tmp_path)]

        # set environment variables
        if env:
            for k, v in env.items():
                monkeypatch.setenv(k, v)

        # stub experiment functions (capture kwargs)
        captures = {}
        if stubs is None:
            stubs = ['run_2d_experiments', 'run_robustness_analysis', 'run_sam_sensitivity', 'run_ablation_study']

        for name in stubs:
            def _make_stub(n):
                def _stub(**kwargs):
                    captures[n] = kwargs
                    return pd.DataFrame()
                return _stub
            monkeypatch.setattr(rag, name, _make_stub(name))

        res = rag.main()
        return captures, res
    finally:
        sys.argv = old_argv


def test_experiment_mode_env_quick_sets_quick(monkeypatch, tmp_path):
    env = {'EXPERIMENT_MODE': 'quick'}
    argv = ['--experiments', '2d,robustness,sam', '--seeds', '42,123,456']
    captures, _ = _call_main_with_args(argv, monkeypatch, tmp_path, env=env)

    # All three experiment stubs should have been called with quick=True
    assert 'run_2d_experiments' in captures
    assert captures['run_2d_experiments'].get('quick') is True
    assert 'run_robustness_analysis' in captures
    assert captures['run_robustness_analysis'].get('quick') is True
    assert 'run_sam_sensitivity' in captures
    assert captures['run_sam_sensitivity'].get('quick') is True
    # ULTRA_QUICK_MODE must be False for quick
    assert rag.ULTRA_QUICK_MODE is False


def test_experiment_mode_env_ultra_sets_ultra_flag(monkeypatch, tmp_path):
    env = {'EXPERIMENT_MODE': 'ultra_quick'}
    argv = ['--experiments', '2d,robustness,sam', '--seeds', '42,123,456']
    captures, _ = _call_main_with_args(argv, monkeypatch, tmp_path, env=env)

    assert captures['run_2d_experiments'].get('quick') is True
    assert captures['run_robustness_analysis'].get('quick') is True
    assert captures['run_sam_sensitivity'].get('quick') is True
    # ULTRA_QUICK_MODE must be True when ultra_quick requested
    assert rag.ULTRA_QUICK_MODE is True


def test_experiment_mode_cli_overrides_env(monkeypatch, tmp_path):
    # ENV says quick, CLI explicitly sets full -> should override
    monkeypatch.setenv('EXPERIMENT_MODE', 'quick')
    argv = ['--experiment-mode', 'full', '--experiments', '2d,robustness', '--seeds', '42,123,456']
    captures, _ = _call_main_with_args(argv, monkeypatch, tmp_path)

    assert captures['run_2d_experiments'].get('quick') is False
    assert captures['run_robustness_analysis'].get('quick') is False
    assert rag.ULTRA_QUICK_MODE is False
