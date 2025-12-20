import json
import os

from src.experiments.run_nn_experiment import parse_experiments_from_config


def test_nn_tuning_config_parses():
    """Test that configs/nn_tuning.json parses to non-empty experiments list."""
    # Get repo root directory (parent of tests directory)
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(tests_dir)
    cfg_path = os.path.join(repo_root, 'configs', 'nn_tuning.json')
    
    assert os.path.exists(cfg_path), f"Config file not found at {cfg_path}"

    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    experiments = parse_experiments_from_config(cfg)
    assert isinstance(experiments, list), "Experiments should be a list"
    assert len(experiments) > 0, "Parsed zero experiments from configs/nn_tuning.json"
    
    # Validate structure of experiments
    for exp in experiments:
        assert 'model' in exp, "Experiment missing 'model' key"
        assert 'dataset' in exp, "Experiment missing 'dataset' key"
        assert 'optimizer' in exp, "Experiment missing 'optimizer' key"
        assert 'lr' in exp, "Experiment missing 'lr' key"
