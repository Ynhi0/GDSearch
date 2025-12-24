import json
import os
from pathlib import Path
from run_all_kaggle import save_tuning_artifact


def test_save_tuning_artifact(tmp_path):
    base = tmp_path
    experiment = 'MNIST_Benchmark'
    model = 'SimpleMLP'
    opt = 'SGD'
    meta = {
        'tuning_method': 'optuna_seed_aggregation',
        'n_trials': 3,
        'tune_seeds': [42],
        'GDSEARCH_TUNE_EVAL_ALL_CANDIDATES': False,
        'GDSEARCH_TUNE_TOPK': 5,
        'per_seed_results': [({'lr': 0.01}, 85.0)],
        'selected_params': {'lr': 0.01},
        'selected_mean_val': 85.0,
        'selected_std_val': 0.0
    }

    out = save_tuning_artifact(str(base), experiment, model, opt, meta)
    assert out is not None

    p = Path(out)
    assert p.exists()

    with open(p, 'r', encoding='utf-8') as f:
        data = json.load(f)

    assert data['tuning_method'] == 'optuna_seed_aggregation'
    assert 'provenance' in data
    assert 'git' in data['provenance']
    assert data['selected_params']['lr'] == 0.01
