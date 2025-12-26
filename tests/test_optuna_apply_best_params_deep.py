from src.core.optuna_tuner import apply_best_params_to_config


def test_deep_merge_of_nested_config():
    cfg = {
        'model': 'SimpleMLP',
        'dataset': 'MNIST',
        'optimizer': 'Adam',
        'lr': 0.01,
        'convergence': {
            'grad_norm_threshold': 1e-6,
            'loss_delta_threshold': 1e-7,
            'loss_window': 100
        },
    }

    best = {
        'lr': 0.001,
        'convergence': {
            'loss_delta_threshold': 1e-9
        }
    }

    merged = apply_best_params_to_config(cfg, best)

    assert merged['lr'] == 0.001
    # Nested value should be updated, others preserved
    assert merged['convergence']['grad_norm_threshold'] == 1e-6
    assert merged['convergence']['loss_delta_threshold'] == 1e-9
    assert merged['convergence']['loss_window'] == 100
