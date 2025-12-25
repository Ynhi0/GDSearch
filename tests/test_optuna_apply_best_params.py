from src.core.optuna_tuner import apply_best_params_to_config


def test_apply_best_params_merges_and_normalizes():
    cfg = {
        'model': 'SimpleMLP',
        'dataset': 'MNIST',
        'optimizer': 'SGD',
        'lr': 0.01
    }

    best = {
        'lr': 0.001,
        'momentum': 0.85,
        'optimizer': 'SGDMomentum'
    }

    merged = apply_best_params_to_config(cfg, best)

    assert merged['lr'] == 0.001
    assert merged['momentum'] == 0.85
    # Normalization should map 'SGDMomentum' -> 'SGD_Momentum'
    assert merged['optimizer'] == 'SGD_Momentum'


def test_apply_best_params_handles_missing_optimizer():
    cfg = {'model': 'SimpleMLP', 'dataset': 'MNIST', 'lr': 0.01}
    best = {'lr': 0.002}
    merged = apply_best_params_to_config(cfg, best)
    assert merged['lr'] == 0.002
    assert 'optimizer' not in merged
