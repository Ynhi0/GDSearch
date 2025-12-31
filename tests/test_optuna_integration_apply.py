import torch
from src.core.optuna_tuner import apply_best_params_to_config
from src.experiments.run_nn_experiment import build_model_and_data, build_optimizer


def test_integration_apply_best_params_and_build_optimizer():
    base_cfg = {
        'model': 'SimpleMLP',
        'dataset': 'MNIST',
        'optimizer': 'SGD',
        'lr': 0.01,
        'momentum': 0.0,
        'batch_size': 16,
        'seed': 42
    }

    # Suppose Optuna found these best params
    best_params = {
        'lr': 0.005,
        'momentum': 0.88,
        'optimizer': 'SGDMomentum'
    }

    merged = apply_best_params_to_config(base_cfg, best_params)

    # Build model and check optimizer produced honors tuned params
    device = torch.device('cpu')
    res = build_model_and_data('MNIST', 'SimpleMLP', batch_size=16, device=device, seed=42)
    assert len(res) == 4, f"Expected 4 returns from build_model_and_data, got {len(res)}"
    model, train_loader, val_loader, test_loader = res
    assert val_loader is None, "val_loader should be None when val_split not provided"

    opt = build_optimizer(
        optimizer_name=merged['optimizer'],
        model=model,
        lr=float(merged['lr']),
        momentum=float(merged.get('momentum', 0.0))
    )

    # For SGD with momentum we expect an optimizer with a momentum attribute
    import pytest
    if isinstance(opt, torch.optim.SGD):
        assert opt.defaults['lr'] == float(merged['lr'])
        assert opt.defaults.get('momentum', 0.0) == pytest.approx(0.88)
    else:
        # If custom wrapper is returned, attempt to inspect via attributes
        # Generic expectation: the optimizer should at least accept lr and momentum
        assert hasattr(opt, 'param_groups')
