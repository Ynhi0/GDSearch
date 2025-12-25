import torch
from src.experiments.run_nn_experiment import build_optimizer


def test_build_optimizer_nesterov_returns_sgd_with_nesterov():
    model = torch.nn.Linear(10, 1)
    opt = build_optimizer('SGD_Nesterov', model, lr=0.01, momentum=0.9)
    assert isinstance(opt, torch.optim.SGD)
    # PyTorch optimizer stores default params in `defaults`
    assert opt.defaults.get('nesterov', False) is True
