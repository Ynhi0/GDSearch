import torch
from src.core.optimizer_factory import OptimizerFactory


def test_optimizer_factory_registers_wrappers_and_creates_instances():
    # Ensure fresh init for test isolation
    OptimizerFactory._initialized = False

    model = torch.nn.Linear(2, 2)

    for name in ('radam', 'adabound', 'lamb'):
        opt = OptimizerFactory.create(name, model.parameters(), lr=1e-3)
        assert opt is not None
        # must be a torch Optimizer or a wrapper inheriting from it
        assert isinstance(opt, torch.optim.Optimizer)
