import torch
import torch.nn as nn

from src.core.torch_native_optimizers import TorchSAM


def _build_model():
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))
    return model


def test_sam_cleans_old_p_on_success():
    model = _build_model()
    criterion = nn.MSELoss()

    # Setup SAM with SGD as base optimizer
    sam = TorchSAM(model.parameters(), base_optimizer=torch.optim.SGD, rho=0.05, lr=0.1)

    def closure():
        sam.zero_grad()
        x = torch.randn(2, 4)
        y = torch.randn(2, 4)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        return loss

    # Run a SAM step successfully
    sam.step(closure)

    # Ensure no saved copies remain
    for p in sam.param_groups[0]['params']:
        assert 'old_p' not in sam.state.get(p, {}), "old_p leaked after successful SAM step"


def test_sam_cleans_old_p_on_exception():
    model = _build_model()
    criterion = nn.MSELoss()

    sam = TorchSAM(model.parameters(), base_optimizer=torch.optim.SGD, rho=0.05, lr=0.1)

    # Closure that raises on second call to simulate failure during second pass
    state = {'calls': 0}

    def closure():
        state['calls'] += 1
        sam.zero_grad()
        x = torch.randn(2, 4)
        y = torch.randn(2, 4)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        if state['calls'] >= 2:
            raise RuntimeError("simulated failure during SAM second pass")
        return loss

    try:
        sam.step(closure)
    except RuntimeError:
        pass

    # Even after an exception, ensure cleanup happened
    for p in sam.param_groups[0]['params']:
        assert 'old_p' not in sam.state.get(p, {}), "old_p leaked after SAM exception"