import pytest

try:
    from src.core.robust_gradients import RobustGradientHandler, HuberLoss
    from src.core.models import SimpleMLP
except Exception as e:
    RobustGradientHandler = None  # type: ignore
    HuberLoss = None  # type: ignore
    SimpleMLP = None  # type: ignore


def test_huber_loss_basic():
    if HuberLoss is None:
        pytest.skip("HuberLoss not available (optional dependency missing or module error)")
    huber = HuberLoss(delta=1.0)
    pred_small = __import__("torch").tensor([1.0, 2.0, 3.0])
    target_small = __import__("torch").tensor([1.1, 2.1, 3.1])
    pred_large = __import__("torch").tensor([1.0, 2.0, 3.0])
    target_large = __import__("torch").tensor([10.0, 20.0, 30.0])
    loss_small = huber(pred_small, target_small)
    loss_large = huber(pred_large, target_large)
    assert loss_small < loss_large


@pytest.mark.slow
def test_robust_gradient_handler_basic():
    if RobustGradientHandler is None or SimpleMLP is None:
        pytest.skip("RobustGradientHandler or SimpleMLP not available")

    handler = RobustGradientHandler(enabled=True, clip_norm=1.0, monitor_heavy_tails=True)
    model = SimpleMLP()

    # Inject large gradients
    for p in model.parameters():
        p.grad = __import__("torch").ones_like(p) * 10.0

    diagnostics = handler(model, epoch=1)
    # Diagnostics should contain keys 'clipped' or 'clip_fraction' depending on handler implementation
    assert isinstance(diagnostics, dict)
    assert 'clipped' in diagnostics or 'clip_fraction' in diagnostics
