import pytest
from src.core.oom_handler import _call_optimizer_step


class DummyClosureOptimizer:
    def step(self, closure):
        # Simulate optimizer that returns None when given a closure
        closure()
        return None


def test_call_optimizer_step_requires_return_from_closure():
    opt = DummyClosureOptimizer()

    def closure():
        return 0.5

    with pytest.raises(RuntimeError):
        _call_optimizer_step(opt, closure=closure)