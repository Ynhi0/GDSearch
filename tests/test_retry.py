import pytest
import time
from src.core.retry import retry_with_backoff, retry_operation


def test_retry_retries_on_transient_exception(monkeypatch):
    calls = {"count": 0}

    @retry_with_backoff(max_retries=3, initial_backoff=0.0, backoff_factor=1.0)
    def flaky():
        calls["count"] += 1
        if calls["count"] < 3:
            raise OSError("transient")
        return "ok"

    # Patch sleep to avoid slowing tests
    monkeypatch.setattr(time, "sleep", lambda s: None)

    result = flaky()
    assert result == "ok"
    assert calls["count"] == 3


def test_retry_operation_functional_interface(monkeypatch):
    calls = {"count": 0}

    def op():
        calls["count"] += 1
        if calls["count"] < 2:
            raise OSError("transient")
        return 42

    monkeypatch.setattr(time, "sleep", lambda s: None)
    res = retry_operation(op, max_retries=3, initial_backoff=0.0, backoff_factor=1.0)
    assert res == 42
    assert calls["count"] == 2


def test_retry_does_not_catch_keyboardinterrupt():
    @retry_with_backoff(max_retries=3)
    def raises_kb():
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        raises_kb()
