import pytest
import io
import os
from pathlib import Path
import logging

pytest.importorskip("matplotlib")
import matplotlib.pyplot as plt
from src.visualization.plotting_utils import plot_protect
from src.visualization import cifar_viz


def make_sample_csv(tmp_path: Path):
    p1 = tmp_path / "run1.csv"
    p1.write_text("epoch,optimizer,train_loss,val_acc\n0,sgd,1.0,0.2\n1,sgd,0.8,0.3")
    p2 = tmp_path / "summary.csv"
    p2.write_text("optimizer,mean,std\nsgd,0.3,0.05")
    return [p1, p2]


def test_plot_failure_does_not_disable_future_plots(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    files = make_sample_csv(tmp_path)

    # monkeypatch savefig so first call raises, subsequent calls behave normally
    orig_save = plt.savefig
    call_count = {"n": 0}

    def fake_savefig(path, *args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise OSError("disk error simulated")
        return orig_save(path, *args, **kwargs)

    monkeypatch.setattr(plt, "savefig", fake_savefig)

    # First invocation should not raise but should log a warning
    caplog.clear()
    outputs1 = cifar_viz.create_cifar10_visualizations(str(tmp_path), [str(files[0])])
    assert any("Plotting failed" in rec.message or "Could not" in rec.message for rec in caplog.records)

    # Restore normal save behavior for subsequent plots
    monkeypatch.setattr(plt, "savefig", orig_save)

    # Second invocation should succeed and produce output files
    caplog.clear()
    outputs2 = cifar_viz.create_cifar10_visualizations(str(tmp_path), [str(files[0])])
    # Expect at least one output file (e.g., train_loss or val_accuracy)
    assert outputs2
    for p in outputs2.values():
        assert Path(p).exists()


def test_plot_protect_strict_re_raises():
    with pytest.raises(OSError):
        with plot_protect(strict=True):
            raise OSError("boom")


def test_plot_protect_logs_stack_on_debug(caplog):
    caplog.set_level(logging.DEBUG)
    caplog.clear()
    # Invoke a failing protected block and assert debug contains traceback
    with plot_protect(log_on_fail=True):
        raise ValueError("expected")
    assert any(rec.levelname == "DEBUG" for rec in caplog.records)
