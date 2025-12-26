import os
import tempfile
import json
import pytest
from scripts.tune_nn import tune_optimizer


def test_tune_optimizer_missing_metadata_raises(tmp_path, monkeypatch):
    # Create minimal base and spec
    base = {'dataset':'MNIST','model':'SimpleMLP','seed':42,'batch_size':32}
    spec = {'optimizer':'Adam', 'lr_values':[1e-3], 'epochs':1}

    # Monkeypatch run_and_save to write only CSV and not _meta.json
    def fake_run_and_save(cfg, tag):
        out = str(tmp_path / "fake_result.csv")
        # Write CSV with minimal columns
        with open(out, 'w', encoding='utf-8') as f:
            f.write('phase,train_loss\ntrain,0.5\n')
        return out, None

    monkeypatch.setattr('scripts.tune_nn.run_and_save', fake_run_and_save)

    # Run tune_optimizer which should try to read meta and then raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        tune_optimizer(base, spec)
