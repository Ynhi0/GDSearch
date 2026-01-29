import logging
import pytest
from src.utils.experiment_config import ExperimentConfig


def test_seed_alias_migration():
    cfg = ExperimentConfig.from_dict({'seed': 42})
    assert cfg.seeds == [42]


def test_seed_list_warns_on_too_few(caplog):
    caplog.set_level(logging.WARNING)
    cfg = ExperimentConfig.from_dict({'seeds': [7]})
    assert cfg.seeds == [7]
    assert any('fewer than 3 seeds' in r.message for r in caplog.records)
