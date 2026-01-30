from types import SimpleNamespace
from src.utils.experiment_config import get_config_from_args


def test_default_resume_behavior_when_resume_flag():
    args = SimpleNamespace(resume=True, resume_behavior=None, seeds='42', experiments='mnist', results_dir='results')
    cfg = get_config_from_args(args)
    assert cfg.resume is True
    assert cfg.resume_behavior == 'skip_if_results_exist'


def test_default_resume_behavior_without_resume_flag():
    args = SimpleNamespace(resume=False, resume_behavior=None, seeds='42', experiments='mnist', results_dir='results')
    cfg = get_config_from_args(args)
    assert cfg.resume is False
    assert cfg.resume_behavior == 'restart_if_no_checkpoint'


def test_override_resume_behavior():
    args = SimpleNamespace(resume=True, resume_behavior='error_if_no_checkpoint', seeds='42', experiments='mnist', results_dir='results')
    cfg = get_config_from_args(args)
    assert cfg.resume_behavior == 'error_if_no_checkpoint'
