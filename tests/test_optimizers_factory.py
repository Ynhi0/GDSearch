from src.core.optimizers import create_optimizer_instance, SGDMomentum, SGD, Adam


def test_factory_sgdmomentum_alias():
    opt = create_optimizer_instance('SGDMomentum', lr=0.02, beta=0.75)
    assert isinstance(opt, SGDMomentum)
    assert hasattr(opt, 'step')


def test_factory_sgd_alias_lowercase():
    opt = create_optimizer_instance('sgd', lr=0.01)
    assert isinstance(opt, SGD)


def test_factory_adam_params():
    opt = create_optimizer_instance('adam', lr=0.005, beta1=0.8, beta2=0.995)
    assert isinstance(opt, Adam)
