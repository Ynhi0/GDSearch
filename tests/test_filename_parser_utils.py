from src.utils.filename import parse_experiment_filename


def test_parse_patterns_seen_in_repo():
    cases = [
        ("CIFAR10_ResNet18_Adam_seed42", ("Adam", 42, None)),
        ("NN_ResNet18_CIFAR10_Adam_lr0.001_seed42", ("Adam", 42, 0.001)),
        ("NN_ResNet18_CIFAR10_SGD_Momentum_lr0.01_seed1011", ("SGD_Momentum", 1011, 0.01)),
        ("CIFAR10_ResNet18_AdaBound_seed123", ("AdaBound", 123, None)),
        ("random_unrecognized_name", (None, None, None)),
    ]

    for stem, expected in cases:
        parsed = parse_experiment_filename(stem)
        opt, seed, lr = expected
        if opt is None:
            assert parsed['optimizer'] is None
        else:
            assert opt.lower() in parsed['optimizer'].lower()
        assert parsed['seed'] == seed
        if lr is None:
            assert parsed['lr'] is None
        else:
            assert abs(parsed['lr'] - lr) < 1e-8


def test_edge_cases():
    parsed = parse_experiment_filename('UPPERCASE_SGD_SEED123')
    assert parsed['seed'] == 123
    assert parsed['orig'] == 'UPPERCASE_SGD_SEED123'

    parsed = parse_experiment_filename('lr0.01_only')
    assert abs(parsed['lr'] - 0.01) < 1e-9

    parsed = parse_experiment_filename('')
    assert parsed['optimizer'] is None and parsed['seed'] is None and parsed['lr'] is None
