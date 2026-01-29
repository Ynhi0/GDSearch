from run_all_kaggle import parse_opt_seed_from_stem


def test_parse_common_patterns():
    cases = [
        ("CIFAR10_ResNet18_Adam_seed42", ("Adam", 42)),
        ("NN_ResNet18_CIFAR10_Adam_lr0.001_seed42", ("Adam", 42)),
        ("NN_ResNet18_CIFAR10_SGD_Momentum_lr0.01_seed1011", ("SGD_Momentum", 1011)),
        ("CIFAR10_ResNet18_AdaBound_seed123", ("AdaBound", 123)),
        ("random_unrecognized_name", (None, None)),
    ]
    for stem, expected in cases:
        opt, seed = parse_opt_seed_from_stem(stem)
        assert (opt, seed) == expected
