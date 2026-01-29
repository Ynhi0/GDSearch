import pytest
from src.utils.filename import parse_opt_seed_from_stem


def test_standard_cases():
    cases = [
        ("CIFAR10_ResNet18_Adam_seed42", ("Adam", 42)),
        ("NN_ResNet18_CIFAR10_Adam_lr0.001_seed42", ("Adam", 42)),
        ("CIFAR10_ResNet18_AdaBound_seed1011", ("AdaBound", 1011)),
        ("cifar10-resnet18-adam-seed42", ("adam", 42)),
        ("CIFAR10_ResNet18_adamw_seed_123", ("adamw", 123)),
        ("experiment_Adam_lr0.1", ("Adam", None)),
        ("nooptimizer_seed", (None, None)),
        ("randomname", (None, None)),
        ("ResNet_SGD_seed00042", ("SGD", 42)),
        ("ResNet18-Adam-seed42-extra", ("Adam", 42)),
    ]

    for stem, expected in cases:
        opt, seed = parse_opt_seed_from_stem(stem)
        exp_opt, exp_seed = expected
        # allow case-insensitive matching for optimizer token
        if exp_opt is None:
            assert opt is None
        else:
            assert opt is not None and opt.lower() == exp_opt.lower()
        if exp_seed is None:
            assert seed is None
        else:
            assert seed == exp_seed


def test_malformed_and_edge_cases():
    cases = [
        ("seed", (None, None)),
        ("seed_", (None, None)),
        ("seed42", (None, 42)),
        ("_seed42", (None, 42)),
        ("Adam_seed_notanumber", ("Adam", None)),
    ]
    for stem, expected in cases:
        opt, seed = parse_opt_seed_from_stem(stem)
        exp_opt, exp_seed = expected
        if exp_opt is None:
            assert opt is None
        else:
            assert opt is not None and opt.lower() == exp_opt.lower()
        if exp_seed is None:
            assert seed is None
        else:
            assert seed == exp_seed
