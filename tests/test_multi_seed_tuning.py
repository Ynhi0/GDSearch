import math
from run_all_kaggle import select_best_params_across_seeds


def test_select_best_params_across_seeds_basic():
    per_seed_results = [
        ({'lr': 0.01, 'momentum': 0.9}, 85.0),
        ({'lr': 0.001, 'momentum': 0.9}, 87.0),
        ({'lr': 0.001, 'momentum': 0.9}, 86.5),
    ]

    best_params, mean_val, std_val = select_best_params_across_seeds(per_seed_results)

    assert isinstance(best_params, dict)
    assert best_params == {'lr': 0.001, 'momentum': 0.9}
    assert math.isclose(mean_val, (87.0 + 86.5) / 2, rel_tol=1e-8)
    assert std_val >= 0.0
