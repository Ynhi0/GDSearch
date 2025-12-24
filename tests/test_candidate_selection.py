import json
from run_all_kaggle import choose_best_candidate_from_eval


def test_choose_best_candidate_from_eval_basic():
    # Three candidates with these mean accuracies: p1=80.5, p2=84.0, p3=84.5 -> pick p3
    cand_eval = {
        json.dumps({'lr': 0.01}, sort_keys=True): [80.0, 81.0],
        json.dumps({'lr': 0.001}, sort_keys=True): [78.0, 90.0],
        json.dumps({'lr': 0.005}, sort_keys=True): [85.0, 84.0],
    }

    best_params, mean_val, std_val = choose_best_candidate_from_eval(cand_eval)

    assert best_params == {'lr': 0.005}
    assert abs(mean_val - 84.5) < 1e-6
    assert std_val >= 0.0
