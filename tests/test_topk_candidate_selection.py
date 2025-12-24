import os
import json
from run_all_kaggle import choose_best_candidate_from_eval


def test_topk_selection_behaviour():
    # Simulate candidate eval results for three candidates
    cand_eval = {
        json.dumps({'lr': 0.01}, sort_keys=True): [80.0, 82.0],
        json.dumps({'lr': 0.001}, sort_keys=True): [87.0, 86.0],
        json.dumps({'lr': 0.005}, sort_keys=True): [84.0, 85.0],
    }

    best_params, mean_val, std_val = choose_best_candidate_from_eval(cand_eval)
    assert best_params == {'lr': 0.001}
    assert abs(mean_val - 86.5) < 1e-6
    assert std_val >= 0.0
