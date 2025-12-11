import os
import pandas as pd
import numpy as np
from src.analysis.statistical_analysis import extract_final_metric
from src.experiments.run_multi_seed import aggregate_results


def make_eval_df(acc: float, tainted: bool = False):
    # Minimal DataFrame with eval rows
    return pd.DataFrame([
        {'phase': 'eval', 'epoch': 1, 'test_accuracy': acc, 'tainted': tainted}
    ])


def test_extract_final_metric_excludes_tainted():
    df1 = make_eval_df(0.8, tainted=False)
    df2 = make_eval_df(0.5, tainted=True)
    df3 = make_eval_df(0.9, tainted=False)

    vals_include = extract_final_metric([df1, df2, df3], metric='test_accuracy', exclude_tainted=False)
    assert np.allclose(vals_include, np.array([0.8, 0.5, 0.9]))

    vals_exclude = extract_final_metric([df1, df2, df3], metric='test_accuracy', exclude_tainted=True)
    assert np.allclose(vals_exclude, np.array([0.8, 0.9]))


def test_aggregate_results_excludes_tainted(tmp_path):
    from src.experiments.run_multi_seed import aggregate_results

    df1 = make_eval_df(0.8, tainted=False)
    df2 = make_eval_df(0.5, tainted=True)
    df3 = make_eval_df(0.9, tainted=False)

    p1 = tmp_path / "run1.csv"
    p2 = tmp_path / "run2.csv"
    p3 = tmp_path / "run3.csv"
    df1.to_csv(str(p1), index=False)
    df2.to_csv(str(p2), index=False)
    df3.to_csv(str(p3), index=False)

    agg_include = aggregate_results([str(p1), str(p2), str(p3)], metric='test_accuracy', exclude_tainted=False)
    assert np.isclose(agg_include['mean'], np.mean([0.8, 0.5, 0.9]))
    assert agg_include['n'] == 3

    agg_exclude = aggregate_results([str(p1), str(p2), str(p3)], metric='test_accuracy', exclude_tainted=True)
    assert np.isclose(agg_exclude['mean'], np.mean([0.8, 0.9]))
    assert agg_exclude['n'] == 2
