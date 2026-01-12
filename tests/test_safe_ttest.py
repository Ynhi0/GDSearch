import numpy as np
from src.analysis.statistical_analysis import safe_ttest_rel


def test_safe_ttest_rel_returns_numbers():
    a = np.array([0.1, 0.2, 0.15, 0.18])
    b = np.array([0.11, 0.19, 0.14, 0.17])

    t, p = safe_ttest_rel(a, b)

    assert isinstance(t, float)
    assert isinstance(p, float)
    # p should be in [0,1] or NaN
    assert (0.0 <= p <= 1.0) or np.isnan(p)
