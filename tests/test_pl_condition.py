import numpy as np
from src.analysis.pl_condition import pl_mu_estimate, pl_holds_at_point


def test_pl_estimate_and_check():
    loss = 1.0
    grad_norm_sq = 4.0
    mu = pl_mu_estimate(loss, grad_norm_sq, f_star=0.0)
    # mu_hat = 4 / (2 * 1) = 2
    assert abs(mu - 2.0) < 1e-8
    assert pl_holds_at_point(loss, grad_norm_sq, mu=2.0)
    assert not pl_holds_at_point(loss, grad_norm_sq, mu=3.0)
