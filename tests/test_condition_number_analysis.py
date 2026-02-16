import numpy as np
from src.analysis.condition_number_analysis import logistic_regression_with_condition_number


def test_logistic_regression_with_condition_number_loss_and_grad():
    loss_fn, grad_fn, meta = logistic_regression_with_condition_number(
        kappa=5.0, n_samples=400, n_features=10, seed=1
    )

    w_true = meta['w_true']
    # Loss at the true weight should be lower than at a random weight
    loss_true = loss_fn(w_true)
    rng = np.random.RandomState(0)
    loss_rand = loss_fn(rng.randn(*w_true.shape))
    assert loss_true < loss_rand

    # Gradient should match finite differences (numerical tolerance)
    w = rng.randn(*w_true.shape)
    analytic = grad_fn(w)
    eps = 1e-6
    numeric = np.zeros_like(w)
    for i in range(w.size):
        wp = w.copy(); wp[i] += eps
        wm = w.copy(); wm[i] -= eps
        numeric[i] = (loss_fn(wp) - loss_fn(wm)) / (2 * eps)

    assert np.allclose(analytic, numeric, rtol=1e-3, atol=1e-4)
