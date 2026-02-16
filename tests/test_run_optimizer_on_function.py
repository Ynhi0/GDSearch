import numpy as np
from src.analysis.condition_number_analysis import run_optimizer_on_function, quadratic_with_condition_number
from src.core.optimizers import Adam


def test_run_optimizer_on_function_uses_canonical_adam():
    # Simple 2D quadratic for deterministic behavior
    f, grad_f, meta = quadratic_with_condition_number(kappa=2.0, n_dims=2, random_rotation=False, seed=0)
    x0 = np.array([1.23, -0.75], dtype=float)

    opt_cfg = {'type': 'adam', 'lr': 0.1, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8}

    # Run the high-level runner
    out = run_optimizer_on_function(f, grad_f, x0.copy(), opt_cfg, n_iterations=5, metadata=meta)

    # Run manual Adam steps using canonical Adam implementation
    adam = Adam(lr=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8)
    x = x0.copy()
    manual_losses = []
    for _ in range(5):
        manual_losses.append(f(x))
        g = grad_f(x)
        x = adam.step(x, g)

    # Compare final parameter and recorded losses
    assert np.allclose(out['final_x'], x, rtol=1e-6, atol=1e-8)
    assert np.allclose(out['losses'], np.array(manual_losses), rtol=1e-6, atol=1e-9)
