import pytest
from src.core.optimizers import SAM


def test_sam_requires_adversarial_interface():
    sam = SAM(lr=0.1, rho=0.05, base_optimizer='SGD')
    params = (1.0, 1.0)
    grads = (0.1, 0.2)

    with pytest.raises(RuntimeError):
        sam.step(params, grads)


def test_sam_uses_adversarial_gradients_if_provided():
    sam = SAM(lr=0.1, rho=0.05, base_optimizer='SGD')
    params = (1.0, 1.5)
    grads = (0.1, 0.2)
    adv_grads = (-0.1, -0.2)

    res = sam.step(params, grads, adversarial_gradients=adv_grads)

    # base SGD: new = x - lr * grad
    expected = (params[0] - 0.1 * adv_grads[0], params[1] - 0.1 * adv_grads[1])
    assert pytest.approx(res[0], rel=1e-6) == expected[0]
    assert pytest.approx(res[1], rel=1e-6) == expected[1]


def test_sam_with_loss_fn_compute_adv_gradients():
    sam = SAM(lr=0.1, rho=0.05, base_optimizer='SGD')
    params = (2.0, -1.0)
    grads = (0.2, -0.1)

    # loss_fn returns gradients at adv params; here we just return a constant
    loss_fn = lambda adv_params: (0.05, 0.05)

    res = sam.step(params, grads, loss_fn=loss_fn)

    expected = (params[0] - 0.1 * 0.05, params[1] - 0.1 * 0.05)
    assert pytest.approx(res[0], rel=1e-6) == expected[0]
    assert pytest.approx(res[1], rel=1e-6) == expected[1]
