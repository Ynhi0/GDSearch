import numpy as np
from src.analysis.statistical_analysis import cohens_d_ci_paired


def test_cohens_d_ci_does_not_mutate_global_rng():
    # Capture RNG state before and after the function to ensure it is not modified
    np.random.seed(123)
    state_before = np.random.get_state()

    cohens_d_ci_paired([1.0, 2.0, 3.0], [1.0, 2.0, 4.0], n_bootstrap=10)

    state_after = np.random.get_state()

    # Compare RNG states element-wise (arrays require array_equal)
    assert len(state_before) == len(state_after)
    for a, b in zip(state_before, state_after):
        if isinstance(a, (list, tuple)):
            # nested structures: compare recursively
            assert a == b
        else:
            try:
                import numpy as _np
                if hasattr(a, 'shape'):
                    assert _np.array_equal(a, b)
                else:
                    assert a == b
            except Exception:
                assert a == b
