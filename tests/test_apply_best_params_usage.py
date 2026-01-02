"""Test to ensure apply_best_params_to_config is present in the canonical tuning-to-final pipeline.

This ensures that Optuna-style best params can be merged into final configs automatically
instead of relying on brittle manual reconstruction.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_apply_best_params_is_used_in_tuning_pipeline():
    # Check script locations where tuning decisions are applied
    candidates = [
        REPO_ROOT / 'scripts' / 'tune_nn.py',
        REPO_ROOT / 'run_all_kaggle.py'
    ]

    found = False
    for p in candidates:
        try:
            text = p.read_text(encoding='utf-8')
        except Exception:
            continue
        if 'apply_best_params_to_config' in text:
            found = True
            break

    assert found, (
        "apply_best_params_to_config() is not used in the canonical tuning pipeline. "
        "Please ensure tuned parameters are programmatically applied to final experiment configs in 'scripts/tune_nn.py' or 'run_all_kaggle.py'."
    )
