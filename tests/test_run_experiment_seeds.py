import pytest
from src.experiments.run_experiment import create_experiment_configs


def test_create_experiment_configs_expands_to_multiple_seeds():
    configs = create_experiment_configs()
    assert len(configs) > 0

    seeds = sorted({c['seed'] for c in configs})
    # Expect at least the default three seeds
    assert len(seeds) >= 3, f"Expected >=3 seeds in expanded configs, found {seeds}"

    # Check at least one experiment_id repeats with different seeds (i.e., we produced per-seed variants)
    base_ids = {}
    for c in configs:
        eid = c.get('experiment_id')
        if eid is None:
            continue
        # strip _seed suffix if present
        if isinstance(eid, str) and '_seed' in eid:
            base = eid.rsplit('_seed', 1)[0]
        else:
            base = eid
        base_ids.setdefault(base, set()).add(c['seed'])

    assert any(len(s) > 1 for s in base_ids.values()), "Expected at least one experiment to have multiple seeds expanded"
