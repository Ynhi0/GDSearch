import json
from pathlib import Path


def test_no_top_level_seed_in_configs():
    config_files = list(Path('configs').glob('*.json'))
    for p in config_files:
        data = json.loads(p.read_text(encoding='utf-8'))
        assert not (isinstance(data, dict) and 'seed' in data and 'seeds' not in data), f"{p} contains top-level 'seed' key; use 'seeds' list instead"
