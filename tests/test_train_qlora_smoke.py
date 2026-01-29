import json
import sys
import subprocess
from pathlib import Path

def test_train_qlora_dry_run(tmp_path):
    # create a tiny jsonl dataset
    data = [
        {"id":"t-001","scenario":"patrol","lang":"en","context":"test","agent_state":{"health":100,"position":{"x":0,"y":0}},"instruction":"test","output":"ok","expected_actions":[],"quality":{"annotator":"a","iaa":"pending"}},
        {"id":"t-002","scenario":"combat","lang":"vi","context":"thử","agent_state":{"health":100,"position":{"x":1,"y":1}},"instruction":"thử","output":"ok","expected_actions":[],"quality":{"annotator":"a","iaa":"pending"}}
    ]
    p = tmp_path / "mini.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for obj in data:
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    proc = subprocess.run([sys.executable, "scripts/train_qlora.py", "--data-path", str(p), "--output-dir", str(out_dir), "--dry-run"], capture_output=True, text=True)
    assert proc.returncode == 0
    assert "Dry run: validating inputs" in proc.stdout
    assert "Loaded 2 examples" in proc.stdout
