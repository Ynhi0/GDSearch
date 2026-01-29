import sys
import subprocess
from pathlib import Path
import pytest
import json

# Skip network-dependent FastAPI TestClient tests if httpx is not installed in this env
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from integration.inference_server import app
from scripts.train_qlora import load_data


def test_sample_dataset_loads():
    p = Path("data/sample_bilingual_200.jsonl")
    data = load_data(str(p))
    assert isinstance(data, list)
    assert len(data) == 200


def test_train_cli_dry_run():
    proc = subprocess.run([sys.executable, "scripts/train_qlora.py", "--data-path", "data/sample_bilingual_200.jsonl", "--output-dir", "./tmp_out", "--dry-run"], capture_output=True, text=True)
    assert proc.returncode == 0
    assert "Dry run" in proc.stdout


def test_inference_server_response():
    client = TestClient(app)
    req = {
        "id": "test-req-1",
        "scenario": "patrol",
        "context": "suspicious footprints near alley",
        "agent_state": {"health": 100, "position": {"x": 10, "y": 10}},
        "lang": "en"
    }
    resp = client.post("/generate", json=req)
    assert resp.status_code == 200
    payload = resp.json()
    assert payload.get("id") == req["id"]
    assert "intent" in payload
    assert isinstance(payload.get("actions"), list)
