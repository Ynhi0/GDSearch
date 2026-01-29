# GDSearch/tests/test_inference_server.py
import pytest
# Skip if httpx is not available (env may not have httpx installed)
pytest.importorskip("httpx")
from fastapi.testclient import TestClient
from GDSearch.integration import inference_server as srv

client = TestClient(srv.app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_generate_rule_adapter(monkeypatch):
    monkeypatch.setenv("BD_NSCA_ADAPTER", "rule")
    payload = {
        "id": "t1",
        "scenario": "patrol",
        "context": "There is a suspicious footprint near the alley.",
        "agent_state": {"health": 90, "position": {"x": 10, "y": 5}},
        "lang": "en"
    }
    r = client.post("/generate", json=payload)
    assert r.status_code == 200
    j = r.json()
    assert j["id"] == "t1"
    assert "intent" in j
    assert isinstance(j["actions"], list)
