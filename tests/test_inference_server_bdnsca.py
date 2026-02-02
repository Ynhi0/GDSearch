import pytest
pytest.importorskip("httpx")  # Skip these tests if httpx (TestClient runtime dependency) is not installed

from fastapi.testclient import TestClient
from integration.inference_server import app
from integration.action_schema import GenerateRequest, AgentState

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_generate_rule_adapter():
    req = {
        "id": "s1",
        "scenario": "patrol",
        "context": "suspicious evidence found near crate",
        "agent_state": {"health": 90, "position": {"x":10, "y":10}},
        "lang": "en",
    }
    r = client.post("/generate", json=req)
    assert r.status_code == 200
    data = r.json()
    assert data["id"] == "s1"
    assert "intent" in data
    assert "actions" in data
