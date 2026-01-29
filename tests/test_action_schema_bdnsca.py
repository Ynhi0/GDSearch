from integration.action_schema import AgentState, GenerateRequest, ActionType, ActionItem, GenerateResponse
import pytest


def test_agent_state_validation():
    with pytest.raises(ValueError):
        AgentState(health=50, position={"x": 1})

    s = AgentState(health=100, position={"x": 1, "y": 2})
    assert s.health == 100


def test_generate_response_shape():
    req = GenerateRequest(id="t1", scenario="patrol", context="look for suspicious", agent_state=AgentState(health=90, position={"x":0,"y":0}), lang="en")
    resp = GenerateResponse(id="t1", intent="patrol", actions=[ActionItem(action_type=ActionType.MOVE, params={"route":"default"})])
    assert resp.id == "t1"
    assert isinstance(resp.actions[0].action_type, ActionType)
