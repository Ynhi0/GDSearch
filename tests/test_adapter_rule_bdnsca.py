from integration.adapter import RuleBasedAdapter
from integration.action_schema import GenerateRequest, AgentState


def test_rule_adapter_detects_combat():
    r = RuleBasedAdapter()
    req = GenerateRequest(id="r1", scenario="patrol", context="we heard gunshots nearby", agent_state=AgentState(health=80, position={"x":0,"y":0}))
    resp = r.generate(req)
    assert resp.intent == "combat"
    assert any(a.action_type.name.lower() == "assess" for a in resp.actions)


def test_rule_adapter_default_patrol():
    r = RuleBasedAdapter()
    req = GenerateRequest(id="r2", scenario="patrol", context="walking the market", agent_state=AgentState(health=80, position={"x":0,"y":0}))
    resp = r.generate(req)
    assert resp.intent == "patrol"
    assert resp.actions[0].action_type.name.lower() == "move"
