# GDSearch/integration/adapter.py
import os
import json
import subprocess
import requests
from typing import Any
from .action_schema import GenerateRequest, GenerateResponse, ActionItem, ActionType

class ModelAdapter:
    """Base class: subclass & implement generate(req)."""
    def generate(self, req: GenerateRequest) -> GenerateResponse:
        raise NotImplementedError()

class RuleBasedAdapter(ModelAdapter):
    """Existing PoC rule-based behavior (deterministic)."""
    def generate(self, req: GenerateRequest) -> GenerateResponse:
        ctx = req.context.lower()
        actions = []
        intent = "observe"
        if any(k in ctx for k in ["enemy", "threat", "gun"]):
            intent = "combat"
            actions = [
                ActionItem(action_type=ActionType.ASSESS, params={}),
                ActionItem(action_type=ActionType.ENGAGE, params={"mode": "defensive"}),
                ActionItem(action_type=ActionType.TAKE_COVER, params={}),
            ]
        elif any(k in ctx for k in ["escort", "vip", "protect"]):
            intent = "escort"
            actions = [
                ActionItem(action_type=ActionType.MOVE, params={"destination": "waypoint_alpha"}),
                ActionItem(action_type=ActionType.ESCORT, params={"target_id": "vip_1"}),
            ]
        elif any(k in ctx for k in ["shop", "merchant", "trade"]):
            intent = "interact"
            actions = [ActionItem(action_type=ActionType.INTERACT, params={"type": "trade"})]
        elif any(k in ctx for k in ["suspicious", "evidence", "clue"]):
            intent = "investigate"
            actions = [
                ActionItem(action_type=ActionType.SCAN, params={}),
                ActionItem(action_type=ActionType.COLLECT_EVIDENCE, params={}),
            ]
        else:
            intent = "patrol"
            actions = [ActionItem(action_type=ActionType.MOVE, params={"route": "default_patrol"})]

        return GenerateResponse(id=req.id, intent=intent, actions=actions)

class RemoteHTTPAdapter(ModelAdapter):
    """Call a remote inference HTTP server that returns BD-NSCA schema."""
    def __init__(self, base_url: str, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, req: GenerateRequest) -> GenerateResponse:
        url = f"{self.base_url}/generate"
        payload = req.dict()
        r = requests.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return GenerateResponse(**data)

class LocalOllamaAdapter(ModelAdapter):
    """Example: call `ollama` CLI or other local runtime. Returns parsed JSON actions.

    NOTE: This is intentionally minimal and robustly guarded.
    """
    def __init__(self, model_name: str = "local-model", cmd: str = "ollama", timeout: float = 15.0):
        self.model_name = model_name
        self.cmd = cmd
        self.timeout = timeout

    def generate(self, req: GenerateRequest) -> GenerateResponse:
        prompt = json.dumps(req.dict(), ensure_ascii=False)
        try:
            proc = subprocess.run([self.cmd, "run", self.model_name, "--prompt", prompt],
                                  capture_output=True, text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            raise RuntimeError("local model execution timed out")
        if proc.returncode != 0:
            raise RuntimeError(f"local model failed: {proc.stderr[:100]}")
        out = proc.stdout.strip()
        try:
            data = json.loads(out)
            return GenerateResponse(**data)
        except Exception:
            intent = "unknown"
            actions = []
            lines = out.splitlines()
            if lines:
                intent = lines[0].strip()
                for l in lines[1:]:
                    parts = l.split(":", 1)
                    if parts:
                        typ = parts[0].strip().lower()
                        params = {}
                        if len(parts) > 1:
                            try:
                                params = json.loads(parts[1])
                            except Exception:
                                params = {"raw": parts[1].strip()}
                        atype = ActionType.MOVE
                        if "engage" in typ:
                            atype = ActionType.ENGAGE
                        elif "move" in typ:
                            atype = ActionType.MOVE
                        actions.append(ActionItem(action_type=atype, params=params))
            return GenerateResponse(id=req.id, intent=intent, actions=actions)
