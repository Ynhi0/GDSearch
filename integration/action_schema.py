# GDSearch/integration/action_schema.py
from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, validator

class ActionType(str, Enum):
    MOVE = "move"
    ASSESS = "assess"
    ENGAGE = "engage"
    TAKE_COVER = "take_cover"
    ESCORT = "escort"
    INTERACT = "interact"
    SCAN = "scan"
    COLLECT_EVIDENCE = "collect_evidence"
    PATROL = "patrol"

class AgentState(BaseModel):
    health: int = Field(..., ge=0, le=100)
    position: Dict[str, int]  # x,y ints

    @validator("position")
    def validate_pos(cls, v):
        if "x" not in v or "y" not in v:
            raise ValueError("position must contain 'x' and 'y' keys")
        return v

class GenerateRequest(BaseModel):
    id: str
    scenario: str
    context: str
    agent_state: AgentState
    lang: str = "en"

class ActionItem(BaseModel):
    action_type: ActionType
    params: Dict[str, Any] = {}

class GenerateResponse(BaseModel):
    id: str
    intent: str
    actions: List[ActionItem]
