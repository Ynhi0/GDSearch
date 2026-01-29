# UE5 Adapter (Blueprints) — HTTP Integration Guide

This short guide shows how to call the BD-NSCA FastAPI `/generate` endpoint from Unreal Engine 5 using common Blueprint plugins (VaRest) or C++ (FHttpModule).

## Example JSON request

POST http://<HOST>:8000/generate

Body (JSON):
```json
{
  "id": "ex-0001",
  "scenario": "patrol",
  "context": "Two civilians reported; suspicious footprints near the alley.",
  "agent_state": {"health": 95, "position": {"x": 124, "y": 42}},
  "lang": "en"
}
```

## Using VaRest (Blueprint)
1. Add VaRest plugin to your project.
2. Create a `VaRestRequestJSON` component and set URL to the `/generate` endpoint.
3. Set `Verb` to POST and `ContentType` to `application/json`.
4. Fill JSON fields from your game state (ID, context, agent_state).
5. Call `Execute` and bind to `OnRequestComplete`.
6. Parse returned JSON: get `intent` and iterate `actions` to dispatch gameplay tasks.

## Using FHttpModule (C++)
- Create an `FHttpModule` request, set headers (`Content-Type: application/json`) and `SetVerb("POST")`, `SetURL(...)`.
- Fill request body with the JSON above and call `ProcessRequest()`.
- In the response handler, parse the JSON and map `actions` to gameplay behaviors using your AI controller or behavior tree tasks.

## Mapping actions to gameplay
- `move`: call AIController->MoveToLocation(params.dest)
- `escort`: create follow behavior and condition-checks to maintain distance
- `engage`: set combat mode on and call appropriate attack tasks (high-level, no weapon specifics)
- `interact`: open dialogue or trade UI

---

> Note: Keep the server endpoint reachable from the game environment and validate all returned actions in-game before executing. Use secure channels for production; this PoC uses HTTP in trusted test networks only.