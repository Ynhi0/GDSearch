# BD-NSCA Integration — Quickstart

This document explains how to run the BD-NSCA inference server and call it from the NPC AI client.

Quick steps:

1. Start the server locally:
   - `uvicorn integration.inference_server:app --reload --port 8000`
2. Run tests (fast):
   - `pytest tests/test_*bdnsca*.py -q`
3. Call from NPC AI client:
   - `python ../NPC AI/scripts/call_bd_nsca.py --context "guard the plaza"`

Notes:
- The server uses a rule-based adapter by default (`BD_NSCA_ADAPTER=rule`).
- The sample bilingual dataset is in `data/sample_bilingual_200.jsonl` (GDSearch) and a small subset lives in `NPC AI/data/` for integration demos.
