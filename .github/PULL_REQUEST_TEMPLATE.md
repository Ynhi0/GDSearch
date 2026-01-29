## Summary

Short description of what this PR adds for BD-NSCA integration.

## Changes
- Files added/modified (server, schema, tests, docs)

## Acceptance criteria ✅
- [ ] Unit tests for schema pass
- [ ] Integration endpoint returns valid schema
- [ ] README/integration doc updated
- [ ] CI workflow added to run BD-NSCA smoke tests

## How to test locally
1. Start server: `uvicorn integration.inference_server:app --reload --port 8000`
2. Run tests: `pytest tests/test_*bdnsca*.py -q`

> Known unrelated failing tests: `tests/test_integration_quick_pipeline.py` (unrelated to BD-NSCA CI). Do not block this PR on those failures.
