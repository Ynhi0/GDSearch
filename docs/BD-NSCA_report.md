# BD-NSCA Proof of Concept Report

## One-Page Summary ✅

This document summarizes a Proof of Concept (PoC) for BD-NSCA (Behavioral Decision - Non-Stationary Contextual Agent), a system for generating structured, safety-aware intents and action plans for in-game NPCs. The PoC focuses on: (1) a bilingual dataset of 200 scenario-driven examples (Vietnamese/English), (2) QLoRA fine-tuning workflow for LLM adapters that run in Colab, (3) a lightweight FastAPI inference server returning intents and action lists ready for integration into UE5, and (4) tests & CI to validate the pipeline. Core findings: a small QLoRA adapter can be trained quickly for constrained behaviors using well-formed JSONL data and safety-aware prompt templates; integration with game engines is straightforward via HTTP endpoints. Key next steps: run a Colab training pass, refine prompt templates with annotator feedback, and integrate the FastAPI endpoint into a UE5 Blueprint using VaRest/FHttpModule.

---

## Project Goals

- Create a minimal end-to-end PoC to go from annotated scenario examples to an inference endpoint that returns structured actions suitable for UE5 consumption.
- Provide bilingual data (EN/VI) to support localization and evaluation by bilingual annotators.
- Provide reproducible training and evaluation scaffolding (QLoRA skeleton, tests, CI, notebook).

## Dataset

- `data/sample_bilingual_200.jsonl` contains 200 balanced examples across five scenarios: `patrol`, `combat`, `escort`, `shopkeeper`, `investigation` (40 per scenario; 20 EN / 20 VI each).
- Each example has: `id`, `scenario`, `lang`, `context`, `agent_state`, `instruction`, `output`, `expected_actions`, and `quality` metadata (`annotator`, `iaa`).

## Modeling approach

- Use QLoRA (Quantized Low-Rank Adapters) with Hugging Face Transformers + PEFT + BitsAndBytes for GPU-efficient fine-tuning in Colab.
- Use instruction-tuning style inputs: combine `context` + `agent_state` + `instruction` to produce `output` (gold) and `expected_actions`.
- Save merged model and provide notes to convert to GGUF for local deployment (e.g., with Ollama) and for offline inference.

## Inference server & integration

- A minimal FastAPI app (`/generate`) serves structured JSON responses with `intent` and `actions` arrays.
- `integration/ue5_adapter.md` shows example UE5 Blueprint HTTP requests (VaRest / FHttpModule) and how to map `actions` to gameplay tasks.

## Evaluation

- Unit tests validate: dataset loading, `--dry-run` training skeleton, and inference response formats (see `tests/`).
- CI workflow runs tests and a notebook smoke runner.

## Limitations & Safety

- PoC models are not safety-audited; do not deploy in production without human review.
- Example dataset avoids PII and violent detail; all outputs are intentionally high-level action plans.

---

## Quick README: Run locally (dev)

1. Create a Python venv and install dev deps:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

2. Run tests:

```bash
pytest -q
```

3. Run the inference server for manual testing:

```bash
uvicorn integration.inference_server:app --reload --port 8000
# then POST to http://localhost:8000/generate
```

## Quick Colab steps

- Open `notebooks/colab_qlora_finetune.ipynb` in Colab, run `Setup` to install dependencies, mount Drive if needed, and run the QLoRA fine-tuning cells. The notebook contains placeholders for long-running steps and explicit instructions for merging adapters and exporting.

---

## Next steps

1. Run a 1-epoch QLoRA in Colab, confirm the merged adapter produces reasonable outputs on held-out examples.
2. Add annotator feedback loop and measurement of IAA on a subset of examples.
3. Integrate the FastAPI endpoint in UE5 via Blueprints and test in a controlled gameplay scenario.

---

*Report generated for BD-NSCA PoC. Contact: project team (internal).*