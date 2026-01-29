# BD-NSCA PoC — Quick Start

## Local (dev)

1. Create virtualenv & install deps:

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

3. Run the inference server (for manual checks):

```bash
uvicorn integration.inference_server:app --reload --port 8000
# POST JSON to http://localhost:8000/generate
```

## Colab (quick)

1. Open `notebooks/colab_qlora_finetune.ipynb` in Colab.
2. Run the `Setup` cell to install packages.
3. Mount Google Drive if you want to persist artifacts.
4. Replace the QLoRA placeholder cell with a minimal training script (use accelerate config and a GPU runtime).
5. Merge adapter and export model; see `docs/BD-NSCA_report.md` for GGUF notes.

---

This PoC is intentionally minimal and intended for iterative refinement. Always review model outputs and human-annotator feedback before further deployment.