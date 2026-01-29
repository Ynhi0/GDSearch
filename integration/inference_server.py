# GDSearch/integration/inference_server.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from .action_schema import GenerateRequest, GenerateResponse
from .adapter import RuleBasedAdapter, RemoteHTTPAdapter, LocalOllamaAdapter, ModelAdapter

logger = logging.getLogger("bd-nsca")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="BD-NSCA Inference Server")

# Simple CORS for UE5 clients in dev - lock down in prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

_MODEL_ADAPTER = os.environ.get("BD_NSCA_ADAPTER", "rule").lower()
_MODEL_URL = os.environ.get("BD_NSCA_MODEL_URL", "http://localhost:8001")
_OLLAMA_MODEL = os.environ.get("BD_NSCA_OLLAMA_MODEL", "local-model")

_adapter: ModelAdapter

@app.on_event("startup")
def _init_adapter():
    global _adapter
    if _MODEL_ADAPTER == "rule":
        _adapter = RuleBasedAdapter()
        logger.info("Using RuleBasedAdapter")
    elif _MODEL_ADAPTER == "remote":
        _adapter = RemoteHTTPAdapter(_MODEL_URL)
        logger.info("Using RemoteHTTPAdapter -> %s", _MODEL_URL)
    elif _MODEL_ADAPTER == "ollama":
        _adapter = LocalOllamaAdapter(model_name=_OLLAMA_MODEL)
        logger.info("Using LocalOllamaAdapter -> %s", _OLLAMA_MODEL)
    else:
        logger.warning("Unknown adapter '%s', falling back to rule-based", _MODEL_ADAPTER)
        _adapter = RuleBasedAdapter()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/meta")
def meta():
    return {"adapter": _MODEL_ADAPTER, "model_url": _MODEL_URL, "ollama_model": _OLLAMA_MODEL}

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    try:
        resp = _adapter.generate(req)
        return resp
    except Exception as exc:
        logger.exception("Failed to generate: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))