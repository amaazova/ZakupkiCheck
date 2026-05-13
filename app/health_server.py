"""FastAPI sidecar exposing /healthz on :8080 next to Streamlit."""
from __future__ import annotations

import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI

APP_VERSION = "2.0.0"
LR_MODEL_PATH = Path(__file__).parent / "models" / "lr_model.joblib"

app = FastAPI(title="ZakupkiCheck health", version=APP_VERSION)


@app.get("/healthz")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "version": APP_VERSION,
        "lr_model_present": LR_MODEL_PATH.is_file(),
        "model": os.environ.get("DEFAULT_MODEL", "deepseek/deepseek-v4-flash"),
    }


@app.get("/livez")
def live() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.environ.get("HEALTH_PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
