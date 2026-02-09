from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from data import coerce_sankey, list_artifacts, load_artifact_payload

REPO_ROOT = Path(__file__).resolve().parents[1]

app = FastAPI(title="Capstone Site Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- API ----------------

@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/results")
def results_index() -> Dict[str, Any]:
    artifacts = list_artifacts()
    return {
        "count": len(artifacts),
        "artifacts": [
            {"key": a.key, "kind": a.kind, "path": str(a.path.relative_to(REPO_ROOT))}
            for a in artifacts
        ],
    }


@app.get("/api/results/{key}")
def results_get(key: str) -> Any:
    payload = load_artifact_payload(key)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Unknown artifact '{key}'")
    return payload


@app.get("/api/sankey/{key}")
def sankey_get(key: str) -> Dict[str, Any]:
    payload = load_artifact_payload(key)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Unknown artifact '{key}'")

    sankey = coerce_sankey(payload)
    if sankey is None:
        raise HTTPException(
            status_code=422,
            detail="Not recognized as sankey data. Expected {nodes:[...], links:[...]} or supported alternates.",
        )
    return sankey


# ---- PROOF: run a tiny model forward pass ----

class RunModelRequest(BaseModel):
    x: list[float]


@app.post("/api/run_model_proof")
def run_model_proof(req: RunModelRequest) -> Dict[str, Any]:
    if len(req.x) == 0:
        raise HTTPException(status_code=400, detail="x must be a non-empty list of floats")

    model = nn.Sequential(
        nn.Linear(len(req.x), 16),
        nn.ReLU(),
        nn.Linear(16, 3),
    )
    model.eval()

    x = torch.tensor(req.x, dtype=torch.float32).unsqueeze(0)

    t0 = time.time()
    with torch.no_grad():
        logits = model(x)
        pred = int(torch.argmax(logits, dim=1).item())
    runtime_ms = int((time.time() - t0) * 1000)

    return {
        "prediction": pred,
        "logits": logits.squeeze(0).tolist(),
        "runtime_ms": runtime_ms,
    }


# ---------------- Static site hosting ----------------

assets_dir = REPO_ROOT / "assets"
if assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

results_dir = REPO_ROOT / "results"
if results_dir.exists():
    app.mount("/results", StaticFiles(directory=str(results_dir)), name="results-static")


@app.get("/")
def root() -> FileResponse:
    index = REPO_ROOT / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="index.html not found at repo root")
    return FileResponse(str(index))


@app.get("/{filename}")
def top_level_files(filename: str) -> FileResponse:
    allowed = {".css", ".js", ".html", ".png", ".jpg", ".jpeg", ".svg", ".ico", ".json"}
    p = REPO_ROOT / filename
    if not p.exists() or not p.is_file() or p.suffix.lower() not in allowed:
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(str(p))


# ---------------- Entrypoint ----------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

