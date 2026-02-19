import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn


REPO_ROOT = Path(__file__).resolve().parents[1]
API_DIR = REPO_ROOT / "backend_api"

TASK_MODES = {"classification", "generation"}
RUN_MODES = {"baseline", "intervened", "compare"}
DATASET_ALIASES = {
    "sst2": "SetFit/sst2",
    "setfit/sst2": "SetFit/sst2",
    "SetFit/sst2": "SetFit/sst2",
    "yelp_polarity": "yelp_polarity",
    "ag_news": "ag_news",
    "agnews": "ag_news",
    "dbpedia_14": "dbpedia_14",
}

app = FastAPI(title="CB-LLM Backend API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:5500/",
        "http://127.0.0.1:5500/",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RequestError(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or []
        self.request_id = request_id
        super().__init__(message)


def error_response(err: RequestError) -> JSONResponse:
    return JSONResponse(
        status_code=err.status_code,
        content={
            "error": {
                "code": err.code,
                "message": err.message,
                "details": err.details,
            },
            "request_id": err.request_id,
        },
    )


@app.exception_handler(RequestError)
async def request_error_handler(_, exc: RequestError) -> JSONResponse:
    return error_response(exc)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


def normalize_dataset_id(dataset_id: str, request_id: str | None) -> str:
    normalized = DATASET_ALIASES.get(dataset_id, DATASET_ALIASES.get(dataset_id.lower()))
    if normalized is None:
        raise RequestError(
            status_code=404,
            code="UNKNOWN_DATASET",
            message=f"Unknown dataset_id: {dataset_id}",
            request_id=request_id,
        )
    return normalized


def run_json_command(cmd: list[str], request_id: str | None, error_prefix: str) -> dict[str, Any]:
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False, cwd=REPO_ROOT)
    if proc.returncode != 0:
        raise RequestError(
            status_code=500,
            code="INFERENCE_ERROR",
            message=f"{error_prefix} failed.",
            details=[{"stderr": proc.stderr.strip()}],
            request_id=request_id,
        )
    try:
        return json.loads(proc.stdout.strip())
    except json.JSONDecodeError as exc:
        raise RequestError(
            status_code=500,
            code="INFERENCE_ERROR",
            message=f"{error_prefix} returned non-JSON output.",
            details=[{"stdout": proc.stdout[:1000], "parse_error": str(exc)}],
            request_id=request_id,
        ) from exc


def choose_classification_checkpoint(
    context: dict[str, Any], dataset: str, request_id: str | None
) -> str:
    model_checkpoint = context.get("model_checkpoint")
    if isinstance(model_checkpoint, str) and model_checkpoint.strip():
        return model_checkpoint.strip()

    model_hint = str(context.get("model_id", "")).lower()
    backbone = "gpt2" if "gpt2" in model_hint else "roberta"
    dataset_dir = dataset.replace("/", "_")
    default_cbl = f"mpnet_acs/{dataset_dir}/{backbone}_cbm/cbl_acc.pt"
    fallback_cbl = f"mpnet_acs/{dataset_dir}/{backbone}_cbm/cbl.pt"
    if (REPO_ROOT / "classification" / default_cbl).exists():
        return default_cbl
    if (REPO_ROOT / "classification" / fallback_cbl).exists():
        return fallback_cbl
    raise RequestError(
        status_code=404,
        code="CHECKPOINT_NOT_FOUND",
        message=f"No classification checkpoint found for dataset={dataset}, backbone={backbone}",
        request_id=request_id,
    )


def run_classification(
    payload: dict[str, Any],
    request_id: str | None,
    intervention: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = payload["context"]
    response_options = payload.get("response_options") or {}
    text = payload["input"]["text"]
    dataset = normalize_dataset_id(context["dataset_id"], request_id)
    cbl_path = choose_classification_checkpoint(context, dataset, request_id)
    top_k = max(1, int(response_options.get("top_k_concepts", 10)))

    cmd = [
        sys.executable,
        str(API_DIR / "classification.py"),
        "--text",
        text,
        "--cbl_path",
        cbl_path,
        "--top_k_concepts",
        str(top_k),
    ]

    if isinstance(intervention, dict) and intervention.get("enabled"):
        cmd += ["--intervention_json", json.dumps(intervention)]

    result = run_json_command(cmd, request_id, "Classification inference")
    return {
        "output": {
            "text": f"Pred: {result['predicted_label']}",
            "classification": {
                "predicted_label": result["predicted_label"],
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
                "logits": result["logits"],
            },
            "generation": None,
        },
        "metrics": {
            "primary_metric": {
                "id": "confidence",
                "name": "Confidence",
                "format": "float",
                "value": result["confidence"],
            },
            "steering_score": None,
        },
        "evidence": {
            "concept_contributions": [],
            "top_concepts": result["top_concepts"],
            "token_time": None,
        },
    }



def run_generation(payload: dict[str, Any], request_id: str | None) -> dict[str, Any]:
    context = payload["context"]
    response_options = payload.get("response_options") or {}
    generation_cfg = payload.get("generation") or {}
    text = payload["input"]["text"]
    dataset = normalize_dataset_id(context["dataset_id"], request_id)
    top_k_concepts = max(1, int(response_options.get("top_k_concepts", 10)))

    cmd = [
        sys.executable,
        str(API_DIR / "generation.py"),
        "--prompt",
        text,
        "--dataset",
        dataset,
        "--max_new_tokens",
        str(int(generation_cfg.get("max_tokens", 120))),
        "--temperature",
        str(float(generation_cfg.get("temperature", 0.7))),
        "--top_p",
        str(float(generation_cfg.get("top_p", 0.9))),
        "--top_k_concepts",
        str(top_k_concepts),
    ]
    result = run_json_command(cmd, request_id, "Generation inference")
    return {
        "output": {
            "text": result["generated_text"],
            "classification": None,
            "generation": {
                "generated_text": result["generated_text"],
                "full_text": result["full_text"],
            },
        },
        "metrics": {"primary_metric": None, "steering_score": None},
        "evidence": {
            "concept_contributions": [],
            "top_concepts": result["top_concepts"],
            "token_time": None,
        },
    }


def validate_request(payload: dict[str, Any], request_id: str | None) -> None:
    required_paths = [
        ("run_mode", payload.get("run_mode")),
        ("context", payload.get("context")),
        ("input", payload.get("input")),
    ]
    missing = [name for name, value in required_paths if value in (None, "")]
    if missing:
        raise RequestError(
            status_code=400,
            code="INVALID_REQUEST",
            message="Missing required fields.",
            details=[{"field": field, "issue": "required"} for field in missing],
            request_id=request_id,
        )

    run_mode = payload["run_mode"]
    if run_mode not in RUN_MODES:
        raise RequestError(
            status_code=400,
            code="INVALID_REQUEST",
            message=f"run_mode must be one of {sorted(RUN_MODES)}",
            request_id=request_id,
        )

    context = payload["context"]
    task_mode = context.get("task_mode")
    if task_mode not in TASK_MODES:
        raise RequestError(
            status_code=400,
            code="INVALID_REQUEST",
            message=f"context.task_mode must be one of {sorted(TASK_MODES)}",
            request_id=request_id,
        )
    if not context.get("dataset_id"):
        raise RequestError(
            status_code=400,
            code="INVALID_REQUEST",
            message="context.dataset_id is required",
            request_id=request_id,
        )
    if not context.get("task_id"):
        raise RequestError(
            status_code=400,
            code="INVALID_REQUEST",
            message="context.task_id is required",
            request_id=request_id,
        )

    input_block = payload["input"]
    if not input_block.get("text"):
        raise RequestError(
            status_code=400,
            code="INVALID_REQUEST",
            message="input.text is required",
            request_id=request_id,
        )

    if task_mode == "generation" and not payload.get("generation"):
        raise RequestError(
            status_code=400,
            code="INVALID_REQUEST",
            message="generation block is required for task_mode=generation",
            request_id=request_id,
        )


@app.post("/api/v1/run")
def run_api(payload: dict[str, Any]) -> dict[str, Any]:
    request_id = payload.get("request_id")
    started = time.perf_counter()
    validate_request(payload, request_id)

    task_mode = payload["context"]["task_mode"]
    run_mode = payload["run_mode"]
    warnings: list[str] = []

    if task_mode == "generation":
        raise RequestError(
            status_code=501,
            code="FEATURE_UNAVAILABLE",
            message="Generation is temporarily disabled until model access is configured.",
            request_id=request_id,
        )

    intervention = payload.get("intervention")
    intervention_enabled = bool(isinstance(intervention, dict) and intervention.get("enabled"))

    # Run according to run_mode
    if run_mode == "baseline":
        baseline = run_classification(payload, request_id, intervention=None)
        intervened = None
        intervention_applied = False

    elif run_mode == "intervened":
        if not intervention_enabled:
            warnings.append(
                "run_mode=intervened but intervention.enabled is false; running without intervention."
            )
        baseline = None
        intervened = run_classification(
            payload,
            request_id,
            intervention=intervention if intervention_enabled else None,
        )
        intervention_applied = intervention_enabled

    else:  # compare
        baseline = run_classification(payload, request_id, intervention=None)
        intervened = run_classification(
            payload,
            request_id,
            intervention=intervention if intervention_enabled else None,
        )
        intervention_applied = intervention_enabled

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    response: dict[str, Any] = {
        "request_id": request_id,
        "status": "ok",
        "timing_ms": elapsed_ms,
        "effective_config": {
            "run_mode": run_mode,
            "task_mode": task_mode,
            "dataset_id": normalize_dataset_id(payload["context"]["dataset_id"], request_id),
            "task_id": payload["context"]["task_id"],
            "model_id": payload["context"].get("model_id"),
            "model_checkpoint": payload["context"].get("model_checkpoint"),
            "intervention_applied": intervention_applied,
        },
        "warnings": warnings,
    }

    if baseline is not None:
        response["baseline"] = baseline
    if intervened is not None:
        response["intervened"] = intervened

    return response
