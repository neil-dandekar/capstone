# Backend API Implementation (Current)

This repo now includes a minimal backend at:

- `/Users/neildandekar/Documents/uc-san-diego/dsc180ab/capstone-backend/backend_api/main.py`

It implements:

- `POST /api/v1/run`
- `GET /healthz`

## Current Behavior

- Accepts the request envelope described in `backend_api_request_spec.md`.
- Validates required fields (`run_mode`, `context`, `input.text`, etc.).
- Accepts `intervention` payloads but does not apply them yet.
- Runs model inference through API-local scripts:
  - classification: `/Users/neildandekar/Documents/uc-san-diego/dsc180ab/capstone-backend/backend_api/classification.py`
  - generation: `/Users/neildandekar/Documents/uc-san-diego/dsc180ab/capstone-backend/backend_api/generation.py`
- Returns structured response with `baseline`/`intervened` depending on `run_mode`.

For now, `run_mode=compare` returns identical baseline/intervened outputs and a warning because intervention is intentionally ignored.

## Run Locally

From repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend_api/requirements.txt
uvicorn backend_api.main:app --host 0.0.0.0 --port 8000 --reload
```

If your venv is Python 3.13, the backend requirements will install a newer compatible `torch` automatically.
If you want exact paper-era pins (`torch==2.4.0`), create the venv with Python 3.12 instead.

## Example Request (Intervention Included but Ignored)

```json
{
  "request_id": "demo-001",
  "run_mode": "compare",
  "context": {
    "task_mode": "classification",
    "dataset_id": "sst2",
    "task_id": "classification",
    "model_id": "roberta"
  },
  "input": {
    "text": "The movie was surprisingly engaging."
  },
  "intervention": {
    "enabled": true,
    "mechanism": "activation",
    "target_concepts": ["positive_sentiment"],
    "activation": {
      "per_concept": {
        "positive_sentiment": {
          "op": "add",
          "value": 0.3
        }
      }
    }
  },
  "response_options": {
    "include_top_concepts": true,
    "top_k_concepts": 8
  }
}
```

## Minimal curl test

```bash
curl -X POST http://localhost:8000/api/v1/run \
  -H "Content-Type: application/json" \
  -d '{
    "request_id":"demo-001",
    "run_mode":"baseline",
    "context":{"task_mode":"classification","dataset_id":"sst2","task_id":"classification","model_id":"roberta"},
    "input":{"text":"A touching and funny movie."},
    "intervention":{"enabled":true,"mechanism":"activation","target_concepts":["positive_sentiment"]},
    "response_options":{"top_k_concepts":5}
  }'
```
