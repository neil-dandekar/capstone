# CB-LLM Backend API Request Spec (Knowledge-Driven)

## Scope

This spec is derived from `codex/knowledge/cb-llm_knowledge.md` and defines the backend request contract independent of the current frontend implementation.

Goal: one primary API request that supports CB-LLM concept intervention for both classification and generation.

## Core Endpoint

`POST /api/v1/run`

- Single endpoint for baseline and intervened runs.
- Supports activation interventions, weight interventions, or both.
- Supports classification and generation in one contract.

---

## User Options That Must Be Available

These options are the product-level controls implied by the CB-LLM knowledge doc.

1. Task mode

- `classification` or `generation`

2. Task context

- `dataset_id`
- `task_id`
- `model_id` or `model_checkpoint`

3. Run type

- `baseline` (no intervention)
- `intervened` (with intervention)
- `compare` (compute baseline and intervened in one request)

4. Intervention mechanism

- `activation` (representation-level)
- `weight` (decision-level / unlearning)
- `hybrid` (both activation and weight)

5. Concept targeting

- Select one or more concepts by stable concept id

6. Activation intervention controls

- Operator per concept: `override`, `add`, `scale`
- Numeric value per concept
- Post-op nonlinearity: `apply_relu_after` (boolean)
- Scope for generation: default static intervention across all decoding steps

7. Weight intervention controls

- Action per concept: `zero`, `scale`
- Scale value when action is `scale`
- Global concept unlearning semantics (concept influence reduced for outputs)

8. Generation decoding controls (generation only)

- `max_tokens`
- `temperature`
- `top_p`
- `seed`

9. Evidence return controls

- Return concept contributions
- Return top concepts
- Return token-time concept traces (optional)

10. Reproducibility metadata

- Echo effective config
- Echo intervention recipe actually used

---

## Request Schema

```json
{
    "request_id": "optional-client-id",
    "run_mode": "compare",
    "context": {
        "task_mode": "classification",
        "dataset_id": "sst2",
        "task_id": "classification",
        "model_id": "cbllm-v1"
    },
    "input": {
        "text": "The movie was surprisingly engaging with strong performances.",
        "true_label": "positive"
    },
    "intervention": {
        "enabled": true,
        "mechanism": "activation",
        "strength_alpha": 0.5,
        "target_concepts": ["positive_sentiment", "uplifting"],
        "activation": {
            "default_apply_relu_after": true,
            "per_concept": {
                "positive_sentiment": {
                    "op": "add",
                    "value": 0.3,
                    "apply_relu_after": true
                },
                "uplifting": {
                    "op": "scale",
                    "value": 1.4
                }
            }
        },
        "weight": {
            "per_concept": {
                "toxic_language": {
                    "action": "zero"
                },
                "politics": {
                    "action": "scale",
                    "value": 0.25
                }
            }
        },
        "generation_policy": {
            "mode": "static",
            "apply_every_step": true
        }
    },
    "generation": {
        "max_tokens": 128,
        "temperature": 0.7,
        "top_p": 0.9,
        "seed": 7
    },
    "response_options": {
        "include_evidence": true,
        "include_top_concepts": true,
        "include_token_time_evidence": false,
        "top_k_concepts": 10
    }
}
```

---

## Field Definitions

### `run_mode`

- Allowed: `baseline | intervened | compare`
- `baseline`: backend must ignore interventions or treat as disabled.
- `intervened`: backend applies interventions and returns intervened result.
- `compare`: backend returns both baseline and intervened results.

### `context.task_mode`

- Allowed: `classification | generation`

### `input`

- `text` required for both modes.
- `true_label` optional (classification evaluation use only).

### `intervention.mechanism`

- Allowed: `activation | weight | hybrid`
- `activation`: use `intervention.activation`
- `weight`: use `intervention.weight`
- `hybrid`: use both

### `intervention.activation.per_concept[*]`

- `op`: `override | add | scale`
- `value`: number
- `apply_relu_after`: optional boolean (falls back to `default_apply_relu_after`)

### `intervention.weight.per_concept[*]`

- `action`: `zero | scale`
- `value`: required when `action=scale`

### `intervention.generation_policy`

- `mode`: `static` (required default)
- `apply_every_step`: boolean, should be `true` for static generation intervention

### `generation` (used only for `task_mode=generation`)

- `max_tokens`: integer > 0
- `temperature`: float >= 0
- `top_p`: float in (0, 1]
- `seed`: integer

---

## Validation Rules

1. Required

- `run_mode`, `context.task_mode`, `context.dataset_id`, `context.task_id`, `input.text`

2. Intervention consistency

- If `run_mode=intervened|compare` and `intervention.enabled=true`, `intervention.mechanism` is required.
- If `mechanism=activation`, at least one activation target is required.
- If `mechanism=weight`, at least one weight target is required.
- If `mechanism=hybrid`, at least one activation or weight target is required.

3. Generation consistency

- For `task_mode=generation`, `generation` block is required.
- For `task_mode=classification`, backend may ignore `generation`.

4. Numeric sanity

- `strength_alpha` recommended range `[0, 1]`
- Reject NaN/inf values in all numeric fields.

---

## Response Contract

```json
{
    "request_id": "optional-client-id",
    "status": "ok",
    "timing_ms": 94,
    "effective_config": {
        "run_mode": "compare",
        "task_mode": "classification",
        "dataset_id": "sst2",
        "task_id": "classification",
        "model_id": "cbllm-v1"
    },
    "baseline": {
        "output": {
            "text": "Pred: positive",
            "classification": {
                "predicted_label": "positive",
                "confidence": 0.93,
                "probabilities": {
                    "negative": 0.07,
                    "positive": 0.93
                },
                "logits": {
                    "negative": -1.02,
                    "positive": 2.14
                }
            },
            "generation": null
        },
        "metrics": {
            "primary_metric": {
                "id": "accuracy",
                "name": "Accuracy",
                "format": "pct",
                "value": 0.92
            },
            "steering_score": 0.1
        },
        "evidence": {
            "concept_contributions": [
                {
                    "concept_id": "positive_sentiment",
                    "concept_name": "positive sentiment",
                    "activation": 0.62,
                    "weight_to_output": 1.31,
                    "contribution": 0.8122
                }
            ],
            "top_concepts": [],
            "token_time": null
        }
    },
    "intervened": {
        "output": {
            "text": "Pred: negative",
            "classification": {
                "predicted_label": "negative",
                "confidence": 0.84,
                "probabilities": {
                    "negative": 0.84,
                    "positive": 0.16
                },
                "logits": {
                    "negative": 1.56,
                    "positive": -0.11
                }
            },
            "generation": null
        },
        "metrics": {
            "primary_metric": {
                "id": "accuracy",
                "name": "Accuracy",
                "format": "pct",
                "value": 0.86
            },
            "steering_score": 0.45
        },
        "evidence": {
            "concept_contributions": [],
            "top_concepts": [],
            "token_time": null
        }
    },
    "warnings": []
}
```

### Output Requirements by Task

1. Classification

- Must return predicted label and confidence.
- Should return probabilities/logits when available.

2. Generation

- Must return generated text.
- Should return token sequence and per-step metadata when available.

---

## Error Contract

For non-2xx:

```json
{
    "error": {
        "code": "INVALID_REQUEST",
        "message": "Invalid intervention payload",
        "details": [
            {
                "field": "intervention.activation.per_concept.positive_sentiment.value",
                "issue": "value must be finite"
            }
        ]
    },
    "request_id": "optional-client-id"
}
```

Recommended status codes:

- `400` malformed or invalid request
- `404` unknown dataset/task/model
- `422` semantically invalid intervention config
- `500` backend inference/runtime failure

---

## Minimal Backend Implementation Strategy

1. Implement `POST /api/v1/run` only.
2. Support `run_mode=compare` so frontend can do one request for baseline+intervened.
3. Start with `generation_policy.mode=static` only.
4. Keep response envelope stable even if some evidence fields are `null` at first.
