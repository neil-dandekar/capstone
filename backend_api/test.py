from fastapi.testclient import TestClient

import backend_api.main as api


def _classification_result() -> dict:
    return {
        "output": {
            "text": "Pred: positive",
            "classification": {
                "predicted_label": "positive",
                "confidence": 0.99,
                "probabilities": {"negative": 0.01, "positive": 0.99},
                "logits": {"negative": -2.0, "positive": 2.0},
            },
            "generation": None,
        },
        "metrics": {
            "primary_metric": {
                "id": "confidence",
                "name": "Confidence",
                "format": "float",
                "value": 0.99,
            },
            "steering_score": None,
        },
        "evidence": {
            "concept_contributions": [],
            "top_concepts": [{"concept_id": "positive_sentiment", "activation": 1.2}],
            "token_time": None,
        },
    }


def _base_payload(run_mode: str = "baseline") -> dict:
    return {
        "request_id": "test-1",
        "run_mode": run_mode,
        "context": {
            "task_mode": "classification",
            "dataset_id": "sst2",
            "task_id": "classification",
            "model_id": "roberta",
        },
        "input": {"text": "A warm and engaging movie."},
        "intervention": {
            "enabled": True,
            "mechanism": "activation",
            "target_concepts": ["positive_sentiment"],
        },
        "response_options": {"top_k_concepts": 5},
    }


def test_healthz() -> None:
    client = TestClient(api.app)
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_validation_error_missing_text() -> None:
    client = TestClient(api.app)
    payload = _base_payload()
    payload["input"] = {}
    resp = client.post("/api/v1/run", json=payload)
    body = resp.json()
    assert resp.status_code == 400
    assert body["error"]["code"] == "INVALID_REQUEST"
    assert "input.text" in body["error"]["message"]


def test_classification_baseline(monkeypatch) -> None:
    client = TestClient(api.app)
    monkeypatch.setattr(api, "run_classification", lambda *_: _classification_result())

    resp = client.post("/api/v1/run", json=_base_payload("baseline"))
    body = resp.json()

    assert resp.status_code == 200
    assert body["status"] == "ok"
    assert "baseline" in body
    assert "intervened" not in body
    assert body["effective_config"]["intervention_applied"] is False
    assert "ignored" in body["warnings"][0].lower()


def test_classification_compare(monkeypatch) -> None:
    client = TestClient(api.app)
    monkeypatch.setattr(api, "run_classification", lambda *_: _classification_result())

    resp = client.post("/api/v1/run", json=_base_payload("compare"))
    body = resp.json()

    assert resp.status_code == 200
    assert "baseline" in body and "intervened" in body
    assert body["baseline"]["output"]["classification"]["predicted_label"] == "positive"
    assert body["intervened"]["output"]["classification"]["predicted_label"] == "positive"
    assert any("identical results" in w for w in body["warnings"])


def test_generation_disabled() -> None:
    client = TestClient(api.app)
    payload = _base_payload("baseline")
    payload["context"]["task_mode"] = "generation"
    payload["context"]["task_id"] = "generation"
    payload["generation"] = {"max_tokens": 20, "temperature": 0.7, "top_p": 0.9}

    resp = client.post("/api/v1/run", json=payload)
    body = resp.json()

    assert resp.status_code == 501
    assert body["error"]["code"] == "FEATURE_UNAVAILABLE"
