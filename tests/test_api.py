from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(autouse=True)
def test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6399/0")


def test_health_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_loaded"] is True
    assert payload["redis_connected"] is True


def test_homepage_served() -> None:
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "Streaming Fraud Detection Engine" in response.text


def test_predict_requires_api_key() -> None:
    request = {
        "user_id": "u1",
        "amount": 120.5,
        "timestamp": "2026-03-30T10:00:00Z",
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=request)

    assert response.status_code == 401


def test_predict_with_history_features() -> None:
    headers = {"x-api-key": "test-key"}

    first_request = {
        "user_id": "u100",
        "amount": 100.0,
        "timestamp": "2026-03-30T10:01:00Z",
    }
    second_request = {
        "user_id": "u100",
        "amount": 220.0,
        "timestamp": "2026-03-30T10:02:00Z",
    }

    with TestClient(app) as client:
        first = client.post("/predict", json=first_request, headers=headers)
        second = client.post("/predict", json=second_request, headers=headers)

    assert first.status_code == 200
    assert second.status_code == 200

    first_payload = first.json()
    second_payload = second.json()

    assert first_payload["features"]["txn_count_last_n"] == 0.0
    assert second_payload["features"]["txn_count_last_n"] >= 1.0
    assert 0.0 <= second_payload["score"] <= 1.0
