import statistics
import time

from fastapi.testclient import TestClient

from main import app
from schemas import FEATURE_COLUMNS

client = TestClient(app)


def make_payload() -> dict[str, float]:
    payload = dict.fromkeys(FEATURE_COLUMNS, 0.0)
    payload["Amount"] = 42.0
    return payload


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_happy_path() -> None:
    response = client.post("/predict", json=make_payload())

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["fraud_probability"], float)
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert isinstance(body["is_fraud"], bool)


def test_predict_missing_required_field() -> None:
    payload = make_payload()
    payload.pop("Amount")

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_invalid_amount_type() -> None:
    payload = make_payload()
    payload["Amount"] = "abc"

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_invalid_feature_vector_size() -> None:
    response = client.post("/predict", json={"features": [0.0, 1.0]})

    assert response.status_code == 422
    assert "features" in response.text


def test_predict_latency_median_under_200_ms() -> None:
    payload = make_payload()
    durations = []

    for _ in range(100):
        start = time.perf_counter()
        response = client.post("/predict", json=payload)
        durations.append((time.perf_counter() - start) * 1000)
        assert response.status_code == 200

    assert statistics.median(durations) < 200
