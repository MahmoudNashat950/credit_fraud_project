from unittest import result

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_valid_transaction():
    payload = {
        "V17": -1,
        "V14": -1,
        "V12": -1,
        "V10": -1,
        "V16": -1,
        "V7": -1,
        "V3": -1,
        "V11": -1,
        "V4": -1,
        "V18": -1
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] in [0, 1]
    assert result["result"] in ["Fraud", "Legitimate"]
    assert 0 <= result["fraud_probability"] <= 1
    assert result["threshold"] == 0.8677499218506227

def test_predict_missing_features():
    response = client.post("/predict", json={"V17": -1})

    assert response.status_code == 422


def test_predict_invalid_data_type():
    payload = {
        "V17": "fraud",
        "V14": -1,
        "V12": -1,
        "V10": -1,
        "V16": -1,
        "V7": -1,
        "V3": -1,
        "V11": -1,
        "V4": -1,
        "V18": -1
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422