from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["model"] == "naive_bayes"


def test_prediction_endpoint_email_classification():
    response = client.post(
        "/api/v1/predict",
        json={"text": "Please submit the project report before Friday"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]["model"] == "naive_bayes"
    assert payload["data"]["prediction"] == "work"
    assert payload["data"]["probability"] == 0.87
