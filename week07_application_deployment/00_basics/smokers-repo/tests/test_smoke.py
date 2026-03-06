from fastapi.testclient import TestClient

from app import app


def test_predict_returns_boolean_shape():
    client = TestClient(app)
    response = client.post("/predict", json={"text": "you are nice"})
    assert response.status_code == 200
    assert "is_toxic" in response.json()
