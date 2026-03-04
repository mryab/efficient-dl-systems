import time

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


@pytest.mark.integration
def test_health_transitions_from_not_ready_to_ready():
    app = create_app(
        Settings(
            random_seed=7,
            model_startup_delay_seconds=0.4,
        )
    )

    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "not ready"

        time.sleep(0.5)
        response = client.get("/health")
        assert response.json()["status"] == "ready"


@pytest.mark.integration
def test_predict_returns_503_while_model_is_loading():
    app = create_app(
        Settings(
            random_seed=7,
            model_startup_delay_seconds=0.5,
        )
    )

    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "hello"})
        assert response.status_code == 503
        assert "loading" in response.json()["detail"].lower()
