import threading
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.config import Settings
from app.model import ToxicityModel

import redis


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    is_toxic: bool


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(title="Non-Smoker Repo")

    app.state.settings = settings
    app.state.model = ToxicityModel(
        seed=settings.random_seed,
        startup_delay_seconds=settings.model_startup_delay_seconds,
    )
    app.state.ready = False
    app.state.redis_client = None

    def _load_model() -> None:
        app.state.model.load()
        app.state.ready = True

    @app.on_event("startup")
    def on_startup() -> None:
        if settings.redis_url:
            app.state.redis_client = redis.Redis.from_url(
                settings.redis_url, decode_responses=True
            )
        thread = threading.Thread(target=_load_model, daemon=True)
        thread.start()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ready" if app.state.ready else "not ready"}

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest) -> PredictResponse:
        if not app.state.ready:
            raise HTTPException(status_code=503, detail="Model is still loading.")

        is_toxic = app.state.model.predict(payload.text)

        if app.state.redis_client is not None:
            try:
                app.state.redis_client.incr("app_http_inference_count")
            except redis.RedisError:
                # Prediction should still succeed if metrics backend is unavailable.
                pass

        return PredictResponse(is_toxic=is_toxic)

    return app


app = create_app()
