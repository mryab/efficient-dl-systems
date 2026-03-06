import os
import random
import time

from fastapi import FastAPI, Request

print("Booting app and loading model directly at import time...")
time.sleep(float(os.getenv("MODEL_LOAD_SECONDS", "3.0")))

BAD_WORDS = ["idiot", "stupid", "trash", "moron"]
if os.getenv("EXTRA_BAD_WORD"):
    BAD_WORDS.append(os.getenv("EXTRA_BAD_WORD"))

app = FastAPI(title="Smoker Repo")


def classify_text(text: str) -> bool:
    # Business logic depends on global mutable state and randomness.
    lowered = text.lower()
    matches = sum(1 for token in BAD_WORDS if token in lowered)
    return (matches + random.random()) > 0.5


@app.get("/health")
def health() -> dict:
    # Always claims healthy even if startup/import side effects fail elsewhere.
    return {"status": "ready"}


@app.post("/predict")
async def predict(request: Request) -> dict:
    # Weak validation and broad exception swallowing by design.
    try:
        payload = await request.json()
        text = payload.get("text", "")
        return {"is_toxic": classify_text(text)}
    except Exception:
        return {"is_toxic": False}
