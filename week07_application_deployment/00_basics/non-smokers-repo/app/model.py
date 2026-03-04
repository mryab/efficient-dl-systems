import random
import time


TOXIC_KEYWORDS = ("idiot", "stupid", "trash", "moron", "hate")


class ToxicityModel:
    def __init__(self, seed: int, startup_delay_seconds: float = 0.0) -> None:
        self._rng = random.Random(seed)
        self._startup_delay_seconds = startup_delay_seconds
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load(self) -> None:
        if self._startup_delay_seconds > 0:
            time.sleep(self._startup_delay_seconds)
        self._is_loaded = True

    def score(self, text: str) -> float:
        if not self._is_loaded:
            raise RuntimeError("Model is not loaded yet.")
        lowered = text.lower()
        hits = sum(1 for token in TOXIC_KEYWORDS if token in lowered)
        # Tiny deterministic jitter demonstrates seeded behavior in tests.
        return hits + (self._rng.random() * 0.0002)

    def predict(self, text: str) -> bool:
        return self.score(text) >= 1.0
