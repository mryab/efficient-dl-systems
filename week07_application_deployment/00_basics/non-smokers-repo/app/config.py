from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    host: str = "0.0.0.0"
    port: int = 8080
    random_seed: int = 7
    model_startup_delay_seconds: float = 2.0
    redis_url: str = ""

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8080")),
            random_seed=int(os.getenv("RANDOM_SEED", "7")),
            model_startup_delay_seconds=float(
                os.getenv("MODEL_STARTUP_DELAY_SECONDS", "2.0")
            ),
            redis_url=os.getenv("REDIS_URL", ""),
        )
