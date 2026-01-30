"""API configuration settings for Digital Court backend."""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuration settings for the FastAPI application."""

    # API Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    # CORS
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://localhost:8080",
    ]

    # Graph/Checkpoint
    checkpoint_namespace: str = "court_trial"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
