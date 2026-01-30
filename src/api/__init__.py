"""API module for Digital Court FastAPI backend."""
from src.api.main import app
from src.api.config import settings

__all__ = ["app", "settings"]
