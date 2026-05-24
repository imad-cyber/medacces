from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional


# Path to the project root (the medacces/ folder)
# __file__ = app/config.py
# .parent   = app/
# .parent   = medacces/   ← this is BASE_DIR
BASE_DIR = Path(__file__).parent.parent


class Settings(BaseSettings):
    """
    All application configuration lives here.
    Pydantic reads values from environment variables automatically.
    Variable names match .env keys exactly (case-insensitive).
    """

    # ── Database ────────────────────────────────────────────
    # database_url: str
    database_url: Optional[str] = "postgresql://dummy_url"
    secret_key: Optional[str] = "default_secret_key"


    # ── Security ─────────────────────────────────────────────
    #   secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # ── Application ──────────────────────────────────────────
    app_name: str = "MedAccès API"
    version: str = "1.0.0"
    environment: str = "development"

    # ── ML Paths ─────────────────────────────────────────────
    # These are built from BASE_DIR — no hardcoding needed
    model_path: str = str(BASE_DIR / "models" / "artifacts" / "medacces_model.joblib")
    metadata_path: str = str(BASE_DIR / "models" / "artifacts" / "model_metadata.json")
    data_path: str = str(BASE_DIR / "data" / "raw" / "communes_health.csv")

    class Config:
        env_file = ".env"          # where to read from
        env_file_encoding = "utf-8"


settings = Settings()