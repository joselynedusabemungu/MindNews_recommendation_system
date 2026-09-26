from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    app_name: str = Field(default="MIND News Recommender", validation_alias="APP_NAME")
    environment: str = Field(default="production", validation_alias="ENVIRONMENT")
    model_path: Path = Field(default=Path("data/artifacts/model.pkl"), validation_alias="MODEL_PATH")
    database_url: str = Field(
        ...,
        validation_alias="DATABASE_URL",
    )
    api_prefix: str = Field(default="/api/v1", validation_alias="API_PREFIX")
    max_top_k: int = Field(default=50, ge=1, le=1000, validation_alias="MAX_TOP_K")
    train_dir: Path = Field(default=Path("data/raw/MINDlarge_train"), validation_alias="TRAIN_DIR")
    dev_dir: Path = Field(default=Path("data/raw/MINDlarge_dev"), validation_alias="DEV_DIR")
    test_dir: Path = Field(default=Path("data/raw/MINDlarge_test"), validation_alias="TEST_DIR")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

settings = get_settings()

def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def resolve_model_path(path: Path) -> Path:
    return resolve_path(path)

__all__ = ["Settings", "get_settings", "resolve_model_path", "resolve_path", "settings"]
