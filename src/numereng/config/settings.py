"""Configuration boundaries for numereng."""

from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Package-level runtime settings."""

    environment: str = Field(default="dev")
    store_root: str = Field(default=".numereng")


def load_settings() -> Settings:
    """Load bootstrap settings."""
    return Settings()
