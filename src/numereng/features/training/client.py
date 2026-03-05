"""External data client wiring for training workflows."""

from __future__ import annotations

from typing import Protocol

from numereng.platform.numerai_client import NumeraiClient


class TrainingDataClient(Protocol):
    """Minimal protocol needed for training data retrieval."""

    def download_dataset(
        self,
        filename: str,
        *,
        dest_path: str | None = None,
        round_num: int | None = None,
    ) -> str:
        """Download one Numerai dataset file."""


def create_training_data_client() -> TrainingDataClient:
    """Create default data client for training workflows."""
    return NumeraiClient()
