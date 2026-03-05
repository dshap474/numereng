"""Client wiring for submission feature services."""

from __future__ import annotations

from typing import Literal, Protocol

from numereng.platform.numerai_client import NumeraiClient

NumeraiTournament = Literal["classic", "signals", "crypto"]


class SubmissionClient(Protocol):
    """Small protocol needed by submission services."""

    def get_models(self) -> dict[str, str]:
        """Return Numerai model mapping."""

    def upload_predictions(self, *, file_path: str, model_id: str) -> str:
        """Upload predictions and return submission id."""


def create_submission_client(*, tournament: NumeraiTournament = "classic") -> SubmissionClient:
    """Create default submission client."""
    return NumeraiClient(tournament=tournament)
