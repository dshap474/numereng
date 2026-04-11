"""Client wiring for submission feature services."""

from __future__ import annotations

from typing import Literal, Protocol

from numereng.platform.numerai_client import NumeraiClient

NumeraiTournament = Literal["classic", "signals", "crypto"]


class SubmissionClient(Protocol):
    """Small protocol needed by submission services."""

    def get_models(self) -> dict[str, str]:
        """Return Numerai model mapping."""

    def list_datasets(self, round_num: int | None = None) -> list[str]:
        """Return available dataset names for the authenticated tournament."""

    def download_dataset(
        self,
        filename: str,
        *,
        dest_path: str | None = None,
        round_num: int | None = None,
    ) -> str:
        """Download one dataset file and return the local path."""

    def upload_predictions(self, *, file_path: str, model_id: str) -> str:
        """Upload predictions and return submission id."""

    def model_upload(
        self,
        *,
        file_path: str,
        model_id: str,
        data_version: str | None = None,
        docker_image: str | None = None,
    ) -> str:
        """Upload a Numerai model pickle and return upload id."""

    def model_upload_data_versions(self) -> list[str]:
        """List supported model-upload data versions."""

    def model_upload_docker_images(self) -> list[str]:
        """List supported model-upload docker images."""


def create_submission_client(*, tournament: NumeraiTournament = "classic") -> SubmissionClient:
    """Create default submission client."""
    return NumeraiClient(tournament=tournament)
