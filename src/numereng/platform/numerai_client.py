"""Thin wrapper around numerapi for Numerai Classic operations."""

from __future__ import annotations

from typing import Any, Literal

from numerapi import CryptoAPI, NumerAPI, SignalsAPI

from numereng.platform.errors import NumeraiClientError

NumeraiTournament = Literal["classic", "signals", "crypto"]


class NumeraiClient:
    """Minimal adapter over numerapi.NumerAPI used by feature services."""

    def __init__(
        self,
        *,
        tournament: NumeraiTournament = "classic",
        public_id: str | None = None,
        secret_key: str | None = None,
        verbosity: str = "INFO",
        show_progress_bars: bool = True,
    ) -> None:
        api_class: type[Any]
        if tournament == "classic":
            api_class = NumerAPI
        elif tournament == "signals":
            api_class = SignalsAPI
        elif tournament == "crypto":
            api_class = CryptoAPI
        else:  # pragma: no cover - protected by Literal in typed callers
            raise NumeraiClientError("numerai_tournament_not_supported")

        self._client = api_class(
            public_id=public_id,
            secret_key=secret_key,
            verbosity=verbosity,
            show_progress_bars=show_progress_bars,
        )

    def list_datasets(self, round_num: int | None = None) -> list[str]:
        """List available dataset paths for a round."""
        try:
            datasets = self._client.list_datasets(round_num=round_num)
        except Exception as exc:  # pragma: no cover - error mapping exercised in tests
            raise NumeraiClientError("numerai_list_datasets_failed") from exc
        return [str(item) for item in datasets]

    def download_dataset(
        self,
        filename: str,
        *,
        dest_path: str | None = None,
        round_num: int | None = None,
    ) -> str:
        """Download one Numerai dataset file."""
        try:
            return str(
                self._client.download_dataset(
                    filename=filename,
                    dest_path=dest_path,
                    round_num=round_num,
                )
            )
        except Exception as exc:  # pragma: no cover - error mapping exercised in tests
            raise NumeraiClientError("numerai_download_dataset_failed") from exc

    def get_models(self) -> dict[str, str]:
        """Return model name -> model id mapping for the authenticated account."""
        try:
            models = self._client.get_models()
        except Exception as exc:  # pragma: no cover - error mapping exercised in tests
            raise NumeraiClientError("numerai_get_models_failed") from exc

        mapping: dict[str, str] = {}
        for name, model_id in models.items():
            mapping[str(name)] = str(model_id)
        return mapping

    def upload_predictions(self, *, file_path: str, model_id: str) -> str:
        """Upload predictions and return submission id."""
        try:
            submission_id = self._client.upload_predictions(
                file_path=file_path,
                model_id=model_id,
            )
        except Exception as exc:  # pragma: no cover - error mapping exercised in tests
            raise NumeraiClientError("numerai_upload_predictions_failed") from exc
        return str(submission_id)

    def model_upload(
        self,
        *,
        file_path: str,
        model_id: str,
        data_version: str | None = None,
        docker_image: str | None = None,
    ) -> str:
        """Upload a Numerai model pickle and return the upload id."""
        try:
            upload_id = self._client.model_upload(
                file_path=file_path,
                model_id=model_id,
                data_version=data_version,
                docker_image=docker_image,
            )
        except Exception as exc:  # pragma: no cover - error mapping exercised in tests
            raise NumeraiClientError("numerai_model_upload_failed") from exc
        return str(upload_id)

    def model_upload_data_versions(self) -> list[str]:
        """Return available Numerai model-upload data versions."""
        try:
            values = self._client.model_upload_data_versions()
        except Exception as exc:  # pragma: no cover - error mapping exercised in tests
            raise NumeraiClientError("numerai_model_upload_data_versions_failed") from exc
        return [str(item) for item in values]

    def model_upload_docker_images(self) -> list[str]:
        """Return available Numerai model-upload docker images."""
        try:
            values = self._client.model_upload_docker_images()
        except Exception as exc:  # pragma: no cover - error mapping exercised in tests
            raise NumeraiClientError("numerai_model_upload_docker_images_failed") from exc
        return [str(item) for item in values]

    def get_current_round(self) -> int | None:
        """Return current round number when available."""
        try:
            round_num = self._client.get_current_round()
        except Exception as exc:  # pragma: no cover - error mapping exercised in tests
            raise NumeraiClientError("numerai_get_current_round_failed") from exc
        if round_num is None:
            return None
        return int(round_num)

    @property
    def raw_client(self) -> Any:
        """Expose underlying numerapi object for advanced/debug use only."""
        return self._client
