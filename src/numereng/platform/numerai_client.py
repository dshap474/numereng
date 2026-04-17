"""Thin wrapper around numerapi for Numerai Classic operations."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from numerapi import CryptoAPI, NumerAPI, SignalsAPI

from numereng.platform.errors import NumeraiClientError

NumeraiTournament = Literal["classic", "signals", "crypto"]

_COMPUTE_PICKLE_STATUS_QUERY = """
query PackageComputePickleStatus($id: ID, $modelId: ID, $unassigned: Boolean!) {
  computePickles(id: $id, modelId: $modelId, unassigned: $unassigned) {
    id
    filename
    modelId
    assignedModelSlots
    validationStatus
    diagnosticsStatus
    triggerStatus
    insertedAt
    updatedAt
  }
}
"""

_DIAGNOSTICS_TRIGGER_LOGS_QUERY = """
query PackageDiagnosticsTriggerLogs($pickleId: ID!) {
  diagnosticsTriggerLogs(pickleId: $pickleId) {
    timestamp
    message
  }
}
"""


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

    def diagnostics(self, *, model_id: str, diagnostics_id: str | None = None) -> dict[str, Any]:
        """Return one normalized Numerai diagnostics payload for one model."""
        try:
            payload = self._client.diagnostics(model_id=model_id, diagnostics_id=diagnostics_id)
        except Exception as exc:  # pragma: no cover - error mapping exercised in tests
            raise NumeraiClientError("numerai_diagnostics_failed") from exc
        return _normalize_diagnostics_payload(payload)

    def compute_pickle_status(
        self,
        *,
        pickle_id: str,
        model_id: str | None = None,
        unassigned: bool = False,
    ) -> dict[str, Any] | None:
        """Read back upload-specific compute pickle status."""
        try:
            payload = self._client.raw_query(
                _COMPUTE_PICKLE_STATUS_QUERY,
                {"id": pickle_id, "modelId": model_id, "unassigned": unassigned},
                authorization=True,
            )
        except Exception as exc:  # pragma: no cover - error mapping exercised in tests
            raise NumeraiClientError("numerai_compute_pickle_status_failed") from exc
        rows = payload.get("data", {}).get("computePickles", [])
        if not rows:
            return None
        return dict(rows[0])

    def compute_pickle_diagnostics_logs(self, *, pickle_id: str) -> list[dict[str, Any]]:
        """Return diagnostics trigger logs for one compute pickle upload."""
        try:
            payload = self._client.raw_query(
                _DIAGNOSTICS_TRIGGER_LOGS_QUERY,
                {"pickleId": pickle_id},
                authorization=True,
            )
        except Exception as exc:  # pragma: no cover - error mapping exercised in tests
            raise NumeraiClientError("numerai_compute_pickle_logs_failed") from exc
        logs = payload.get("data", {}).get("diagnosticsTriggerLogs", [])
        return [dict(item) for item in logs]

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


def _normalize_diagnostics_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return dict(payload)
    if isinstance(payload, list):
        dict_items = [dict(item) for item in payload if isinstance(item, dict)]
        if not dict_items:
            raise NumeraiClientError("numerai_diagnostics_empty")
        with_timestamps = [
            (idx, item, _diagnostics_updated_at_sort_key(item.get("updatedAt")))
            for idx, item in enumerate(dict_items)
        ]
        timestamped = [(idx, item, parsed) for idx, item, parsed in with_timestamps if parsed is not None]
        if timestamped:
            _, selected, _ = max(timestamped, key=lambda row: (row[2], -row[0]))
            return dict(selected)
        return dict(dict_items[0])
    raise NumeraiClientError("numerai_diagnostics_invalid")


def _parse_diagnostics_updated_at(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _diagnostics_updated_at_sort_key(value: object) -> float | None:
    parsed = _parse_diagnostics_updated_at(value)
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()
