"""Business logic for Numerai submission workflows."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from numereng.features.feature_neutralization import (
    NeutralizationMode,
    neutralize_predictions_file,
    neutralize_run_predictions,
)
from numereng.features.feature_neutralization.contracts import NeutralizePredictionsRequest
from numereng.features.submission.client import SubmissionClient, create_submission_client

NumeraiTournament = Literal["classic", "signals", "crypto"]

_PREDICTIONS_CANDIDATES = (
    "live_predictions.csv",
    "live_predictions.parquet",
    "predictions.csv",
    "predictions.parquet",
    "val_predictions.parquet",
    "val_predictions.csv",
)
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class SubmissionModelNotFoundError(Exception):
    """Raised when a requested Numerai model name is not found."""


class SubmissionPredictionsFileNotFoundError(Exception):
    """Raised when a predictions file path does not exist."""


class SubmissionRunNotFoundError(Exception):
    """Raised when a run directory does not exist in the local store."""


class SubmissionRunPredictionsNotFoundError(Exception):
    """Raised when no predictions artifact is found for a run."""


class SubmissionPredictionsReadError(Exception):
    """Raised when predictions file columns cannot be inspected."""


class SubmissionRunPredictionsNotLiveEligibleError(Exception):
    """Raised when run predictions do not look like live-submittable artifacts."""


class SubmissionRunIdInvalidError(Exception):
    """Raised when run_id is unsafe for filesystem lookup."""


class SubmissionRunPredictionsPathUnsafeError(Exception):
    """Raised when manifest predictions path escapes the run directory."""


@dataclass(frozen=True)
class SubmissionResult:
    """Submission result returned by the feature service."""

    submission_id: str
    model_name: str
    model_id: str
    predictions_path: Path
    run_id: str | None = None


def _resolve_model_id(*, model_name: str, client: SubmissionClient) -> str:
    models = client.get_models()
    model_id = models.get(model_name)
    if model_id is None:
        raise SubmissionModelNotFoundError(model_name)
    return str(model_id)


def _resolve_predictions_path(predictions_path: str | Path) -> Path:
    path = Path(predictions_path)
    if not path.is_file():
        raise SubmissionPredictionsFileNotFoundError(str(path))
    return path.resolve()


def _validate_run_id(run_id: str) -> str:
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise SubmissionRunIdInvalidError("submission_run_id_invalid")
    return run_id


def _is_within(path: Path, *, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_manifest_predictions_path(*, run_dir: Path) -> Path | None:
    manifest_path = run_dir / "run.json"
    if not manifest_path.is_file():
        return None

    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        return None

    predictions_ref = artifacts.get("predictions")
    if not isinstance(predictions_ref, str) or not predictions_ref:
        return None

    candidate = Path(predictions_ref)
    if not candidate.is_absolute():
        candidate = run_dir / candidate
    resolved_candidate = candidate.resolve()
    if not _is_within(resolved_candidate, root=run_dir):
        raise SubmissionRunPredictionsPathUnsafeError("submission_run_predictions_path_unsafe")
    if resolved_candidate.is_file():
        return resolved_candidate
    return None


def _resolve_run_predictions_path(*, store_root: Path, run_id: str) -> Path:
    safe_run_id = _validate_run_id(run_id)
    runs_root = (store_root / "runs").resolve()
    run_dir = (runs_root / safe_run_id).resolve()
    if not _is_within(run_dir, root=runs_root):
        raise SubmissionRunIdInvalidError("submission_run_id_invalid")
    if not run_dir.is_dir():
        raise SubmissionRunNotFoundError(safe_run_id)

    manifest_candidate = _resolve_manifest_predictions_path(run_dir=run_dir)
    if manifest_candidate is not None:
        return manifest_candidate

    predictions_dir = run_dir / "artifacts" / "predictions"
    if not predictions_dir.is_dir():
        raise SubmissionRunPredictionsNotFoundError(safe_run_id)

    for filename in _PREDICTIONS_CANDIDATES:
        candidate = predictions_dir / filename
        if candidate.is_file():
            return candidate.resolve()

    generic_candidates = sorted(
        path for path in predictions_dir.iterdir() if path.is_file() and path.suffix.lower() in {".parquet", ".csv"}
    )
    if len(generic_candidates) == 1:
        return generic_candidates[0].resolve()

    raise SubmissionRunPredictionsNotFoundError(safe_run_id)


def _validate_predictions_live_eligible(
    *,
    predictions_path: Path,
    allow_non_live_artifact: bool,
) -> None:
    if allow_non_live_artifact:
        return

    columns = _read_prediction_columns(predictions_path)
    if "prediction" not in columns:
        raise SubmissionRunPredictionsNotLiveEligibleError("submission_run_predictions_not_live_eligible")
    if {"target", "cv_fold"} & columns:
        raise SubmissionRunPredictionsNotLiveEligibleError("submission_run_predictions_not_live_eligible")


def _read_prediction_columns(predictions_path: Path) -> set[str]:
    suffix = predictions_path.suffix.lower()
    try:
        if suffix == ".csv":
            frame = pd.read_csv(predictions_path, nrows=0)
            return {str(column) for column in frame.columns}
        if suffix == ".parquet":
            try:
                import pyarrow.parquet as pq  # type: ignore[import-untyped]
            except Exception:
                pq = None
            if pq is not None:
                parquet_file = pq.ParquetFile(predictions_path)
                return {str(column) for column in parquet_file.schema.names}
            frame = pd.read_parquet(predictions_path)
            return {str(column) for column in frame.columns}
    except Exception as exc:
        raise SubmissionPredictionsReadError("submission_predictions_read_failed") from exc
    raise SubmissionRunPredictionsNotLiveEligibleError("submission_run_predictions_not_live_eligible")


def submit_predictions_file(
    *,
    predictions_path: str | Path,
    model_name: str,
    tournament: NumeraiTournament = "classic",
    client: SubmissionClient | None = None,
    allow_non_live_artifact: bool = False,
    neutralize: bool = False,
    neutralizer_path: str | Path | None = None,
    neutralization_proportion: float = 0.5,
    neutralization_mode: NeutralizationMode = "era",
    neutralizer_cols: tuple[str, ...] | None = None,
    neutralization_rank_output: bool = True,
) -> SubmissionResult:
    """Submit one predictions file for a Numerai model name."""
    submission_client = create_submission_client(tournament=tournament) if client is None else client
    resolved_predictions_path = _resolve_predictions_path(predictions_path)
    if neutralize:
        if neutralizer_path is None:
            raise ValueError("submission_neutralizer_path_required")
        neutralized = neutralize_predictions_file(
            request=NeutralizePredictionsRequest(
                predictions_path=resolved_predictions_path,
                neutralizer_path=Path(neutralizer_path),
                proportion=neutralization_proportion,
                mode=neutralization_mode,
                neutralizer_cols=neutralizer_cols,
                rank_output=neutralization_rank_output,
            ),
        )
        resolved_predictions_path = neutralized.output_path

    _validate_predictions_live_eligible(
        predictions_path=resolved_predictions_path,
        allow_non_live_artifact=allow_non_live_artifact,
    )
    model_id = _resolve_model_id(model_name=model_name, client=submission_client)
    submission_id = submission_client.upload_predictions(
        file_path=str(resolved_predictions_path),
        model_id=model_id,
    )

    return SubmissionResult(
        submission_id=str(submission_id),
        model_name=model_name,
        model_id=model_id,
        predictions_path=resolved_predictions_path,
    )


def submit_run_predictions(
    *,
    run_id: str,
    model_name: str,
    tournament: NumeraiTournament = "classic",
    store_root: str | Path = ".numereng",
    client: SubmissionClient | None = None,
    allow_non_live_artifact: bool = False,
    neutralize: bool = False,
    neutralizer_path: str | Path | None = None,
    neutralization_proportion: float = 0.5,
    neutralization_mode: NeutralizationMode = "era",
    neutralizer_cols: tuple[str, ...] | None = None,
    neutralization_rank_output: bool = True,
) -> SubmissionResult:
    """Resolve run artifacts and submit the discovered predictions file."""
    submission_client = create_submission_client(tournament=tournament) if client is None else client
    resolved_predictions_path = _resolve_run_predictions_path(store_root=Path(store_root), run_id=run_id)
    _validate_predictions_live_eligible(
        predictions_path=resolved_predictions_path,
        allow_non_live_artifact=allow_non_live_artifact,
    )
    if neutralize:
        if neutralizer_path is None:
            raise ValueError("submission_neutralizer_path_required")
        neutralized = neutralize_run_predictions(
            run_id=run_id,
            neutralizer_path=neutralizer_path,
            store_root=store_root,
            proportion=neutralization_proportion,
            mode=neutralization_mode,
            neutralizer_cols=neutralizer_cols,
            rank_output=neutralization_rank_output,
        )
        resolved_predictions_path = neutralized.output_path

    result = submit_predictions_file(
        predictions_path=resolved_predictions_path,
        model_name=model_name,
        tournament=tournament,
        client=submission_client,
        allow_non_live_artifact=allow_non_live_artifact,
    )

    return SubmissionResult(
        submission_id=result.submission_id,
        model_name=result.model_name,
        model_id=result.model_id,
        predictions_path=result.predictions_path,
        run_id=run_id,
    )


__all__ = [
    "SubmissionModelNotFoundError",
    "SubmissionPredictionsFileNotFoundError",
    "SubmissionPredictionsReadError",
    "SubmissionResult",
    "SubmissionRunIdInvalidError",
    "SubmissionRunNotFoundError",
    "SubmissionRunPredictionsNotFoundError",
    "SubmissionRunPredictionsNotLiveEligibleError",
    "SubmissionRunPredictionsPathUnsafeError",
    "submit_predictions_file",
    "submit_run_predictions",
]
