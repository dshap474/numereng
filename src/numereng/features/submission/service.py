"""Business logic for Numerai submission workflows."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import pandas as pd
from numerai_tools.submissions import validate_submission_numerai

from numereng.features.feature_neutralization import (
    NeutralizationMode,
    neutralize_predictions_file,
    neutralize_run_predictions,
)
from numereng.features.feature_neutralization.contracts import NeutralizePredictionsRequest
from numereng.features.submission.client import SubmissionClient, create_submission_client

NumeraiTournament = Literal["classic", "signals", "crypto"]

_PREDICTIONS_CANDIDATES = (
    "live_predictions.parquet",
    "predictions.parquet",
    "val_predictions.parquet",
)
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_CLASSIC_LIVE_DATASET_PATTERN = re.compile(r"^v(?P<major>\d+)(?:\.(?P<minor>\d+))?(?:\.(?P<patch>\d+))?/live\.parquet$")


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


class SubmissionPredictionsFormatUnsupportedError(Exception):
    """Raised when predictions input is not parquet."""


class SubmissionRunPredictionsNotLiveEligibleError(Exception):
    """Raised when run predictions do not look like live-submittable artifacts."""


class SubmissionRunIdInvalidError(Exception):
    """Raised when run_id is unsafe for filesystem lookup."""


class SubmissionRunPredictionsPathUnsafeError(Exception):
    """Raised when manifest predictions path escapes the run directory."""


class SubmissionLiveUniverseUnavailableError(Exception):
    """Raised when the classic live universe cannot be loaded for validation."""


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
    if path.suffix.lower() != ".parquet":
        raise SubmissionPredictionsFormatUnsupportedError("submission_predictions_format_unsupported")
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
        path for path in predictions_dir.iterdir() if path.is_file() and path.suffix.lower() == ".parquet"
    )
    if len(generic_candidates) == 1:
        return generic_candidates[0].resolve()

    raise SubmissionRunPredictionsNotFoundError(safe_run_id)


def _read_predictions_frame(predictions_path: Path) -> pd.DataFrame:
    suffix = predictions_path.suffix.lower()
    try:
        if suffix == ".parquet":
            return pd.read_parquet(predictions_path)
    except Exception as exc:
        raise SubmissionPredictionsReadError("submission_predictions_read_failed") from exc
    raise SubmissionPredictionsFormatUnsupportedError("submission_predictions_format_unsupported")


def _resolve_classic_live_dataset_name(*, client: SubmissionClient) -> str:
    dataset_names = client.list_datasets()
    candidates = [
        item
        for item in dataset_names
        if item.lower().endswith("/live.parquet") and not item.lower().startswith(("signals/", "crypto/"))
    ]
    if not candidates:
        raise SubmissionLiveUniverseUnavailableError("submission_live_universe_unavailable")
    return max(candidates, key=_classic_live_dataset_sort_key)


def _classic_live_dataset_sort_key(dataset_name: str) -> tuple[int, int, int, str]:
    match = _CLASSIC_LIVE_DATASET_PATTERN.fullmatch(dataset_name.lower())
    if match is None:
        return (-1, -1, -1, dataset_name)
    return (
        int(match.group("major")),
        int(match.group("minor") or 0),
        int(match.group("patch") or 0),
        dataset_name,
    )


def _load_classic_live_ids(*, client: SubmissionClient) -> pd.Series:
    dataset_name = _resolve_classic_live_dataset_name(client=client)
    try:
        with TemporaryDirectory(prefix="numereng-submit-live-") as tmp_dir:
            destination = Path(tmp_dir) / "live.parquet"
            local_path = Path(
                client.download_dataset(
                    dataset_name,
                    dest_path=str(destination),
                )
            )
            frame = pd.read_parquet(local_path, columns=["id"])
    except Exception as exc:
        raise SubmissionLiveUniverseUnavailableError("submission_live_universe_unavailable") from exc
    if "id" not in frame.columns:
        raise SubmissionLiveUniverseUnavailableError("submission_live_universe_unavailable")
    return frame["id"]


def _validate_predictions_live_eligible(
    *,
    predictions_path: Path,
    tournament: NumeraiTournament,
    client: SubmissionClient,
    allow_non_live_artifact: bool,
) -> None:
    if allow_non_live_artifact:
        return

    if tournament != "classic":
        _read_predictions_frame(predictions_path)
        return

    submission = _read_predictions_frame(predictions_path)
    live_ids = _load_classic_live_ids(client=client)
    try:
        validate_submission_numerai(live_ids, submission)
    except AssertionError as exc:
        raise SubmissionRunPredictionsNotLiveEligibleError("submission_run_predictions_not_live_eligible") from exc


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
        tournament=tournament,
        client=submission_client,
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
    "SubmissionLiveUniverseUnavailableError",
    "submit_predictions_file",
    "submit_run_predictions",
]
