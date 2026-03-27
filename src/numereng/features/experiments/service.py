"""Experiment lifecycle services."""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from numereng.config.training.contracts import PostTrainingScoringPolicy
from numereng.features.experiments.contracts import (
    ExperimentArchiveResult,
    ExperimentPackResult,
    ExperimentPromotionResult,
    ExperimentRecord,
    ExperimentReport,
    ExperimentReportRow,
    ExperimentScoreRoundResult,
    ExperimentStatus,
    ExperimentTrainResult,
)
from numereng.features.scoring.batch_service import score_run_batch
from numereng.features.scoring.summary_metrics import SHARED_RUN_METRIC_NAMES, normalize_shared_run_metrics
from numereng.features.store import StoreError, index_run, resolve_store_root, upsert_experiment
from numereng.features.training import TrainingProfile, run_training
from numereng.features.training.run_log import log_error, resolve_run_log_path
from numereng.features.training.service import (
    is_round_post_training_scoring_policy,
    post_training_scoring_requested_stage,
    resolve_post_training_scoring_policy_from_config,
)

_SAFE_ID = re.compile(r"^[\w\-.]+$")
_EXPERIMENT_ID_FORMAT = re.compile(r"^\d{4}-\d{2}-\d{2}_[a-z0-9][a-z0-9-]*$")
_ROUND_LABEL_RE = re.compile(r"^r\d+$")
_ROUND_CONFIG_STEM_RE = re.compile(r"^(r\d+)_")
_DEFAULT_PROMOTION_METRIC = "bmc_last_200_eras.mean"
_ARCHIVE_DIRNAME = "_archive"
_PRE_ARCHIVE_STATUS_KEY = "pre_archive_status"
_PACK_OUTPUT_NAME = "EXPERIMENT.pack.md"
_RUN_PLAN_STUB_HEADER = "plan_index,round,seed,target,horizon,config_path,score_stage_default\n"


@dataclass(frozen=True)
class _ExperimentPaths:
    live_dir: Path
    archived_dir: Path
    active_dir: Path | None
    is_archived: bool


class ExperimentError(Exception):
    """Base error for experiment workflows."""


class ExperimentValidationError(ExperimentError):
    """Raised when experiment inputs are invalid."""


class ExperimentNotFoundError(ExperimentError):
    """Raised when experiment manifest is missing."""


class ExperimentAlreadyExistsError(ExperimentError):
    """Raised when experiment ID already exists."""


class ExperimentRunNotFoundError(ExperimentError):
    """Raised when requested run is not registered in experiment."""


def create_experiment(
    *,
    store_root: str | Path = ".numereng",
    experiment_id: str,
    name: str | None = None,
    hypothesis: str | None = None,
    tags: list[str] | None = None,
) -> ExperimentRecord:
    """Create one experiment manifest and index metadata in store DB."""

    safe_experiment_id = _ensure_safe_id(experiment_id)
    _ensure_experiment_id_format(safe_experiment_id)

    root = resolve_store_root(store_root)
    existing_paths = _experiment_paths(root, safe_experiment_id)
    if existing_paths.active_dir is not None:
        raise ExperimentAlreadyExistsError(f"experiment_already_exists:{safe_experiment_id}")
    exp_dir = _live_experiment_dir(root, safe_experiment_id)
    manifest_path = exp_dir / "experiment.json"

    exp_dir.mkdir(parents=True, exist_ok=False)
    (exp_dir / "configs").mkdir(exist_ok=True)
    (exp_dir / "run_scripts").mkdir(exist_ok=True)
    now = _utc_now_iso()
    normalized_tags = _normalize_tags(tags)

    manifest: dict[str, object] = {
        "schema_version": "1",
        "experiment_id": safe_experiment_id,
        "name": name or safe_experiment_id,
        "status": "draft",
        "hypothesis": hypothesis,
        "tags": normalized_tags,
        "created_at": now,
        "updated_at": now,
        "champion_run_id": None,
        "runs": [],
        "metadata": {},
    }
    _save_manifest(manifest_path, manifest)
    _save_run_plan_stub(exp_dir / "run_plan.csv")
    _save_experiment_launchers(exp_dir / "run_scripts", experiment_id=safe_experiment_id)
    _save_experiment_doc(exp_dir / "EXPERIMENT.md", manifest)
    _index_experiment_manifest(root, manifest)
    return _manifest_to_record(manifest_path, manifest)


def list_experiments(
    *,
    store_root: str | Path = ".numereng",
    status: ExperimentStatus | None = None,
) -> tuple[ExperimentRecord, ...]:
    """List experiments from manifest storage."""

    root = resolve_store_root(store_root)
    items: list[ExperimentRecord] = []
    include_archived = status == "archived"
    for manifest_path in _iter_experiment_manifest_paths(root, include_archived=include_archived):
        manifest = _load_manifest(manifest_path)
        record = _manifest_to_record(manifest_path, manifest)
        if status is None:
            if record.status == "archived":
                continue
        elif record.status != status:
            continue
        items.append(record)

    items.sort(key=lambda record: record.updated_at, reverse=True)
    return tuple(items)


def get_experiment(
    *,
    store_root: str | Path = ".numereng",
    experiment_id: str,
) -> ExperimentRecord:
    """Load one experiment record by ID."""

    safe_experiment_id = _ensure_safe_id(experiment_id)
    root = resolve_store_root(store_root)
    manifest_path = _resolved_manifest_path(root, safe_experiment_id)
    if not manifest_path.is_file():
        raise ExperimentNotFoundError(f"experiment_not_found:{safe_experiment_id}")
    manifest = _load_manifest(manifest_path)
    return _manifest_to_record(manifest_path, manifest)


def train_experiment(
    *,
    store_root: str | Path = ".numereng",
    experiment_id: str,
    config_path: str | Path,
    output_dir: str | Path | None = None,
    profile: TrainingProfile | None = None,
    post_training_scoring: PostTrainingScoringPolicy | None = None,
    engine_mode: str | None = None,
    window_size_eras: int | None = None,
    embargo_eras: int | None = None,
) -> ExperimentTrainResult:
    """Execute one training run linked to an experiment."""

    safe_experiment_id = _ensure_safe_id(experiment_id)
    root = resolve_store_root(store_root)
    manifest_path = _resolved_manifest_path(root, safe_experiment_id)

    config_resolved = Path(config_path).expanduser().resolve()
    if not config_resolved.is_file():
        raise ExperimentValidationError(f"experiment_config_not_found:{config_resolved}")
    output_dir_resolved = _resolve_experiment_output_dir(output_dir=output_dir, store_root=root)
    resolved_post_training_scoring = resolve_post_training_scoring_policy_from_config(
        config_path=config_resolved,
        override=post_training_scoring,
    )
    resolved_round = _resolve_round_label_from_config_path(config_resolved)
    if is_round_post_training_scoring_policy(resolved_post_training_scoring) and resolved_round is None:
        raise ExperimentValidationError(
            f"experiment_round_post_training_scoring_requires_round_config:{config_resolved.name}"
        )

    manifest = _load_manifest(manifest_path)
    _ensure_experiment_mutable(manifest)
    if profile is None:
        result = run_training(
            config_path=config_resolved,
            output_dir=output_dir_resolved,
            post_training_scoring=resolved_post_training_scoring,
            engine_mode=engine_mode,
            window_size_eras=window_size_eras,
            embargo_eras=embargo_eras,
            experiment_id=safe_experiment_id,
            allow_round_batch_post_training_scoring=True,
        )
    else:
        result = run_training(
            config_path=config_resolved,
            output_dir=output_dir_resolved,
            profile=profile,
            post_training_scoring=resolved_post_training_scoring,
            engine_mode=engine_mode,
            window_size_eras=window_size_eras,
            embargo_eras=embargo_eras,
            experiment_id=safe_experiment_id,
            allow_round_batch_post_training_scoring=True,
        )

    run_ids = _normalize_run_ids(manifest.get("runs"))
    if result.run_id not in run_ids:
        run_ids.append(result.run_id)
    manifest["runs"] = run_ids
    if cast(str, manifest.get("status", "draft")) == "draft":
        manifest["status"] = "active"
    manifest["updated_at"] = _utc_now_iso()

    _save_manifest(manifest_path, manifest)
    _index_experiment_manifest(root, manifest)
    try:
        index_run(store_root=root, run_id=result.run_id)
    except StoreError as exc:
        raise ExperimentError(f"experiment_run_index_failed:{result.run_id}") from exc
    if is_round_post_training_scoring_policy(resolved_post_training_scoring):
        requested_stage = post_training_scoring_requested_stage(resolved_post_training_scoring)
        if requested_stage is None or resolved_round is None:
            raise ExperimentValidationError(f"experiment_round_post_training_scoring_invalid:{config_resolved.name}")
        try:
            score_experiment_round(
                store_root=root,
                experiment_id=safe_experiment_id,
                round=resolved_round,
                stage=cast(Literal["post_training_core", "post_training_full"], requested_stage),
            )
        except Exception as exc:
            _record_round_batch_scoring_failure(
                store_root=root,
                run_id=result.run_id,
                requested_stage=requested_stage,
                error_message=str(exc),
            )

    return ExperimentTrainResult(
        experiment_id=safe_experiment_id,
        run_id=result.run_id,
        predictions_path=result.predictions_path,
        results_path=result.results_path,
    )


def score_experiment_round(
    *,
    store_root: str | Path = ".numereng",
    experiment_id: str,
    round: str,
    stage: Literal["post_training_core", "post_training_full"],
) -> ExperimentScoreRoundResult:
    """Deferred-score all completed runs for one experiment round."""

    safe_experiment_id = _ensure_safe_id(experiment_id)
    safe_round = _ensure_round_label(round)
    root = resolve_store_root(store_root)
    manifest_path = _resolved_manifest_path(root, safe_experiment_id)
    manifest = _load_manifest(manifest_path)
    _ensure_experiment_mutable(manifest)

    run_ids = _resolve_round_run_ids(root=root, experiment_id=safe_experiment_id, manifest=manifest, round=safe_round)
    if not run_ids:
        raise ExperimentRunNotFoundError(f"experiment_round_has_no_completed_runs:{safe_experiment_id}:{safe_round}")

    score_run_batch(
        run_ids=run_ids,
        store_root=root,
        stage=stage,
    )

    manifest["updated_at"] = _utc_now_iso()
    _save_manifest(manifest_path, manifest)
    _index_experiment_manifest(root, manifest)
    return ExperimentScoreRoundResult(
        experiment_id=safe_experiment_id,
        round=safe_round,
        stage=stage,
        run_ids=tuple(run_ids),
    )


def promote_experiment(
    *,
    store_root: str | Path = ".numereng",
    experiment_id: str,
    run_id: str | None = None,
    metric: str = _DEFAULT_PROMOTION_METRIC,
) -> ExperimentPromotionResult:
    """Promote one experiment run to champion by explicit run or metric."""

    safe_experiment_id = _ensure_safe_id(experiment_id)
    root = resolve_store_root(store_root)
    manifest_path = _resolved_manifest_path(root, safe_experiment_id)

    manifest = _load_manifest(manifest_path)
    _ensure_experiment_mutable(manifest)
    run_ids = _normalize_run_ids(manifest.get("runs"))
    if not run_ids:
        raise ExperimentRunNotFoundError(f"experiment_has_no_runs:{safe_experiment_id}")

    selected_run_id: str
    selected_value: float | None = None
    auto_selected = False
    if run_id is not None:
        safe_run_id = _ensure_safe_id(run_id)
        if safe_run_id not in run_ids:
            raise ExperimentRunNotFoundError(f"experiment_run_not_found:{safe_run_id}")
        selected_run_id = safe_run_id
        selected_value = _extract_metric_for_run(root=root, run_id=safe_run_id, metric=metric)
    else:
        auto_selected = True
        selected = _select_best_run(root=root, run_ids=run_ids, metric=metric)
        if selected is None:
            raise ExperimentRunNotFoundError(f"experiment_runs_missing_metric:{metric}")
        selected_run_id, selected_value = selected

    manifest["champion_run_id"] = selected_run_id
    manifest["updated_at"] = _utc_now_iso()
    _save_manifest(manifest_path, manifest)
    _index_experiment_manifest(root, manifest)

    return ExperimentPromotionResult(
        experiment_id=safe_experiment_id,
        champion_run_id=selected_run_id,
        metric=metric,
        metric_value=selected_value,
        auto_selected=auto_selected,
    )


def report_experiment(
    *,
    store_root: str | Path = ".numereng",
    experiment_id: str,
    metric: str = _DEFAULT_PROMOTION_METRIC,
    limit: int = 10,
) -> ExperimentReport:
    """Generate one ranked report from experiment run metrics."""

    safe_experiment_id = _ensure_safe_id(experiment_id)
    root = resolve_store_root(store_root)
    manifest_path = _resolved_manifest_path(root, safe_experiment_id)
    if limit < 1:
        raise ExperimentValidationError("experiment_report_limit_invalid")

    manifest = _load_manifest(manifest_path)
    run_ids = _normalize_run_ids(manifest.get("runs"))
    champion_run_id = _as_str(manifest.get("champion_run_id"))

    rows: list[ExperimentReportRow] = []
    for run_id in run_ids:
        run_manifest = _load_json_mapping(root / "runs" / run_id / "run.json")
        metrics = _load_json_mapping(root / "runs" / run_id / "metrics.json")
        row = ExperimentReportRow(
            run_id=run_id,
            status=_as_str(run_manifest.get("status")),
            created_at=_as_str(run_manifest.get("created_at")),
            metric_value=_extract_metric(metrics, metric),
            corr_mean=_extract_metric(metrics, "corr.mean"),
            mmc_mean=_extract_metric(metrics, "mmc.mean"),
            cwmm_mean=_extract_metric(metrics, "cwmm.mean"),
            bmc_mean=_extract_metric(metrics, "bmc.mean"),
            bmc_last_200_eras_mean=_extract_metric(metrics, "bmc_last_200_eras.mean"),
            is_champion=(champion_run_id == run_id),
        )
        rows.append(row)

    rows.sort(
        key=lambda item: (
            item.metric_value is None,
            -(item.metric_value if item.metric_value is not None else float("-inf")),
            item.run_id,
        )
    )
    total_runs = len(rows)
    rows = rows[:limit]

    return ExperimentReport(
        experiment_id=safe_experiment_id,
        metric=metric,
        total_runs=total_runs,
        champion_run_id=champion_run_id,
        rows=tuple(rows),
    )


def pack_experiment(
    *,
    store_root: str | Path = ".numereng",
    experiment_id: str,
) -> ExperimentPackResult:
    """Render one experiment narrative plus run-summary metrics into markdown."""

    safe_experiment_id = _ensure_safe_id(experiment_id)
    root = resolve_store_root(store_root)
    experiment_dir = _resolved_experiment_dir(root, safe_experiment_id)
    manifest_path = experiment_dir / "experiment.json"
    doc_path = experiment_dir / "EXPERIMENT.md"
    output_path = experiment_dir / _PACK_OUTPUT_NAME

    manifest = _load_manifest(manifest_path)
    if not doc_path.is_file():
        raise ExperimentValidationError(f"experiment_doc_missing:{doc_path}")
    try:
        notes_body = doc_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ExperimentError(f"experiment_doc_read_failed:{doc_path}") from exc

    champion_run_id = _as_str(manifest.get("champion_run_id"))
    run_ids = _normalize_run_ids(manifest.get("runs"))
    packed_at = _utc_now_iso()
    rendered = _render_packed_experiment_markdown(
        root=root,
        experiment_dir=experiment_dir,
        manifest=manifest,
        notes_body=notes_body,
        run_ids=run_ids,
        champion_run_id=champion_run_id,
        packed_at=packed_at,
        output_path=output_path,
    )
    output_path.write_text(rendered, encoding="utf-8")

    return ExperimentPackResult(
        experiment_id=safe_experiment_id,
        output_path=output_path,
        experiment_path=experiment_dir,
        source_markdown_path=doc_path,
        run_count=len(run_ids),
        packed_at=packed_at,
    )


def archive_experiment(
    *,
    store_root: str | Path = ".numereng",
    experiment_id: str,
) -> ExperimentArchiveResult:
    """Archive one experiment by moving its directory under the archive root."""

    safe_experiment_id = _ensure_safe_id(experiment_id)
    root = resolve_store_root(store_root)
    paths = _experiment_paths(root, safe_experiment_id)
    if paths.active_dir is None:
        raise ExperimentNotFoundError(f"experiment_not_found:{safe_experiment_id}")
    if paths.is_archived:
        raise ExperimentValidationError(f"experiment_already_archived:{safe_experiment_id}")
    if paths.archived_dir.exists():
        raise ExperimentValidationError(f"experiment_archive_destination_exists:{safe_experiment_id}")
    paths.archived_dir.parent.mkdir(parents=True, exist_ok=True)

    manifest_path = paths.live_dir / "experiment.json"
    manifest = _load_manifest(manifest_path)
    metadata = _normalize_metadata(manifest.get("metadata"))
    current_status = _coerce_status(_as_str(manifest.get("status")) or "draft")
    metadata[_PRE_ARCHIVE_STATUS_KEY] = current_status
    manifest["metadata"] = metadata
    manifest["status"] = "archived"
    manifest["updated_at"] = _utc_now_iso()
    _save_manifest(manifest_path, manifest)

    shutil.move(str(paths.live_dir), str(paths.archived_dir))

    archived_manifest_path = paths.archived_dir / "experiment.json"
    archived_manifest = _load_manifest(archived_manifest_path)
    _index_experiment_manifest(root, archived_manifest)
    return ExperimentArchiveResult(
        experiment_id=safe_experiment_id,
        status="archived",
        manifest_path=archived_manifest_path,
        archived=True,
    )


def unarchive_experiment(
    *,
    store_root: str | Path = ".numereng",
    experiment_id: str,
) -> ExperimentArchiveResult:
    """Restore one archived experiment to the live experiments root."""

    safe_experiment_id = _ensure_safe_id(experiment_id)
    root = resolve_store_root(store_root)
    paths = _experiment_paths(root, safe_experiment_id)
    if paths.active_dir is None:
        raise ExperimentNotFoundError(f"experiment_not_found:{safe_experiment_id}")
    if not paths.is_archived:
        raise ExperimentValidationError(f"experiment_not_archived:{safe_experiment_id}")
    if paths.live_dir.exists():
        raise ExperimentValidationError(f"experiment_unarchive_destination_exists:{safe_experiment_id}")
    paths.live_dir.parent.mkdir(parents=True, exist_ok=True)

    manifest_path = paths.archived_dir / "experiment.json"
    manifest = _load_manifest(manifest_path)
    metadata = _normalize_metadata(manifest.get("metadata"))
    restored_status = _coerce_status(_as_str(metadata.pop(_PRE_ARCHIVE_STATUS_KEY, None)) or "active")
    if restored_status == "archived":
        restored_status = "active"
    manifest["metadata"] = metadata
    manifest["status"] = restored_status
    manifest["updated_at"] = _utc_now_iso()
    _save_manifest(manifest_path, manifest)

    shutil.move(str(paths.archived_dir), str(paths.live_dir))

    live_manifest = _load_manifest(paths.live_dir / "experiment.json")
    _index_experiment_manifest(root, live_manifest)
    return ExperimentArchiveResult(
        experiment_id=safe_experiment_id,
        status=cast(ExperimentStatus, live_manifest.get("status")),
        manifest_path=paths.live_dir / "experiment.json",
        archived=False,
    )


def _ensure_safe_id(value: str) -> str:
    if not value or not _SAFE_ID.match(value):
        raise ExperimentValidationError(f"experiment_id_invalid:{value}")
    return value


def _ensure_experiment_id_format(experiment_id: str) -> None:
    if not _EXPERIMENT_ID_FORMAT.match(experiment_id):
        raise ExperimentValidationError("experiment_id_format_invalid:expected_YYYY-MM-DD_slug")


def _ensure_round_label(value: str) -> str:
    candidate = value.strip()
    if not candidate or not _ROUND_LABEL_RE.match(candidate):
        raise ExperimentValidationError(f"experiment_round_invalid:{value}")
    return candidate


def _live_experiment_dir(root: Path, experiment_id: str) -> Path:
    return root / "experiments" / experiment_id


def _archived_experiment_dir(root: Path, experiment_id: str) -> Path:
    return root / "experiments" / _ARCHIVE_DIRNAME / experiment_id


def _experiment_paths(root: Path, experiment_id: str) -> _ExperimentPaths:
    live_dir = _live_experiment_dir(root, experiment_id)
    archived_dir = _archived_experiment_dir(root, experiment_id)
    live_exists = (live_dir / "experiment.json").is_file()
    archived_exists = (archived_dir / "experiment.json").is_file()
    if live_exists and archived_exists:
        raise ExperimentValidationError(f"experiment_manifest_conflict:{experiment_id}")
    if live_exists:
        return _ExperimentPaths(live_dir=live_dir, archived_dir=archived_dir, active_dir=live_dir, is_archived=False)
    if archived_exists:
        return _ExperimentPaths(
            live_dir=live_dir,
            archived_dir=archived_dir,
            active_dir=archived_dir,
            is_archived=True,
        )
    return _ExperimentPaths(live_dir=live_dir, archived_dir=archived_dir, active_dir=None, is_archived=False)


def _resolved_manifest_path(root: Path, experiment_id: str) -> Path:
    paths = _experiment_paths(root, experiment_id)
    if paths.active_dir is None:
        raise ExperimentNotFoundError(f"experiment_not_found:{experiment_id}")
    return paths.active_dir / "experiment.json"


def _resolve_round_label_from_config_path(config_path: Path) -> str | None:
    match = _ROUND_CONFIG_STEM_RE.match(config_path.stem)
    if match is None:
        return None
    return match.group(1)


def _resolved_experiment_dir(root: Path, experiment_id: str) -> Path:
    paths = _experiment_paths(root, experiment_id)
    if paths.active_dir is None:
        raise ExperimentNotFoundError(f"experiment_not_found:{experiment_id}")
    return paths.active_dir


def _resolve_round_run_ids(
    *,
    root: Path,
    experiment_id: str,
    manifest: dict[str, object],
    round: str,
) -> tuple[str, ...]:
    experiment_dir = _resolved_experiment_dir(root, experiment_id)
    round_stems = _load_round_config_stems(experiment_dir=experiment_dir, round=round)
    if not round_stems:
        raise ExperimentValidationError(f"experiment_round_not_found:{experiment_id}:{round}")

    run_by_stem: dict[str, tuple[datetime, int, str]] = {}
    expected_stems = set(round_stems)
    for manifest_index, run_id in enumerate(_normalize_run_ids(manifest.get("runs"))):
        run_manifest = _load_json_mapping(root / "runs" / run_id / "run.json")
        if not _round_run_is_eligible(root=root, run_id=run_id, run_manifest=run_manifest):
            continue
        config_path = _as_str(_as_mapping(run_manifest.get("config")).get("path"))
        if config_path is None:
            continue
        stem = Path(config_path).stem
        if stem not in expected_stems:
            continue
        candidate = (
            _parse_run_created_at(run_manifest),
            manifest_index,
            run_id,
        )
        current = run_by_stem.get(stem)
        if current is None or candidate[:2] >= current[:2]:
            run_by_stem[stem] = candidate
    return tuple(run_by_stem[stem][2] for stem in round_stems if stem in run_by_stem)


def _round_run_is_eligible(*, root: Path, run_id: str, run_manifest: dict[str, object]) -> bool:
    if _as_str(run_manifest.get("status")) != "FINISHED":
        return False
    return _run_predictions_exist(root=root, run_id=run_id, run_manifest=run_manifest)


def _run_predictions_exist(*, root: Path, run_id: str, run_manifest: dict[str, object]) -> bool:
    run_dir = root / "runs" / run_id
    artifacts = _as_mapping(run_manifest.get("artifacts"))
    predictions_rel = _as_str(artifacts.get("predictions"))
    if predictions_rel is not None:
        return (run_dir / predictions_rel).is_file()
    predictions_dir = run_dir / "artifacts" / "predictions"
    return len(tuple(predictions_dir.glob("*.parquet"))) == 1


def _parse_run_created_at(run_manifest: dict[str, object]) -> datetime:
    created_at = _as_str(run_manifest.get("created_at"))
    if created_at is None:
        return datetime.min.replace(tzinfo=UTC)
    try:
        parsed = datetime.fromisoformat(created_at)
    except ValueError:
        return datetime.min.replace(tzinfo=UTC)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _load_round_config_stems(*, experiment_dir: Path, round: str) -> tuple[str, ...]:
    run_plan_path = experiment_dir / "run_plan.csv"
    stems: list[str] = []
    if run_plan_path.is_file():
        try:
            with run_plan_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    config_path = row.get("config_path")
                    if not isinstance(config_path, str) or not config_path.strip():
                        continue
                    stem = Path(config_path).stem
                    if stem.startswith(f"{round}_"):
                        stems.append(stem)
        except OSError as exc:
            raise ExperimentError(f"experiment_run_plan_read_failed:{run_plan_path}") from exc
    if stems:
        return tuple(stems)

    config_dir = experiment_dir / "configs"
    if not config_dir.is_dir():
        return ()
    return tuple(path.stem for path in sorted(config_dir.glob(f"{round}_*.json"), key=lambda item: item.name))


def _iter_experiment_manifest_paths(root: Path, *, include_archived: bool = False) -> tuple[Path, ...]:
    manifest_paths: list[Path] = []
    live_root = root / "experiments"
    if live_root.is_dir():
        for experiment_dir in sorted(live_root.iterdir(), key=lambda path: path.name):
            if not experiment_dir.is_dir() or experiment_dir.name == _ARCHIVE_DIRNAME:
                continue
            manifest_path = experiment_dir / "experiment.json"
            if manifest_path.is_file():
                manifest_paths.append(manifest_path)
    if include_archived:
        archive_root = live_root / _ARCHIVE_DIRNAME
        if archive_root.is_dir():
            for experiment_dir in sorted(archive_root.iterdir(), key=lambda path: path.name):
                if not experiment_dir.is_dir():
                    continue
                manifest_path = experiment_dir / "experiment.json"
                if manifest_path.is_file():
                    manifest_paths.append(manifest_path)
    return tuple(manifest_paths)


def _ensure_experiment_mutable(manifest: dict[str, object]) -> None:
    if _coerce_status(_as_str(manifest.get("status")) or "draft") == "archived":
        raise ExperimentValidationError("experiment_archived_read_only")


def _load_manifest(path: Path) -> dict[str, object]:
    payload = _load_json_mapping(path)
    if not payload:
        raise ExperimentValidationError(f"experiment_manifest_invalid:{path}")
    return payload


def _load_json_mapping(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return cast(dict[str, object], payload)


def _require_json_mapping(path: Path, *, error_code: str) -> dict[str, object]:
    payload = _load_json_mapping(path)
    if payload:
        return payload
    raise ExperimentValidationError(f"{error_code}:{path}")


def _save_manifest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _record_round_batch_scoring_failure(
    *,
    store_root: Path,
    run_id: str,
    requested_stage: str,
    error_message: str,
) -> None:
    run_dir = store_root / "runs" / run_id
    run_manifest_path = run_dir / "run.json"
    results_path = run_dir / "results.json"
    metrics_path = run_dir / "metrics.json"
    run_log_path = resolve_run_log_path(run_dir)

    log_error(
        run_log_path,
        event="post_training_round_batch_failed",
        message=f"stage={requested_stage} error={error_message}",
    )

    run_manifest = _load_json_mapping(run_manifest_path)
    results = _load_json_mapping(results_path)

    results_training = _as_mapping(results.get("training"))
    existing_results_scoring = _as_mapping(results_training.get("scoring"))
    if _as_str(existing_results_scoring.get("status")) != "succeeded":
        results["metrics"] = {
            "status": "failed",
            "reason": "experiment_round_batch_failed",
            "error": error_message,
        }
        results_training["scoring"] = _failed_round_batch_scoring_payload(
            existing_payload=existing_results_scoring,
            requested_stage=requested_stage,
            error_message=error_message,
        )
        results["training"] = results_training
        _save_manifest(results_path, results)
        _save_manifest(metrics_path, cast(dict[str, object], results["metrics"]))

    manifest_training = _as_mapping(run_manifest.get("training"))
    existing_manifest_scoring = _as_mapping(manifest_training.get("scoring"))
    if _as_str(existing_manifest_scoring.get("status")) != "succeeded":
        manifest_training["scoring"] = _failed_round_batch_scoring_payload(
            existing_payload=existing_manifest_scoring,
            requested_stage=requested_stage,
            error_message=error_message,
        )
        run_manifest["training"] = manifest_training
        _save_manifest(run_manifest_path, run_manifest)
        try:
            index_run(store_root=store_root, run_id=run_id)
        except StoreError:
            pass


def _failed_round_batch_scoring_payload(
    *,
    existing_payload: dict[str, object],
    requested_stage: str,
    error_message: str,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": "failed",
        "requested_stage": requested_stage,
        "refreshed_stages": [],
        "reason": "experiment_round_batch_failed",
        "error": error_message,
    }
    policy = _as_str(existing_payload.get("policy"))
    if policy is not None:
        payload["policy"] = policy
    return payload


def _save_experiment_doc(path: Path, manifest: dict[str, object]) -> None:
    path.write_text(_render_experiment_doc_template(manifest), encoding="utf-8")


def _save_run_plan_stub(path: Path) -> None:
    path.write_text(_RUN_PLAN_STUB_HEADER, encoding="utf-8")


def _save_experiment_launchers(run_scripts_dir: Path, *, experiment_id: str) -> None:
    launch_all_py = run_scripts_dir / "launch_all.py"
    launch_all_sh = run_scripts_dir / "launch_all.sh"
    launch_all_ps1 = run_scripts_dir / "launch_all.ps1"
    launch_all_py.write_text(_render_launch_all_py(experiment_id=experiment_id), encoding="utf-8")
    launch_all_sh.write_text(_render_launch_all_sh(), encoding="utf-8")
    launch_all_ps1.write_text(_render_launch_all_ps1(), encoding="utf-8")
    launch_all_py.chmod(0o755)
    launch_all_sh.chmod(0o755)


def _render_experiment_doc_template(manifest: dict[str, object]) -> str:
    experiment_id = _as_str(manifest.get("experiment_id")) or "<YYYY-MM-DD_slug>"
    name = _as_str(manifest.get("name")) or experiment_id
    created_at = _as_str(manifest.get("created_at")) or _utc_now_iso()
    updated_at = _as_str(manifest.get("updated_at")) or created_at
    created_date = created_at.split("T", 1)[0]
    status = _as_str(manifest.get("status")) or "draft"
    champion_run_id = _as_str(manifest.get("champion_run_id")) or "none"
    hypothesis = _as_str(manifest.get("hypothesis")) or "<short hypothesis>"
    tags = _normalize_tags(manifest.get("tags"))
    tags_text = ",".join(tags) if tags else "<tag1,tag2>"
    return "\n".join(
        [
            f"# {name}",
            "",
            f"**ID**: `{experiment_id}`",
            f"**Created**: {created_date}",
            f"**Updated**: {updated_at}",
            f"**Status**: {status}",
            f"**Champion run**: {champion_run_id}",
            f"**Tags**: {tags_text}",
            "",
            "## Summary",
            "",
            f"- **Hypothesis**: {hypothesis}",
            "- **Primary metric**: `bmc_last_200_eras.mean`",
            "- **Tie-break metric**: `bmc.mean`",
            "- **Outcome**: <what the experiment established so far>",
            "",
            "## Abstract",
            "",
            "- What was tested:",
            "- Headline result:",
            "- Best run / leading candidate:",
            "- Why this result matters:",
            "",
            "## Method",
            "",
            "- Data split / CV setup:",
            "- Feature set or dataset scope:",
            "- Model family and key hyperparameters:",
            "- Target(s) evaluated:",
            "- Transforms / neutralization / special scoring notes:",
            "",
            "## Execution Inventory",
            "",
            "- Planned configs:",
            "- Executed configs:",
            "- Failed / interrupted configs:",
            "- Skipped / superseded configs:",
            f"- Launcher path (if used): `.numereng/experiments/{experiment_id}/run_scripts/launch_all.sh`",
            (
                "- Verification source of truth: `experiment.json`, `run_plan.csv`, "
                "`run.json`, `metrics.json`, `results.json`, `score_provenance.json`"
            ),
            "- Notes on what was actually run versus what remains only planned:",
            "",
            "## Ambiguity Resolution (If Applicable)",
            "",
            "| interpretation_id | interpretation | scout configs tested | winner run_id | rationale |",
            "|---|---|---|---|---|",
            "| `int-a` | `<interpretation>` | `<config_a,config_b>` | `<run_id>` | `<why selected>` |",
            "",
            "## Scout -> Scale Tracker",
            "",
            "- Current stage: `<scout|scale>`",
            "- Scout compute profile used (for example dataset variant/model capacity):",
            "- Scale gate met: `<yes|no>`",
            "- Confirmatory scaled round run: `<yes|no>`",
            "",
            "## Plateau Gate Settings",
            "",
            "- Primary stop metric: `bmc_last_200_eras.mean`",
            "- Meaningful gain threshold: `<default 1e-4 to 3e-4>`",
            "- Consecutive non-improving rounds required: `<default 2>`",
            "",
            "## Round Log",
            "",
            "### Round <N>",
            "",
            "#### Intent",
            "- Question this round answers:",
            "- Single variable changed:",
            "- Why this change now:",
            "",
            "#### Configs Executed",
            "| config_path | command | status |",
            "|---|---|---|",
            (
                f"| `.numereng/experiments/{experiment_id}/configs/<config>.json` | "
                f"`uv run numereng experiment train --id {experiment_id} --config "
                "<config>.json --post-training-scoring none` | "
                "<planned|running|done|failed> |"
            ),
            "",
            "Execution notes:",
            (
                "- State explicitly if a config was skipped, superseded, "
                "deduplicated, or replaced by a recovered canonical run."
            ),
            "- If a round used a launcher script, still record the canonical per-config command family.",
            "",
            "#### Results",
            (
                "| run_id | status (`run.json`) | created_at (`run.json`) | "
                "bmc_last_200_eras.mean | bmc.mean | corr.mean | mmc.mean | "
                "cwmm.mean | notes |"
            ),
            "|---|---|---|---:|---:|---:|---:|---:|---|",
            "",
            "Artifact checks:",
            (
                "- Confirm whether each listed run has `run.json`, `resolved.json`, "
                "`metrics.json`, `results.json`, `score_provenance.json`, and "
                "persisted predictions."
            ),
            "- If a required artifact is missing, state it here instead of treating the run as fully complete.",
            "",
            "#### Decision",
            "- Winner:",
            "- Round-best delta vs prior-best (`bmc_last_200_eras.mean`):",
            "- Why winner:",
            "- Risks observed:",
            "- Plateau gate status: `<continue|pivot|stop>`",
            "- Next round action:",
            "",
            "## Results",
            "",
            "- Best run on primary metric:",
            "- Best run on tie-break metric:",
            "- Main trade-offs across leading runs:",
            "- Detailed evidence source: `Round Log` and `EXPERIMENT.pack.md`",
            "",
            "Compact summary table (use one row per executed run or per leading candidate set):",
            "",
            (
                "| run_id | config_path | round | status | bmc_last_200_eras.mean | "
                "bmc.mean | corr.mean | mmc.mean | avg_corr_with_benchmark | notes |"
            ),
            "|---|---|---:|---|---:|---:|---:|---:|---:|---|",
            (
                f"| `<run_id>` | `.numereng/experiments/{experiment_id}/configs/<config>.json` | "
                "`<N>` | `<FINISHED>` | `<...>` | `<...>` | `<...>` | "
                "`<...|n/a>` | `<...|n/a>` | `<trade-off or selection note>` |"
            ),
            "",
            "Result interpretation:",
            "- What pattern actually changed the leaderboard:",
            "- What looked good on one metric but failed on another:",
            "- Whether the best result is robust enough to promote:",
            "",
            "## Standard Plots / Visual Checks",
            "",
            "- Plot objective: `<benchmark comparison|candidate comparison|seed stability|other>`",
            "- Plot generation command:",
            "- Artifact path(s):",
            "- Included in report: `<yes|no>`",
            "- If no plot was generated, explain why and what scalar evidence substituted for it:",
            "",
            "| plot_id | purpose | command | artifact_path | notes |",
            "|---|---|---|---|---|",
            (
                "| `<plot-1>` | `<what this plot shows>` | `<command>` | "
                "`<relative/path.png>` | `<interpretation or caveat>` |"
            ),
            "",
            "## Ensemble Log (Optional)",
            "",
            "### Ensemble <N>",
            "- Command used:",
            "- Method: `rank_avg`",
            "- Optimization metric:",
            "- Neutralization mode:",
            "- Include heavy artifacts: `<yes|no>`",
            "- Artifacts path:",
            "",
            "| ensemble_id | run_ids | method | metric | weights source | heavy artifacts | artifacts_path | notes |",
            "|---|---|---|---|---|---|---|---|",
            (
                "| `<ensemble_id>` | `<run_a,run_b,...>` | `rank_avg` | "
                "`<corr20v2_sharpe|corr20v2_mean|max_drawdown>` | "
                "`<explicit|optimized|equal>` | `<yes|no>` | `<path>` | "
                "`<selection rationale>` |"
            ),
            "",
            "Artifact checklist:",
            "- `weights.csv`",
            "- `component_metrics.csv`",
            "- `era_metrics.csv`",
            "- `regime_metrics.csv`",
            "- `lineage.json`",
            "- optional heavy: `component_predictions.parquet`, `bootstrap_metrics.json`",
            "- optional final-neutralization: `predictions_pre_neutralization.parquet`",
            "",
            "## Remaining Knobs Audit",
            "",
            "| knob/dimension | tried ranges | remaining options | expected value | overfit risk | decision |",
            "|---|---|---|---|---|---|",
            (
                "| `model.params.learning_rate` | `<...>` | `<...>` | "
                "`<high|medium|low>` | `<high|medium|low>` | "
                "`<continue|defer|drop>` |"
            ),
            "",
            "## Final Decision",
            "",
            "- Selected champion run:",
            "- Promotion command used:",
            "- Promotion metric/value:",
            "",
            "## Stopping Rationale",
            "",
            "- Why iteration stopped:",
            "- Plateau or diminishing-returns evidence:",
            "- Confirmatory run or scale-check evidence:",
            "- Remaining uncertainty accepted:",
            "",
            "## Findings",
            "",
            "- What worked:",
            "- What did not:",
            "- Unexpected observations:",
            "",
            "## Anti-Patterns Observed",
            "",
            "- <anti-pattern>",
            "",
            "## Next Experiments",
            "",
            "1.",
            "2.",
            "3.",
            "",
            "## Final Checks",
            "",
            "- `EXPERIMENT.md` clearly separates executed configs from planned-only configs.",
            "- Metrics reported here match the underlying run artifacts.",
            "- Linked artifact paths and plot paths resolve.",
            "- Champion run is either recorded or `none` is explained.",
            "- If the experiment is complete, `EXPERIMENT.pack.md` has been regenerated.",
            "",
            "## Repro Commands",
            "",
            "```bash",
            f"uv run numereng experiment details --id {experiment_id} --format json",
            f"uv run numereng experiment report --id {experiment_id} --metric bmc_last_200_eras.mean --format table",
            f"bash .numereng/experiments/{experiment_id}/run_scripts/launch_all.sh",
            f"uv run numereng experiment pack --id {experiment_id}",
            "uv run numereng ensemble details --ensemble-id <ensemble_id> --format json",
            f"uv run numereng experiment promote --id {experiment_id} --metric bmc_last_200_eras.mean",
            "# optional plot / viz command:",
            "# <plot command here>",
            "```",
            "",
        ]
    )


def _render_launch_all_py(*, experiment_id: str) -> str:
    return "\n".join(
        [
            "#!/usr/bin/env python3",
            '"""Run one experiment run_plan with script-owned round scoring."""',
            "",
            "from __future__ import annotations",
            "",
            "import argparse",
            "import csv",
            "import re",
            "import subprocess",
            "from dataclasses import dataclass",
            "from pathlib import Path",
            "",
            f'EXPERIMENT_ID = "{experiment_id}"',
            'ROUND_RE = re.compile(r"^(r\\d+)_")',
            'VALID_SCORE_STAGES = ("post_training_core", "post_training_full")',
            "",
            "",
            "@dataclass(frozen=True)",
            "class PlanRow:",
            "    index: int",
            "    config_path: Path",
            "    round_label: str",
            "",
            "",
            "def find_repo_root(start: Path) -> Path:",
            "    checked: set[Path] = set()",
            "    for root in (start.resolve(), Path.cwd().resolve()):",
            "        for candidate in (root, *root.parents):",
            "            if candidate in checked:",
            "                continue",
            "            checked.add(candidate)",
            '            if (candidate / "pyproject.toml").is_file():',
            "                return candidate",
            '    raise SystemExit(f"repo_root_not_found_from:{start}")',
            "",
            "",
            "SCRIPT_PATH = Path(__file__).resolve()",
            "RUN_SCRIPTS_DIR = SCRIPT_PATH.parent",
            "EXPERIMENT_DIR = RUN_SCRIPTS_DIR.parent",
            "REPO_ROOT = find_repo_root(RUN_SCRIPTS_DIR)",
            'RUN_PLAN_PATH = EXPERIMENT_DIR / "run_plan.csv"',
            "",
            "",
            "def parse_args() -> argparse.Namespace:",
            "    parser = argparse.ArgumentParser(",
            '        description="Train selected run-plan rows and score each completed round once."',
            "    )",
            (
                '    parser.add_argument("--start-index", type=int, default=1, '
                'help="1-based run_plan row index to start from.")'
            ),
            (
                '    parser.add_argument("--end-index", type=int, default=None, '
                'help="Optional 1-based run_plan row index to stop at.")'
            ),
            "    parser.add_argument(",
            '        "--score-stage",',
            "        choices=VALID_SCORE_STAGES,",
            '        default="post_training_core",',
            '        help="Round scoring stage to materialize after the last planned config in a round.",',
            "    )",
            "    return parser.parse_args()",
            "",
            "",
            "def resolve_config_path(raw_config_path: str) -> Path:",
            "    path = Path(raw_config_path)",
            "    if not path.is_absolute():",
            "        path = (REPO_ROOT / path).resolve()",
            "    else:",
            "        path = path.resolve()",
            "    return path",
            "",
            "",
            "def resolve_round_label(raw_round: str | None, *, config_path: Path) -> str:",
            "    if raw_round is not None and raw_round.strip():",
            "        round_label = raw_round.strip()",
            '        if re.fullmatch(r"r\\d+", round_label) is None:',
            '            raise SystemExit(f"invalid_round_label:{round_label}:{config_path.name}")',
            "        return round_label",
            "    match = ROUND_RE.match(config_path.stem)",
            "    if match is None:",
            '        raise SystemExit(f"round_label_missing:{config_path.name}")',
            "    return match.group(1)",
            "",
            "",
            "def load_run_plan() -> list[PlanRow]:",
            "    if not RUN_PLAN_PATH.is_file():",
            '        raise SystemExit(f"run_plan_missing:{RUN_PLAN_PATH}")',
            '    with RUN_PLAN_PATH.open("r", encoding="utf-8", newline="") as handle:',
            "        reader = csv.DictReader(handle)",
            "        rows: list[PlanRow] = []",
            "        for index, row in enumerate(reader, start=1):",
            '            raw_config_path = row.get("config_path")',
            "            if not isinstance(raw_config_path, str) or not raw_config_path.strip():",
            '                raise SystemExit(f"run_plan_config_path_missing:index={index}")',
            "            config_path = resolve_config_path(raw_config_path)",
            "            if not config_path.is_file():",
            '                raise SystemExit(f"config_missing:{config_path}")',
            '            round_label = resolve_round_label(row.get("round"), config_path=config_path)',
            "            rows.append(PlanRow(index=index, config_path=config_path, round_label=round_label))",
            "    if not rows:",
            '        raise SystemExit(f"run_plan_empty:{RUN_PLAN_PATH}")',
            "    return rows",
            "",
            "",
            "def run_command(command: list[str]) -> None:",
            '    print(f"+ {" ".join(command)}", flush=True)',
            "    completed = subprocess.run(command, cwd=REPO_ROOT)",
            "    if completed.returncode != 0:",
            "        raise SystemExit(completed.returncode)",
            "",
            "",
            "def main() -> int:",
            "    args = parse_args()",
            "    rows = load_run_plan()",
            "    total = len(rows)",
            "",
            "    if args.start_index < 1 or args.start_index > total:",
            '        raise SystemExit(f"start_index_out_of_range:{args.start_index}:1..{total}")',
            "    if args.end_index is not None and (args.end_index < args.start_index or args.end_index > total):",
            '        raise SystemExit(f"end_index_out_of_range:{args.end_index}:{args.start_index}..{total}")',
            "",
            "    selected_rows = rows[args.start_index - 1 : args.end_index]",
            "    last_index_by_round: dict[str, int] = {}",
            "    for row in rows:",
            "        last_index_by_round[row.round_label] = row.index",
            "",
            '    print(f"Experiment: {EXPERIMENT_ID}")',
            '    print(f"Run plan: {RUN_PLAN_PATH}")',
            '    print(f"Selected rows: {args.start_index}..{args.end_index or total}")',
            '    print(f"Round score stage: {args.score_stage}")',
            "",
            "    scored_rounds: set[str] = set()",
            "    for row in selected_rows:",
            "        print()",
            '        print(f">>> [{row.index}/{total}] Training: {row.config_path}")',
            "        run_command(",
            "            [",
            '                "uv",',
            '                "run",',
            '                "numereng",',
            '                "experiment",',
            '                "train",',
            '                "--id",',
            "                EXPERIMENT_ID,",
            '                "--config",',
            "                str(row.config_path),",
            '                "--post-training-scoring",',
            '                "none",',
            "            ]",
            "        )",
            "        if row.index == last_index_by_round[row.round_label] and row.round_label not in scored_rounds:",
            '            print(f">>> [{row.index}/{total}] Batch scoring round {row.round_label}: {args.score_stage}")',
            "            run_command(",
            "                [",
            '                    "uv",',
            '                    "run",',
            '                    "numereng",',
            '                    "experiment",',
            '                    "score-round",',
            '                    "--id",',
            "                    EXPERIMENT_ID,",
            '                    "--round",',
            "                    row.round_label,",
            '                    "--stage",',
            "                    args.score_stage,",
            "                ]",
            "            )",
            "            scored_rounds.add(row.round_label)",
            "",
            "    print()",
            '    print(f"Completed rows {args.start_index}..{args.end_index or total}.")',
            "    return 0",
            "",
            "",
            'if __name__ == "__main__":',
            "    raise SystemExit(main())",
            "",
        ]
    )


def _render_launch_all_sh() -> str:
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
            'RUNNER="${SCRIPT_DIR}/launch_all.py"',
            "",
            "resolve_repo_root() {",
            '  local start_dir="${1}"',
            '  local search_dir="${start_dir}"',
            '  while [[ "${search_dir}" != "/" ]]; do',
            '    if [[ -f "${search_dir}/pyproject.toml" ]]; then',
            '      printf "%s\\n" "${search_dir}"',
            "      return 0",
            "    fi",
            '    search_dir="$(dirname "${search_dir}")"',
            "  done",
            "  return 1",
            "}",
            "",
            'REPO_ROOT="$(resolve_repo_root "${SCRIPT_DIR}" || true)"',
            'if [[ -z "${REPO_ROOT}" ]]; then',
            '  REPO_ROOT="$(resolve_repo_root "$(pwd)" || true)"',
            "fi",
            "",
            'if [[ -z "${REPO_ROOT}" ]]; then',
            '  echo "Could not locate repo root (pyproject.toml) from ${SCRIPT_DIR} or $(pwd)" >&2',
            "  exit 1",
            "fi",
            "",
            'if [[ ! -f "${RUNNER}" ]]; then',
            '  echo "Launcher not found: ${RUNNER}" >&2',
            "  exit 1",
            "fi",
            "",
            "(",
            '  cd "${REPO_ROOT}"',
            '  uv run python "${RUNNER}" "$@"',
            ")",
            "",
        ]
    )


def _render_launch_all_ps1() -> str:
    return "\n".join(
        [
            "param(",
            "    [Parameter(ValueFromRemainingArguments = $true)]",
            "    [string[]]$RunnerArgs",
            ")",
            "",
            "Set-StrictMode -Version Latest",
            '$ErrorActionPreference = "Stop"',
            "",
            "$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path",
            '$runner = Join-Path $scriptDir "launch_all.py"',
            "",
            "function Resolve-RepoRoot {",
            "    param([string]$StartDir)",
            "",
            "    $searchDir = $StartDir",
            "    while ($true) {",
            '        if (Test-Path (Join-Path $searchDir "pyproject.toml")) {',
            "            return $searchDir",
            "        }",
            "        $parent = Split-Path $searchDir -Parent",
            "        if (-not $parent -or $parent -eq $searchDir) {",
            "            break",
            "        }",
            "        $searchDir = $parent",
            "    }",
            "",
            '    throw "Could not locate repo root (pyproject.toml) from $StartDir"',
            "}",
            "",
            "if (-not (Test-Path $runner)) {",
            '    throw "Launcher not found: $runner"',
            "}",
            "",
            "$repoRoot = $null",
            "try {",
            "    $repoRoot = Resolve-RepoRoot -StartDir $scriptDir",
            "}",
            "catch {",
            "    $repoRoot = $null",
            "}",
            "if (-not $repoRoot) {",
            "    $repoRoot = Resolve-RepoRoot -StartDir (Get-Location).Path",
            "}",
            "Push-Location $repoRoot",
            "try {",
            "    & uv run python $runner @RunnerArgs",
            "    if ($LASTEXITCODE -ne 0) {",
            "        exit $LASTEXITCODE",
            "    }",
            "}",
            "finally {",
            "    Pop-Location",
            "}",
            "",
        ]
    )


def _manifest_to_record(path: Path, manifest: dict[str, object]) -> ExperimentRecord:
    status_raw = _as_str(manifest.get("status")) or "draft"
    status = _coerce_status(status_raw)
    return ExperimentRecord(
        experiment_id=_as_str(manifest.get("experiment_id")) or path.parent.name,
        name=_as_str(manifest.get("name")) or path.parent.name,
        status=status,
        hypothesis=_as_str(manifest.get("hypothesis")),
        tags=tuple(_normalize_tags(manifest.get("tags"))),
        created_at=_as_str(manifest.get("created_at")) or _utc_now_iso(),
        updated_at=_as_str(manifest.get("updated_at")) or _utc_now_iso(),
        champion_run_id=_as_str(manifest.get("champion_run_id")),
        runs=tuple(_normalize_run_ids(manifest.get("runs"))),
        metadata=_normalize_metadata(manifest.get("metadata")),
        manifest_path=path,
    )


def _normalize_tags(raw: object) -> list[str]:
    if raw is None:
        return []
    tags: list[str] = []
    if isinstance(raw, str):
        parts = [item.strip() for item in raw.split(",")]
        tags = [item for item in parts if item]
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, str):
                stripped = item.strip()
                if stripped:
                    tags.append(stripped)
    deduped: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        if tag in seen:
            continue
        seen.add(tag)
        deduped.append(tag)
    return deduped


def _normalize_run_ids(raw: object) -> list[str]:
    run_ids: list[str] = []
    if isinstance(raw, list):
        for value in raw:
            if not isinstance(value, str):
                continue
            stripped = value.strip()
            if not stripped:
                continue
            run_ids.append(stripped)
    deduped: list[str] = []
    seen: set[str] = set()
    for run_id in run_ids:
        if run_id in seen:
            continue
        seen.add(run_id)
        deduped.append(run_id)
    return deduped


def _normalize_metadata(raw: object) -> dict[str, Any]:
    if isinstance(raw, dict):
        return {str(key): value for key, value in raw.items()}
    return {}


def _index_experiment_manifest(root: Path, manifest: dict[str, object]) -> None:
    created_at = _as_str(manifest.get("created_at")) or _utc_now_iso()
    updated_at = _as_str(manifest.get("updated_at")) or created_at
    metadata: dict[str, object] = _normalize_metadata(manifest.get("metadata"))
    metadata["hypothesis"] = _as_str(manifest.get("hypothesis"))
    metadata["tags"] = _normalize_tags(manifest.get("tags"))
    metadata["champion_run_id"] = _as_str(manifest.get("champion_run_id"))
    metadata["runs"] = _normalize_run_ids(manifest.get("runs"))
    try:
        upsert_experiment(
            store_root=root,
            experiment_id=_as_str(manifest.get("experiment_id")) or "",
            name=_as_str(manifest.get("name")) or "",
            status=_as_str(manifest.get("status")) or "draft",
            created_at=created_at,
            updated_at=updated_at,
            metadata=metadata,
        )
    except StoreError as exc:
        raise ExperimentError("experiment_store_index_failed") from exc


def _select_best_run(*, root: Path, run_ids: list[str], metric: str) -> tuple[str, float] | None:
    best: tuple[str, float] | None = None
    for run_id in run_ids:
        value = _extract_metric_for_run(root=root, run_id=run_id, metric=metric)
        if value is None:
            continue
        if best is None or value > best[1]:
            best = (run_id, value)
    return best


def _extract_metric_for_run(*, root: Path, run_id: str, metric: str) -> float | None:
    metrics = _load_json_mapping(root / "runs" / run_id / "metrics.json")
    return _extract_metric(metrics, metric)


def _extract_metric(payload: dict[str, object], key: str) -> float | None:
    current: object = payload
    for token in key.split("."):
        if not isinstance(current, dict):
            return None
        if token not in current:
            return None
        current = current[token]
    return _coerce_float(current)


def _coerce_status(value: str) -> ExperimentStatus:
    if value in {"draft", "active", "complete", "archived"}:
        return cast(ExperimentStatus, value)
    return "draft"


def _coerce_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number):
            return number
    return None


def _as_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _as_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return cast(dict[str, object], value)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _resolve_experiment_output_dir(*, output_dir: str | Path | None, store_root: Path) -> Path:
    if output_dir is None:
        return store_root
    output_path = Path(output_dir).expanduser().resolve()
    if output_path != store_root:
        raise ExperimentValidationError("experiment_output_dir_must_match_store_root")
    return output_path


def _render_packed_experiment_markdown(
    *,
    root: Path,
    experiment_dir: Path,
    manifest: dict[str, object],
    notes_body: str,
    run_ids: list[str],
    champion_run_id: str | None,
    packed_at: str,
    output_path: Path,
) -> str:
    header_lines = [
        "# Experiment Pack",
        "",
        "## Pack Metadata",
        f"- experiment_id: `{_as_str(manifest.get('experiment_id')) or experiment_dir.name}`",
        f"- name: {_markdown_inline(_as_str(manifest.get('name')) or experiment_dir.name)}",
        f"- status: `{_as_str(manifest.get('status')) or 'draft'}`",
        f"- hypothesis: {_markdown_inline(_as_str(manifest.get('hypothesis')) or 'n/a')}",
        f"- tags: {_markdown_inline(', '.join(_normalize_tags(manifest.get('tags'))) or 'none')}",
        f"- champion_run_id: `{champion_run_id or 'none'}`",
        f"- packed_at: `{packed_at}`",
        f"- manifest_path: `{manifest_path_relative(experiment_dir, experiment_dir / 'experiment.json')}`",
        f"- source_markdown_path: `{manifest_path_relative(experiment_dir, experiment_dir / 'EXPERIMENT.md')}`",
        f"- output_path: `{manifest_path_relative(experiment_dir, output_path)}`",
        "",
        "## Experiment Notes",
        "",
        notes_body or "_Empty_",
        "",
        "## Target Metrics Summary",
        "",
    ]
    header_lines.extend(_render_target_metrics_table(root=root, run_ids=run_ids, champion_run_id=champion_run_id))
    header_lines.append("")
    return "\n".join(header_lines)


def _render_target_metrics_table(*, root: Path, run_ids: list[str], champion_run_id: str | None) -> list[str]:
    _ = champion_run_id
    columns = ("target", *SHARED_RUN_METRIC_NAMES)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    grouped_metrics: dict[str, list[dict[str, object]]] = defaultdict(list)
    for run_id in run_ids:
        run_dir = root / "runs" / run_id
        run_manifest = _require_json_mapping(run_dir / "run.json", error_code="experiment_pack_run_manifest_invalid")
        run_metrics = _require_json_mapping(run_dir / "metrics.json", error_code="experiment_pack_run_metrics_invalid")
        score_provenance = _load_json_mapping(run_dir / "score_provenance.json")
        normalized = normalize_shared_run_metrics(run_metrics, score_provenance=score_provenance)
        target_name = _target_name_from_run_manifest(run_manifest, run_id=run_id)
        grouped_metrics[target_name].append(normalized)

    for target_name in sorted(grouped_metrics):
        row = [
            _markdown_cell(target_name),
        ]
        for metric_name in SHARED_RUN_METRIC_NAMES:
            row.append(_format_metric_cell(_average_metric(grouped_metrics[target_name], metric_name)))
        lines.append("| " + " | ".join(row) + " |")
    if len(lines) == 2:
        lines.append("| _none_ | " + " | ".join("n/a" for _ in SHARED_RUN_METRIC_NAMES) + " |")
    return lines


def _target_name_from_run_manifest(run_manifest: dict[str, object], *, run_id: str) -> str:
    data = run_manifest.get("data")
    if isinstance(data, dict):
        target_name = _as_str(data.get("target_col"))
        if target_name:
            return target_name
    return run_id


def _average_metric(rows: list[dict[str, object]], metric_name: str) -> float | None:
    values = [_coerce_float(row.get(metric_name)) for row in rows]
    numbers = [value for value in values if value is not None]
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def _format_metric_cell(value: object) -> str:
    number = _coerce_float(value)
    if number is None:
        return "n/a"
    return f"{number:.6f}"


def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|")


def _markdown_inline(value: str) -> str:
    escaped = value.replace("`", "\\`")
    return f"`{escaped}`"


def manifest_path_relative(experiment_dir: Path, path: Path) -> str:
    try:
        return str(path.relative_to(experiment_dir))
    except ValueError:
        return str(path)
