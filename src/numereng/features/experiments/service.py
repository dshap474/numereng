"""Experiment lifecycle services."""

from __future__ import annotations

import json
import math
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from numereng.features.experiments.contracts import (
    ExperimentArchiveResult,
    ExperimentPackResult,
    ExperimentPromotionResult,
    ExperimentRecord,
    ExperimentReport,
    ExperimentReportRow,
    ExperimentStatus,
    ExperimentTrainResult,
)
from numereng.features.scoring.summary_metrics import SHARED_RUN_METRIC_NAMES, normalize_shared_run_metrics
from numereng.features.store import StoreError, index_run, resolve_store_root, upsert_experiment
from numereng.features.training import TrainingProfile, run_training

_SAFE_ID = re.compile(r"^[\w\-.]+$")
_EXPERIMENT_ID_FORMAT = re.compile(r"^\d{4}-\d{2}-\d{2}_[a-z0-9][a-z0-9-]*$")
_DEFAULT_PROMOTION_METRIC = "bmc_last_200_eras.mean"
_ARCHIVE_DIRNAME = "_archive"
_PRE_ARCHIVE_STATUS_KEY = "pre_archive_status"
_PACK_OUTPUT_NAME = "EXPERIMENT.pack.md"


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

    manifest = _load_manifest(manifest_path)
    _ensure_experiment_mutable(manifest)
    if profile is None:
        result = run_training(
            config_path=config_resolved,
            output_dir=output_dir_resolved,
            engine_mode=engine_mode,
            window_size_eras=window_size_eras,
            embargo_eras=embargo_eras,
            experiment_id=safe_experiment_id,
        )
    else:
        result = run_training(
            config_path=config_resolved,
            output_dir=output_dir_resolved,
            profile=profile,
            engine_mode=engine_mode,
            window_size_eras=window_size_eras,
            embargo_eras=embargo_eras,
            experiment_id=safe_experiment_id,
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

    return ExperimentTrainResult(
        experiment_id=safe_experiment_id,
        run_id=result.run_id,
        predictions_path=result.predictions_path,
        results_path=result.results_path,
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


def _resolved_experiment_dir(root: Path, experiment_id: str) -> Path:
    paths = _experiment_paths(root, experiment_id)
    if paths.active_dir is None:
        raise ExperimentNotFoundError(f"experiment_not_found:{experiment_id}")
    return paths.active_dir


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


def _save_experiment_doc(path: Path, manifest: dict[str, object]) -> None:
    hypothesis = _as_str(manifest.get("hypothesis")) or "n/a"
    tags = _normalize_tags(manifest.get("tags"))
    body = "\n".join(
        [
            f"# {manifest.get('experiment_id')}",
            "",
            "## Summary",
            f"- name: {manifest.get('name')}",
            f"- hypothesis: {hypothesis}",
            f"- status: {manifest.get('status')}",
            f"- tags: {', '.join(tags) if tags else 'none'}",
            "",
            "## Notes",
            "- Track experiment findings and next actions in this file.",
            "",
        ]
    )
    path.write_text(body, encoding="utf-8")


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
        "## Run Metrics Summary",
        "",
    ]
    header_lines.extend(_render_run_metrics_table(root=root, run_ids=run_ids, champion_run_id=champion_run_id))
    header_lines.append("")
    return "\n".join(header_lines)


def _render_run_metrics_table(*, root: Path, run_ids: list[str], champion_run_id: str | None) -> list[str]:
    columns = ("run_id", "status", "created_at", "champion", *SHARED_RUN_METRIC_NAMES)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for run_id in run_ids:
        run_dir = root / "runs" / run_id
        run_manifest = _require_json_mapping(run_dir / "run.json", error_code="experiment_pack_run_manifest_invalid")
        run_metrics = _require_json_mapping(run_dir / "metrics.json", error_code="experiment_pack_run_metrics_invalid")
        score_provenance = _load_json_mapping(run_dir / "score_provenance.json")
        normalized = normalize_shared_run_metrics(run_metrics, score_provenance=score_provenance)

        row = [
            _markdown_cell(run_id),
            _markdown_cell(_as_str(run_manifest.get("status")) or "n/a"),
            _markdown_cell(_as_str(run_manifest.get("created_at")) or "n/a"),
            _markdown_cell("yes" if champion_run_id == run_id else "no"),
        ]
        for metric_name in SHARED_RUN_METRIC_NAMES:
            row.append(_format_metric_cell(normalized.get(metric_name)))
        lines.append("| " + " | ".join(row) + " |")
    if len(lines) == 2:
        lines.append("| _none_ | n/a | n/a | no | " + " | ".join("n/a" for _ in SHARED_RUN_METRIC_NAMES) + " |")
    return lines


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
