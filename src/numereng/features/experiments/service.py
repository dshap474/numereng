"""Experiment lifecycle services."""

from __future__ import annotations

import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from numereng.features.experiments.contracts import (
    ExperimentPromotionResult,
    ExperimentRecord,
    ExperimentReport,
    ExperimentReportRow,
    ExperimentStatus,
    ExperimentTrainResult,
)
from numereng.features.store import StoreError, index_run, resolve_store_root, upsert_experiment
from numereng.features.training import TrainingProfile, run_training

_SAFE_ID = re.compile(r"^[\w\-.]+$")
_EXPERIMENT_ID_FORMAT = re.compile(r"^\d{4}-\d{2}-\d{2}_[a-z0-9][a-z0-9-]*$")
_DEFAULT_PROMOTION_METRIC = "bmc_last_200_eras.mean"


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
    exp_dir = _experiment_dir(root, safe_experiment_id)
    manifest_path = exp_dir / "experiment.json"
    if manifest_path.exists():
        raise ExperimentAlreadyExistsError(f"experiment_already_exists:{safe_experiment_id}")

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
    experiments_dir = root / "experiments"
    if not experiments_dir.is_dir():
        return ()

    items: list[ExperimentRecord] = []
    for experiment_dir in sorted(experiments_dir.iterdir(), key=lambda path: path.name):
        if not experiment_dir.is_dir():
            continue
        manifest_path = experiment_dir / "experiment.json"
        if not manifest_path.is_file():
            continue
        manifest = _load_manifest(manifest_path)
        record = _manifest_to_record(manifest_path, manifest)
        if status is not None and record.status != status:
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
    manifest_path = _experiment_dir(root, safe_experiment_id) / "experiment.json"
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
    manifest_path = _experiment_dir(root, safe_experiment_id) / "experiment.json"
    if not manifest_path.is_file():
        raise ExperimentNotFoundError(f"experiment_not_found:{safe_experiment_id}")

    config_resolved = Path(config_path).expanduser().resolve()
    if not config_resolved.is_file():
        raise ExperimentValidationError(f"experiment_config_not_found:{config_resolved}")
    output_dir_resolved = _resolve_experiment_output_dir(output_dir=output_dir, store_root=root)

    manifest = _load_manifest(manifest_path)
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
    manifest_path = _experiment_dir(root, safe_experiment_id) / "experiment.json"
    if not manifest_path.is_file():
        raise ExperimentNotFoundError(f"experiment_not_found:{safe_experiment_id}")

    manifest = _load_manifest(manifest_path)
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
    manifest_path = _experiment_dir(root, safe_experiment_id) / "experiment.json"
    if not manifest_path.is_file():
        raise ExperimentNotFoundError(f"experiment_not_found:{safe_experiment_id}")
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


def _ensure_safe_id(value: str) -> str:
    if not value or not _SAFE_ID.match(value):
        raise ExperimentValidationError(f"experiment_id_invalid:{value}")
    return value


def _ensure_experiment_id_format(experiment_id: str) -> None:
    if not _EXPERIMENT_ID_FORMAT.match(experiment_id):
        raise ExperimentValidationError("experiment_id_format_invalid:expected_YYYY-MM-DD_slug")


def _experiment_dir(root: Path, experiment_id: str) -> Path:
    return root / "experiments" / experiment_id


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
