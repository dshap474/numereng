"""Safe pruning for heavyweight persisted prediction artifacts."""

from __future__ import annotations

import json
import shutil
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from numereng.features.store.predictions import run_has_persisted_predictions
from numereng.features.store.service import StoreError, index_run, resolve_store_root


@dataclass(frozen=True)
class StorePrunePredictionsRun:
    """One run whose predictions were, or would be, pruned."""

    run_id: str
    bytes: int
    predictions_dir: Path


@dataclass(frozen=True)
class StorePrunePredictionsExcluded:
    """One selected run excluded from prediction pruning."""

    run_id: str
    reason: str


@dataclass(frozen=True)
class StorePrunePredictionsResult:
    """Summary for a prediction-pruning dry-run or apply operation."""

    store_root: Path
    dry_run: bool
    candidate_count: int
    pruned_count: int
    excluded_count: int
    reclaimable_bytes: int
    reclaimed_bytes: int
    pruned: tuple[StorePrunePredictionsRun, ...]
    excluded: tuple[StorePrunePredictionsExcluded, ...]


@dataclass(frozen=True)
class _IndexedRun:
    run_id: str
    status: str
    run_path: Path
    experiment_id: str | None


def prune_predictions(
    *,
    store_root: str | Path = ".numereng",
    run_ids: Sequence[str] | None = None,
    experiment_id: str | None = None,
    all_runs: bool = False,
    apply: bool = False,
) -> StorePrunePredictionsResult:
    """Dry-run or delete persisted predictions directories for safe runs."""

    normalized_run_ids = _normalize_run_ids(run_ids or ())
    scope_flags = int(bool(normalized_run_ids)) + int(experiment_id is not None) + int(all_runs)
    if scope_flags != 1:
        raise StoreError("store_prune_predictions_scope_invalid")

    root = resolve_store_root(store_root)
    indexed_runs = _load_indexed_runs(root)
    target_experiment_id = _ensure_safe_id(experiment_id, field="experiment_id") if experiment_id is not None else None
    selected_run_ids = _select_run_ids(
        root=root,
        indexed_runs=indexed_runs,
        run_ids=normalized_run_ids,
        experiment_id=target_experiment_id,
        all_runs=all_runs,
    )
    champion_run_ids = _collect_champion_run_ids(root)
    package_run_ids = _collect_submission_package_run_ids(root)

    prunable: list[StorePrunePredictionsRun] = []
    excluded: list[StorePrunePredictionsExcluded] = []

    for run_id in selected_run_ids:
        indexed = indexed_runs.get(run_id)
        if indexed is None:
            excluded.append(StorePrunePredictionsExcluded(run_id=run_id, reason="not_indexed"))
            continue

        exclusion = _exclusion_reason(
            root=root,
            indexed=indexed,
            champion_run_ids=champion_run_ids,
            package_run_ids=package_run_ids,
        )
        if exclusion is not None:
            excluded.append(StorePrunePredictionsExcluded(run_id=run_id, reason=exclusion))
            continue

        predictions_dir = indexed.run_path / "artifacts" / "predictions"
        prunable.append(
            StorePrunePredictionsRun(
                run_id=run_id,
                bytes=_directory_size_bytes(predictions_dir),
                predictions_dir=predictions_dir,
            )
        )

    reclaimable_bytes = sum(item.bytes for item in prunable)
    reclaimed_bytes = 0

    if apply:
        for item in prunable:
            try:
                shutil.rmtree(item.predictions_dir)
            except OSError as exc:
                raise StoreError(f"store_prune_predictions_delete_failed:{item.run_id}:{exc}") from exc
            try:
                index_run(store_root=root, run_id=item.run_id)
            except StoreError as exc:
                raise StoreError(f"store_prune_predictions_reindex_failed:{item.run_id}:{exc}") from exc
            reclaimed_bytes += item.bytes

    return StorePrunePredictionsResult(
        store_root=root,
        dry_run=not apply,
        candidate_count=len(selected_run_ids),
        pruned_count=len(prunable),
        excluded_count=len(excluded),
        reclaimable_bytes=reclaimable_bytes,
        reclaimed_bytes=reclaimed_bytes,
        pruned=tuple(prunable),
        excluded=tuple(excluded),
    )


def _normalize_run_ids(run_ids: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    normalized: list[str] = []
    for run_id in run_ids:
        safe_run_id = _ensure_safe_id(run_id, field="run_id")
        if safe_run_id in seen:
            continue
        seen.add(safe_run_id)
        normalized.append(safe_run_id)
    return tuple(normalized)


def _ensure_safe_id(value: str | None, *, field: str) -> str:
    if value is None:
        raise StoreError(f"store_prune_predictions_{field}_invalid")
    candidate = value.strip()
    if not candidate or "/" in candidate or "\\" in candidate or candidate in {".", ".."}:
        raise StoreError(f"store_prune_predictions_{field}_invalid:{value}")
    return candidate


def _load_indexed_runs(root: Path) -> dict[str, _IndexedRun]:
    db_path = root / "numereng.db"
    if not db_path.is_file():
        return {}

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        with conn:
            rows = conn.execute("SELECT run_id, status, run_path, experiment_id FROM runs ORDER BY run_id").fetchall()
    except sqlite3.Error as exc:
        raise StoreError(f"store_prune_predictions_index_read_failed:{exc}") from exc
    finally:
        try:
            conn.close()
        except UnboundLocalError:
            pass

    indexed: dict[str, _IndexedRun] = {}
    for row in rows:
        run_id = _as_nonempty_str(row["run_id"])
        status = _as_nonempty_str(row["status"])
        run_path = _as_nonempty_str(row["run_path"])
        if run_id is None or status is None or run_path is None:
            continue
        indexed[run_id] = _IndexedRun(
            run_id=run_id,
            status=status,
            run_path=Path(run_path),
            experiment_id=_as_nonempty_str(row["experiment_id"]),
        )
    return indexed


def _select_run_ids(
    *,
    root: Path,
    indexed_runs: Mapping[str, _IndexedRun],
    run_ids: tuple[str, ...],
    experiment_id: str | None,
    all_runs: bool,
) -> tuple[str, ...]:
    if run_ids:
        return run_ids
    if all_runs:
        return tuple(sorted(indexed_runs))
    if experiment_id is None:
        return ()

    manifest_run_ids = _load_experiment_manifest_run_ids(root=root, experiment_id=experiment_id)
    if manifest_run_ids:
        return manifest_run_ids
    return tuple(run_id for run_id, item in sorted(indexed_runs.items()) if item.experiment_id == experiment_id)


def _load_experiment_manifest_run_ids(*, root: Path, experiment_id: str) -> tuple[str, ...]:
    for manifest_path in (
        root / "experiments" / experiment_id / "experiment.json",
        root / "experiments" / "_archive" / experiment_id / "experiment.json",
    ):
        payload = _read_json_mapping(manifest_path)
        if not payload:
            continue
        return _normalize_manifest_run_ids(payload.get("runs"))
    return ()


def _normalize_manifest_run_ids(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    run_ids: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str) or not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        run_ids.append(item)
    return tuple(run_ids)


def _collect_champion_run_ids(root: Path) -> dict[str, str]:
    champions: dict[str, str] = {}
    for manifest_path in _iter_experiment_manifests(root):
        payload = _read_json_mapping(manifest_path)
        champion_run_id = _as_nonempty_str(payload.get("champion_run_id"))
        if champion_run_id is None:
            continue
        experiment_id = _as_nonempty_str(payload.get("experiment_id")) or manifest_path.parent.name
        champions.setdefault(champion_run_id, experiment_id)
    return champions


def _iter_experiment_manifests(root: Path) -> tuple[Path, ...]:
    experiments_root = root / "experiments"
    if not experiments_root.is_dir():
        return ()
    return tuple(sorted(experiments_root.rglob("experiment.json"), key=lambda item: item.as_posix()))


def _collect_submission_package_run_ids(root: Path) -> dict[str, str]:
    packages_root = root / "experiments"
    if not packages_root.is_dir():
        return {}

    referenced: dict[str, str] = {}
    for package_path in sorted(
        packages_root.rglob("submission_packages/*/package.json"), key=lambda item: item.as_posix()
    ):
        payload = _read_json_mapping(package_path)
        for run_id in _extract_package_run_ids(payload):
            try:
                source = package_path.relative_to(root).as_posix()
            except ValueError:
                source = str(package_path)
            referenced.setdefault(run_id, source)
    return referenced


def _extract_package_run_ids(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in {"run_id", "source_run_id"}:
                run_id = _as_nonempty_str(child)
                if run_id is not None:
                    found.add(run_id)
            elif key in {"run_ids", "source_run_ids"} and isinstance(child, list):
                found.update(item for item in child if isinstance(item, str) and item)
            found.update(_extract_package_run_ids(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_extract_package_run_ids(child))
    return found


def _exclusion_reason(
    *,
    root: Path,
    indexed: _IndexedRun,
    champion_run_ids: Mapping[str, str],
    package_run_ids: Mapping[str, str],
) -> str | None:
    if indexed.run_id in champion_run_ids:
        return f"champion_run:{champion_run_ids[indexed.run_id]}"
    if indexed.run_id in package_run_ids:
        return f"submission_package:{package_run_ids[indexed.run_id]}"
    if indexed.status != "FINISHED":
        return f"status_not_finished:{indexed.status}"
    if not indexed.run_path.is_dir():
        return "run_dir_missing"

    manifest, manifest_error = _load_run_manifest(indexed.run_path / "run.json")
    if manifest_error is not None:
        return manifest_error
    if _as_nonempty_str(manifest.get("status")) != "FINISHED":
        return f"status_not_finished:{_as_nonempty_str(manifest.get('status')) or 'UNKNOWN'}"

    predictions_dir = indexed.run_path / "artifacts" / "predictions"
    if predictions_dir.is_symlink():
        return "predictions_dir_unsafe"
    if not predictions_dir.is_dir():
        return "predictions_dir_missing"
    if not run_has_persisted_predictions(root=root, run_id=indexed.run_id, run_manifest=manifest):
        return "persisted_predictions_missing"
    return None


def _load_run_manifest(path: Path) -> tuple[dict[str, object], str | None]:
    if not path.is_file():
        return {}, "run_manifest_missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, "run_manifest_invalid"
    if not isinstance(payload, dict):
        return {}, "run_manifest_invalid"
    return payload, None


def _read_json_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _directory_size_bytes(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_dir() and not item.is_symlink():
            continue
        try:
            stat = item.lstat() if item.is_symlink() else item.stat()
        except OSError as exc:
            raise StoreError(f"store_prune_predictions_stat_failed:{path}:{exc}") from exc
        total += stat.st_size
    return total


def _as_nonempty_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


__all__ = [
    "StorePrunePredictionsExcluded",
    "StorePrunePredictionsResult",
    "StorePrunePredictionsRun",
    "prune_predictions",
]
