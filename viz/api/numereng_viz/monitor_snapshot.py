"""Normalized monitor snapshots for one numereng store and merged workspaces."""

from __future__ import annotations

import copy
import json
import logging
import socket
import subprocess
import threading
import time
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from numereng.features.cloud.aws import CloudAwsManagedService
from numereng.features.remote_ops.bootstrap_state import load_viz_bootstrap_state
from numereng.features.store.layout import resolve_cloud_run_state_path
from numereng.features.telemetry import reconcile_run_lifecycles
from numereng.platform import load_remote_targets
from numereng.platform.remotes.contracts import SshRemoteTargetProfile
from numereng.platform.remotes.ssh import build_monitor_snapshot_command, build_ssh_command
from numereng_viz.store_adapter import (
    _ATTENTION_STATUS_RANK,
    VizStoreAdapter,
    VizStoreConfig,
    _parse_json_object,
    _to_non_empty_str,
)

logger = logging.getLogger(__name__)

_ACTIVE_MONITOR_STATUSES = {"queued", "starting", "running", "canceling"}
_TERMINAL_MONITOR_STATUSES = {"completed", "failed", "canceled", "stale"}
_REMOTE_CACHE_TTL_SECONDS = 5.0
_CLOUD_PHASE_PROGRESS = {
    "starting": 8.0,
    "downloading": 22.0,
    "training": 68.0,
    "uploading": 92.0,
    "stopping": 96.0,
}


@lru_cache(maxsize=1)
def _cloud_state_store() -> Any:
    from numereng.features.cloud.aws.managed_state_store import CloudAwsStateStore

    return CloudAwsStateStore()


def build_monitor_snapshot(
    *,
    store_root: str | Path = ".numereng",
    refresh_cloud: bool = True,
) -> dict[str, Any]:
    """Build one normalized read-only monitor snapshot for a single store."""

    reconcile_run_lifecycles(store_root=store_root, active_only=True)
    adapter = VizStoreAdapter(VizStoreConfig.from_env(store_root=store_root))
    overview = adapter.get_experiments_overview()
    source = _local_source(adapter.store_root)
    execution_by_run_id = _execution_summary_by_run_id(adapter)

    experiments = [
        _decorate_experiment(dict(item), source=source, detail_href=True) for item in overview["experiments"]
    ]
    experiment_ids = {
        str(item.get("experiment_id")) for item in experiments if isinstance(item.get("experiment_id"), str)
    }

    live_experiments = []
    for item in overview["live_experiments"]:
        live_experiments.append(
            _decorate_live_experiment(
                dict(item),
                source=source,
                detail_href=str(item.get("experiment_id")) in experiment_ids,
                execution_by_run_id=execution_by_run_id,
            )
        )

    recent_activity = [
        _decorate_recent_activity(dict(item), source=source, execution_by_run_id=execution_by_run_id)
        for item in overview["recent_activity"]
    ]

    cloud_payload = _cloud_monitor_payload(adapter, refresh_cloud=refresh_cloud)
    if cloud_payload["live_experiments"] or cloud_payload["recent_activity"]:
        _merge_cloud_payload(
            experiments=experiments,
            live_experiments=live_experiments,
            recent_activity=recent_activity,
            cloud_payload=cloud_payload,
            source=source,
        )

    live_experiments = _sort_live_experiments(live_experiments)
    recent_activity = _sort_recent_activity(recent_activity)[:8]
    experiments = _sort_experiments(experiments)
    summary = _recompute_summary(
        experiments=experiments,
        live_experiments=live_experiments,
    )
    return {
        "generated_at": _utc_now_iso(),
        "source": source,
        "summary": summary,
        "experiments": experiments,
        "live_experiments": live_experiments,
        "live_runs": _flatten_live_runs(live_experiments),
        "recent_activity": recent_activity,
    }


def merge_monitor_snapshots(local_snapshot: dict[str, Any], remote_snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge the local store snapshot with zero or more SSH remote snapshots."""

    experiments = [copy.deepcopy(item) for item in local_snapshot.get("experiments", [])]
    live_experiments = [copy.deepcopy(item) for item in local_snapshot.get("live_experiments", [])]
    recent_activity = [copy.deepcopy(item) for item in local_snapshot.get("recent_activity", [])]
    sources = [copy.deepcopy(local_snapshot.get("source", {}))]
    local_experiments_by_id = {
        str(item["experiment_id"]): item
        for item in experiments
        if isinstance(item, dict) and isinstance(item.get("experiment_id"), str)
    }
    local_live_by_id = {
        str(item["experiment_id"]): item
        for item in live_experiments
        if isinstance(item, dict) and isinstance(item.get("experiment_id"), str)
    }

    for snapshot in remote_snapshots:
        source = copy.deepcopy(snapshot.get("source", {}))
        sources.append(source)
        for item in snapshot.get("experiments", []):
            if not isinstance(item, dict):
                continue
            experiment_id = _to_non_empty_str(item.get("experiment_id"))
            if experiment_id is not None and experiment_id in local_experiments_by_id:
                _apply_remote_overlay(local_experiments_by_id[experiment_id], item, source=source)
                continue
            experiments.append(copy.deepcopy(item))
        for item in snapshot.get("live_experiments", []):
            if not isinstance(item, dict):
                continue
            experiment_id = _to_non_empty_str(item.get("experiment_id"))
            if experiment_id is not None and experiment_id in local_experiments_by_id:
                _apply_remote_overlay(local_experiments_by_id[experiment_id], item, source=source)
                existing_live = local_live_by_id.get(experiment_id)
                if existing_live is None:
                    canonical_live = _canonicalize_remote_live_experiment(item, source=source)
                    live_experiments.append(canonical_live)
                    local_live_by_id[experiment_id] = canonical_live
                    existing_live = canonical_live
                _merge_remote_live_overlay(existing_live, item, source=source)
                continue
            live_experiments.append(copy.deepcopy(item))
        recent_activity.extend(copy.deepcopy(snapshot.get("recent_activity", [])))

    experiments = _sort_experiments(experiments)
    live_experiments = _sort_live_experiments(live_experiments)
    recent_activity = _sort_recent_activity(recent_activity)[:8]
    live_runs = _flatten_live_runs(live_experiments)
    summary = _recompute_summary(experiments=experiments, live_experiments=live_experiments)

    return {
        "generated_at": _utc_now_iso(),
        "summary": summary,
        "experiments": experiments,
        "live_experiments": live_experiments,
        "live_runs": live_runs,
        "recent_activity": recent_activity,
        "sources": sources,
    }


def _apply_remote_overlay(local_item: dict[str, Any], remote_item: dict[str, Any], *, source: dict[str, Any]) -> None:
    local_item["has_remote_overlay"] = True
    local_item["overlay_source_kind"] = source.get("kind")
    local_item["overlay_source_id"] = source.get("id")
    local_item["overlay_source_label"] = source.get("label")
    local_item["overlay_state"] = source.get("state")
    remote_updated_at = _to_non_empty_str(remote_item.get("updated_at")) or _to_non_empty_str(
        remote_item.get("latest_activity_at")
    )
    if remote_updated_at is not None:
        local_item["overlay_updated_at"] = remote_updated_at
        local_item["latest_activity_at"] = _latest_timestamp(local_item.get("latest_activity_at"), remote_updated_at)


def _canonicalize_remote_live_experiment(item: dict[str, Any], *, source: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(item)
    payload["source_kind"] = "local"
    payload["source_id"] = "local"
    payload["source_label"] = "Local store"
    experiment_id = _to_non_empty_str(payload.get("experiment_id"))
    payload["detail_href"] = f"/experiments/{experiment_id}" if experiment_id is not None else None
    payload["has_remote_overlay"] = True
    payload["overlay_source_kind"] = source.get("kind")
    payload["overlay_source_id"] = source.get("id")
    payload["overlay_source_label"] = source.get("label")
    payload["overlay_state"] = source.get("state")
    return payload


def _merge_remote_live_overlay(
    local_item: dict[str, Any],
    remote_item: dict[str, Any],
    *,
    source: dict[str, Any],
) -> None:
    local_item["has_remote_overlay"] = True
    local_item["overlay_source_kind"] = source.get("kind")
    local_item["overlay_source_id"] = source.get("id")
    local_item["overlay_source_label"] = source.get("label")
    local_item["overlay_state"] = source.get("state")
    local_item["latest_activity_at"] = _latest_timestamp(
        local_item.get("latest_activity_at"),
        remote_item.get("latest_activity_at"),
    )
    local_item["aggregate_progress_percent"] = remote_item.get("aggregate_progress_percent")
    local_item["live_run_count"] = max(
        int(local_item.get("live_run_count") or 0),
        int(remote_item.get("live_run_count") or 0),
    )
    local_item["queued_run_count"] = max(
        int(local_item.get("queued_run_count") or 0),
        int(remote_item.get("queued_run_count") or 0),
    )
    local_item["runs"] = _merge_remote_live_runs(local_item.get("runs", []), remote_item.get("runs", []))


def _merge_remote_live_runs(
    local_runs: list[dict[str, Any]],
    remote_runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = [copy.deepcopy(item) for item in local_runs if isinstance(item, dict)]
    seen_run_ids = {str(item.get("run_id")) for item in merged if isinstance(item.get("run_id"), str)}
    for item in remote_runs:
        if not isinstance(item, dict):
            continue
        run_id = _to_non_empty_str(item.get("run_id"))
        if run_id is None or run_id in seen_run_ids:
            continue
        merged.append(copy.deepcopy(item))
        seen_run_ids.add(run_id)
    return merged


def _latest_timestamp(*values: Any) -> str | None:
    timestamps = [str(value) for value in values if isinstance(value, str) and value.strip()]
    if not timestamps:
        return None
    return max(timestamps)


class RemoteSnapshotCoordinator:
    """Fetch SSH-backed remote store snapshots with one-poll-cycle cache fallback."""

    def __init__(
        self,
        *,
        store_root: str | Path = ".numereng",
        cache_ttl_seconds: float = _REMOTE_CACHE_TTL_SECONDS,
    ) -> None:
        self._store_root = Path(store_root).expanduser().resolve()
        self._cache_ttl_seconds = cache_ttl_seconds
        self._lock = threading.RLock()
        self._cache: dict[str, tuple[float, dict[str, Any]]] = {}

    def fetch_snapshots(self) -> list[dict[str, Any]]:
        targets = load_remote_targets()
        if not targets:
            return []
        bootstrap_state = load_viz_bootstrap_state(store_root=self._store_root)
        bootstrap_by_target_id = {
            item.target.id: item for item in (bootstrap_state.targets if bootstrap_state is not None else ())
        }

        max_workers = min(len(targets), 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._fetch_one, target, bootstrap_by_target_id.get(target.id)): target
                for target in targets
            }
            snapshots: list[dict[str, Any]] = []
            for future, target in futures.items():
                bootstrap = bootstrap_by_target_id.get(target.id)
                try:
                    snapshot = future.result()
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("Remote snapshot fetch failed for %s: %s", target.id, exc)
                    snapshot = self._cached_snapshot(target, bootstrap=bootstrap)
                if snapshot is None:
                    snapshot = _empty_remote_snapshot(target=target, state="unavailable", bootstrap=bootstrap)
                snapshots.append(snapshot)
            return snapshots

    def _fetch_one(self, target: SshRemoteTargetProfile, bootstrap: Any) -> dict[str, Any] | None:
        command = self._ssh_command(target)
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=target.command_timeout_seconds,
            check=False,
        )
        if result.returncode != 0:
            logger.warning("Remote snapshot command failed for %s: %s", target.id, result.stderr.strip())
            return self._cached_snapshot(target, bootstrap=bootstrap)

        raw = _extract_json_object(result.stdout)
        if raw is None:
            logger.warning("Remote snapshot did not return JSON for %s", target.id)
            return self._cached_snapshot(target, bootstrap=bootstrap)

        snapshot = _retag_remote_snapshot(raw, target=target, state="live", bootstrap=bootstrap)
        with self._lock:
            self._cache[target.id] = (time.monotonic(), copy.deepcopy(snapshot))
        return snapshot

    def _cached_snapshot(self, target: SshRemoteTargetProfile, *, bootstrap: Any) -> dict[str, Any] | None:
        with self._lock:
            cached = self._cache.get(target.id)
            if cached is None:
                return None
            fetched_at, snapshot = cached
            if (time.monotonic() - fetched_at) > self._cache_ttl_seconds:
                self._cache.pop(target.id, None)
                return None
            return _retag_remote_snapshot(snapshot, target=target, state="cached", bootstrap=bootstrap)

    def _ssh_command(self, target: SshRemoteTargetProfile) -> list[str]:
        return build_ssh_command(target, build_monitor_snapshot_command(target))


def _local_source(store_root: Path) -> dict[str, Any]:
    return {
        "kind": "local",
        "id": "local",
        "label": "Local store",
        "host": socket.gethostname(),
        "store_root": str(store_root),
        "state": "live",
    }


def _execution_summary_by_run_id(adapter: VizStoreAdapter) -> dict[str, dict[str, str | None]]:
    payload: dict[str, dict[str, str | None]] = {}
    if adapter._table_exists("runs"):
        rows = adapter._query("SELECT run_id, manifest_json FROM runs")
        for row in rows:
            run_id = _to_non_empty_str(row["run_id"])
            if run_id is None or run_id in payload:
                continue
            manifest = _parse_json_object(row["manifest_json"])
            execution = _execution_summary_from_manifest(manifest)
            if execution is not None:
                payload[run_id] = execution
    if adapter._table_exists("run_lifecycles"):
        rows = adapter._query("SELECT run_id, backend FROM run_lifecycles ORDER BY updated_at DESC")
        for row in rows:
            run_id = _to_non_empty_str(row["run_id"])
            if run_id is None or run_id in payload:
                continue
            payload[run_id] = {
                "backend": _to_non_empty_str(row["backend"]) or "local",
                "provider_run_id": None,
            }
    return payload


def _execution_summary_from_manifest(manifest: dict[str, Any]) -> dict[str, str | None] | None:
    execution_raw = manifest.get("execution")
    if not isinstance(execution_raw, dict):
        return None
    backend = _to_non_empty_str(execution_raw.get("backend"))
    provider_run_id = _to_non_empty_str(execution_raw.get("provider_job_id"))
    if backend is None and provider_run_id is None:
        return None
    return {
        "backend": backend or "local",
        "provider_run_id": provider_run_id,
    }


def _decorate_experiment(item: dict[str, Any], *, source: dict[str, Any], detail_href: bool) -> dict[str, Any]:
    item["source_kind"] = source["kind"]
    item["source_id"] = source["id"]
    item["source_label"] = source["label"]
    item["detail_href"] = (
        _detail_href(
            experiment_id=str(item["experiment_id"]),
            source_kind=str(source["kind"]),
            source_id=str(source["id"]),
        )
        if detail_href
        else None
    )
    return item


def _decorate_live_experiment(
    item: dict[str, Any],
    *,
    source: dict[str, Any],
    detail_href: bool,
    execution_by_run_id: dict[str, dict[str, str | None]],
) -> dict[str, Any]:
    item["source_kind"] = source["kind"]
    item["source_id"] = source["id"]
    item["source_label"] = source["label"]
    item["detail_href"] = f"/experiments/{item['experiment_id']}" if detail_href else None
    runs = []
    for run in item.get("runs", []):
        run_item = dict(run)
        run_id = _to_non_empty_str(run_item.get("run_id"))
        execution = execution_by_run_id.get(run_id or "", {})
        run_item["source_kind"] = source["kind"]
        run_item["source_id"] = source["id"]
        run_item["source_label"] = source["label"]
        run_item["backend"] = execution.get("backend") or "local"
        run_item["provider_run_id"] = execution.get("provider_run_id")
        run_item["progress_mode"] = _local_progress_mode(run_item.get("progress_percent"))
        run_item["detail_href"] = item["detail_href"]
        runs.append(run_item)
    item["runs"] = runs
    return item


def _decorate_recent_activity(
    item: dict[str, Any],
    *,
    source: dict[str, Any],
    execution_by_run_id: dict[str, dict[str, str | None]],
) -> dict[str, Any]:
    run_id = _to_non_empty_str(item.get("run_id"))
    execution = execution_by_run_id.get(run_id or "", {})
    item["source_kind"] = source["kind"]
    item["source_id"] = source["id"]
    item["source_label"] = source["label"]
    item["backend"] = execution.get("backend") or "local"
    item["provider_run_id"] = execution.get("provider_run_id")
    item["progress_mode"] = _local_progress_mode(item.get("progress_percent"))
    return item


def _cloud_monitor_payload(adapter: VizStoreAdapter, *, refresh_cloud: bool) -> dict[str, list[dict[str, Any]]]:
    rows = _latest_cloud_job_rows(adapter)
    if not rows:
        return {"live_experiments": [], "recent_activity": []}

    service = CloudAwsManagedService()
    live_runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    recent_activity: list[dict[str, Any]] = []
    for row in rows:
        run_id = _to_non_empty_str(row.get("run_id"))
        if run_id is not None and _run_is_materialized(adapter, run_id):
            continue
        normalized = _normalize_cloud_job_row(adapter, row, service=service, refresh_cloud=refresh_cloud)
        if normalized is None:
            continue
        if normalized["status"] in _ACTIVE_MONITOR_STATUSES:
            live_runs[normalized["experiment_id"]].append(normalized)
        elif normalized["status"] in _TERMINAL_MONITOR_STATUSES:
            recent_activity.append(normalized)

    live_experiments: list[dict[str, Any]] = []
    for experiment_id, runs in live_runs.items():
        latest_activity_at = max((str(item.get("updated_at") or "") for item in runs), default=None)
        queued_count = sum(1 for item in runs if str(item.get("status")) == "queued")
        progress_values = [
            float(item["progress_percent"]) for item in runs if isinstance(item.get("progress_percent"), (int, float))
        ]
        aggregate_progress = (sum(progress_values) / len(progress_values)) if progress_values else None
        experiment_name = str(runs[0].get("experiment_name") or experiment_id)
        detail_href = f"/experiments/{experiment_id}" if runs[0].get("detail_enabled") else None
        live_experiments.append(
            {
                "experiment_id": experiment_id,
                "name": experiment_name,
                "status": runs[0].get("experiment_status") or "active",
                "tags": list(runs[0].get("experiment_tags") or []),
                "live_run_count": len(runs),
                "queued_run_count": queued_count,
                "attention_state": "none",
                "latest_activity_at": latest_activity_at,
                "aggregate_progress_percent": aggregate_progress,
                "runs": [item["run"] for item in runs],
                "source_kind": "local",
                "source_id": "local",
                "source_label": "Local store",
                "detail_href": detail_href,
            }
        )

    return {
        "live_experiments": live_experiments,
        "recent_activity": [item["activity"] for item in recent_activity],
    }


def _latest_cloud_job_rows(adapter: VizStoreAdapter) -> list[dict[str, Any]]:
    if not adapter._table_exists("cloud_jobs"):
        return []
    rows = adapter._query(
        """
        WITH ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY run_id, provider, backend
                    ORDER BY updated_at DESC, finished_at DESC
                ) AS __rn
            FROM cloud_jobs
        )
        SELECT *
        FROM ranked
        WHERE __rn = 1
        ORDER BY updated_at DESC
        """
    )
    return [dict(row) for row in rows]


def _normalize_cloud_job_row(
    adapter: VizStoreAdapter,
    row: dict[str, Any],
    *,
    service: Any,
    refresh_cloud: bool,
) -> dict[str, Any] | None:
    refreshed_row = dict(row)
    backend = _to_non_empty_str(row.get("backend"))
    run_id = _to_non_empty_str(row.get("run_id"))
    provider_job_id = _to_non_empty_str(row.get("provider_job_id"))
    region = _to_non_empty_str(row.get("region"))
    if backend is None or run_id is None or provider_job_id is None:
        return None

    metadata = _parse_json_object(row.get("metadata_json"))
    state_path = _resolve_cloud_state_path(store_root=adapter.store_root, run_id=run_id, metadata=metadata)
    recovered_metadata = _load_cloud_state_metadata(state_path)
    if recovered_metadata:
        metadata.update(recovered_metadata)
    if state_path is not None:
        metadata["state_path"] = str(state_path)
    existing_canonical_status = _canonical_cloud_status(str(refreshed_row.get("status") or "unknown"))
    if (
        refresh_cloud
        and backend in {"sagemaker", "batch"}
        and _should_refresh_cloud_job_status(existing_canonical_status)
    ):
        try:
            from numereng.features.cloud.aws import AwsTrainStatusRequest

            response = service.train_status(
                AwsTrainStatusRequest(
                    store_root=str(adapter.store_root),
                    state_path=str(state_path) if state_path is not None else None,
                    backend=backend,
                    run_id=run_id,
                    training_job_name=provider_job_id if backend == "sagemaker" else None,
                    batch_job_id=provider_job_id if backend == "batch" else None,
                    region=region,
                )
            )
            refreshed_row["status"] = response.result.get("status", refreshed_row.get("status"))
            refreshed_row["updated_at"] = (
                response.state.last_updated_at if response.state else refreshed_row.get("updated_at")
            )
            if backend == "sagemaker":
                refreshed_row["error_message"] = response.result.get("failure_reason")
                refreshed_row["secondary_status"] = response.result.get("secondary_status")
            else:
                refreshed_row["error_message"] = response.result.get("status_reason")
                refreshed_row["secondary_status"] = response.result.get("status_reason")
            if response.state is not None and response.state.metadata:
                metadata.update({key: value for key, value in response.state.metadata.items() if value})
                if state_path is not None:
                    metadata["state_path"] = str(state_path)
        except Exception as exc:  # pragma: no cover - defensive cloud failure path
            logger.warning("Cloud status refresh failed for %s/%s: %s", backend, provider_job_id, exc)

    config_path = _to_non_empty_str(metadata.get("config_path"))
    config_id = _to_non_empty_str(metadata.get("config_id"))
    config_label = Path(config_path).name if config_path else _to_non_empty_str(metadata.get("config_label")) or run_id
    experiment_context = _resolve_cloud_experiment_context(adapter, run_id=run_id, metadata=metadata)
    canonical_status = _canonical_cloud_status(str(refreshed_row.get("status") or "unknown"))
    stage = _to_non_empty_str(refreshed_row.get("secondary_status")) or _to_non_empty_str(
        metadata.get("secondary_status")
    )
    progress_percent, progress_mode = _cloud_progress_state(
        backend=backend,
        canonical_status=canonical_status,
        stage=stage,
        metadata=metadata,
    )
    terminal_reason = _to_non_empty_str(refreshed_row.get("error_message"))
    updated_at = _to_non_empty_str(refreshed_row.get("updated_at")) or _to_non_empty_str(
        refreshed_row.get("finished_at")
    )
    detail_enabled = experiment_context["experiment_id"] in {
        str(item.get("experiment_id")) for item in adapter.list_experiments()
    }

    run_item = {
        "run_id": run_id,
        "job_id": None,
        "config_id": config_id,
        "config_label": config_label,
        "status": canonical_status,
        "current_stage": stage,
        "progress_percent": progress_percent,
        "progress_mode": progress_mode,
        "progress_label": stage or _cloud_progress_label(backend, canonical_status),
        "updated_at": updated_at,
        "terminal_reason": terminal_reason,
        "source_kind": "local",
        "source_id": "local",
        "source_label": "Local store",
        "backend": backend,
        "provider_run_id": provider_job_id,
        "detail_href": f"/experiments/{experiment_context['experiment_id']}" if detail_enabled else None,
    }
    activity_item = {
        "experiment_id": experiment_context["experiment_id"],
        "experiment_name": experiment_context["experiment_name"],
        "run_id": run_id,
        "job_id": None,
        "config_id": config_id,
        "config_label": config_label,
        "status": canonical_status,
        "current_stage": stage,
        "progress_percent": progress_percent,
        "progress_mode": progress_mode,
        "progress_label": stage or _cloud_progress_label(backend, canonical_status),
        "updated_at": updated_at,
        "finished_at": _to_non_empty_str(refreshed_row.get("finished_at")),
        "terminal_reason": terminal_reason,
        "source_kind": "local",
        "source_id": "local",
        "source_label": "Local store",
        "backend": backend,
        "provider_run_id": provider_job_id,
    }
    return {
        "experiment_id": experiment_context["experiment_id"],
        "experiment_name": experiment_context["experiment_name"],
        "experiment_status": experiment_context["experiment_status"],
        "experiment_tags": experiment_context["experiment_tags"],
        "detail_enabled": detail_enabled,
        "status": canonical_status,
        "updated_at": updated_at,
        "run": run_item,
        "activity": activity_item,
    }


def _should_refresh_cloud_job_status(canonical_status: str) -> bool:
    return canonical_status in _ACTIVE_MONITOR_STATUSES


def _local_progress_mode(progress_percent: Any) -> str:
    return "exact" if isinstance(progress_percent, (int, float)) else "indeterminate"


def _cloud_progress_state(
    *,
    backend: str,
    canonical_status: str,
    stage: str | None,
    metadata: dict[str, Any],
) -> tuple[float | None, str]:
    previous = _parse_progress_percent(metadata.get("last_progress_percent"))
    stage_value = _cloud_phase_progress(stage)

    if canonical_status == "queued":
        return 0.0, "estimated"
    if canonical_status == "canceling":
        return previous if previous is not None else (stage_value if stage_value is not None else 96.0), "estimated"
    if canonical_status == "running":
        if stage_value is not None:
            return stage_value, "estimated"
        if previous is not None:
            return previous, "estimated"
        fallback = _cloud_backend_active_default(backend)
        if fallback is not None:
            return fallback, "estimated"
        return None, "indeterminate"
    if canonical_status == "completed":
        return 100.0, "estimated"
    if canonical_status in {"failed", "canceled", "stale"}:
        if previous is not None:
            return previous, "estimated"
        if stage_value is not None:
            return stage_value, "estimated"
        return None, "indeterminate"
    return None, "indeterminate"


def _parse_progress_percent(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return max(0.0, min(100.0, float(value)))
    if isinstance(value, str):
        try:
            return max(0.0, min(100.0, float(value.strip())))
        except ValueError:
            return None
    return None


def _cloud_phase_progress(stage: str | None) -> float | None:
    normalized = _normalize_cloud_phase(stage)
    if normalized is None:
        return None
    return _CLOUD_PHASE_PROGRESS.get(normalized)


def _normalize_cloud_phase(stage: str | None) -> str | None:
    if stage is None:
        return None
    normalized = stage.strip().lower().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    return normalized or None


def _cloud_backend_active_default(backend: str) -> float | None:
    if backend == "batch":
        return 68.0
    return None


def _resolve_cloud_experiment_context(
    adapter: VizStoreAdapter,
    *,
    run_id: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    experiment_id = _to_non_empty_str(metadata.get("experiment_id"))
    config_id = _to_non_empty_str(metadata.get("config_id"))
    config_path = _to_non_empty_str(metadata.get("config_path"))
    if experiment_id is None and adapter._table_exists("runs"):
        row = adapter._query_one("SELECT experiment_id FROM runs WHERE run_id = ?", (run_id,))
        experiment_id = _to_non_empty_str(row["experiment_id"]) if row is not None else None
    if experiment_id is None and adapter._table_exists("run_jobs"):
        row = adapter._query_one(
            """
            SELECT experiment_id, config_id, config_path
            FROM run_jobs
            WHERE canonical_run_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (run_id,),
        )
        if row is not None:
            experiment_id = _to_non_empty_str(row["experiment_id"]) or experiment_id
            config_id = _to_non_empty_str(row["config_id"]) or config_id
            config_path = _to_non_empty_str(row["config_path"]) or config_path
    if experiment_id is None and config_path is not None:
        experiment_id = _infer_experiment_id_from_config_path(config_path, store_root=adapter.store_root)
    if experiment_id is None and config_id is not None:
        experiment_id = _infer_experiment_id_from_config_path(config_id, store_root=adapter.store_root)
    if experiment_id is None:
        experiment_id = f"cloud-{run_id}"

    experiment = next((item for item in adapter.list_experiments() if item.get("experiment_id") == experiment_id), None)
    experiment_name = (
        _to_non_empty_str(metadata.get("experiment_name"))
        or (str(experiment.get("name")) if isinstance(experiment, dict) and experiment.get("name") else None)
        or experiment_id
    )
    experiment_status = (
        str(experiment.get("status"))
        if isinstance(experiment, dict) and isinstance(experiment.get("status"), str)
        else "active"
    )
    experiment_tags = (
        list(experiment.get("tags"))
        if isinstance(experiment, dict) and isinstance(experiment.get("tags"), list)
        else []
    )
    return {
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "experiment_status": experiment_status,
        "experiment_tags": experiment_tags,
    }


def _resolve_cloud_state_path(*, store_root: Path, run_id: str, metadata: dict[str, Any]) -> Path | None:
    explicit = _to_non_empty_str(metadata.get("state_path"))
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    canonical = resolve_cloud_run_state_path(store_root=store_root, provider="aws", run_id=run_id)
    if canonical.is_file():
        return canonical
    legacy = (store_root / "cloud" / f"{run_id}.json").resolve()
    if legacy.is_file():
        return legacy
    return None


def _run_is_materialized(adapter: VizStoreAdapter, run_id: str) -> bool:
    if adapter._table_exists("runs"):
        row = adapter._query_one("SELECT 1 FROM runs WHERE run_id = ? LIMIT 1", (run_id,))
        if row is not None:
            return True
    return (adapter.store_root / "runs" / run_id / "run.json").is_file()


def _load_cloud_state_metadata(state_path: Path | None) -> dict[str, Any]:
    if state_path is None:
        return {}
    try:
        state = _cloud_state_store().load(state_path)
    except ValueError:
        return {}
    if state is None:
        return {}
    return {str(key): value for key, value in state.metadata.items() if value}


def _infer_experiment_id_from_config_path(config_path: str, *, store_root: Path) -> str | None:
    path = Path(config_path)
    if not path.is_absolute():
        path = (store_root / path).resolve()
    try:
        relative = path.relative_to(store_root.resolve())
    except ValueError:
        return None
    parts = list(relative.parts)
    if len(parts) >= 4 and parts[0] == "experiments" and parts[2] == "configs":
        return parts[1]
    return None


def _canonical_cloud_status(status: str) -> str:
    normalized = status.strip().lower()
    if normalized in {"queued", "submitted", "pending", "created", "validating"}:
        return "queued"
    if normalized in {"starting", "inprogress", "in_progress", "running", "downloading", "training"}:
        return "running"
    if normalized in {"stopping"}:
        return "canceling"
    if normalized in {"stopped", "cancelled", "canceled", "terminating", "terminated"}:
        return "canceled"
    if normalized in {"completed", "complete", "succeeded", "success"}:
        return "completed"
    if normalized in {"failed", "error"}:
        return "failed"
    return normalized or "unknown"


def _cloud_progress_label(backend: str, status: str) -> str | None:
    if status == "queued":
        return f"{backend} queued"
    if status == "running":
        return f"{backend} active"
    return None


def _merge_cloud_payload(
    *,
    experiments: list[dict[str, Any]],
    live_experiments: list[dict[str, Any]],
    recent_activity: list[dict[str, Any]],
    cloud_payload: dict[str, list[dict[str, Any]]],
    source: dict[str, Any],
) -> None:
    experiment_by_id = {
        str(item["experiment_id"]): item for item in experiments if isinstance(item.get("experiment_id"), str)
    }
    live_by_id = {
        str(item["experiment_id"]): item for item in live_experiments if isinstance(item.get("experiment_id"), str)
    }

    for live_item in cloud_payload["live_experiments"]:
        experiment_id = str(live_item["experiment_id"])
        existing = live_by_id.get(experiment_id)
        if existing is None:
            live_experiments.append(live_item)
            live_by_id[experiment_id] = live_item
        else:
            existing["runs"].extend(live_item["runs"])
            existing["live_run_count"] = int(existing.get("live_run_count", 0)) + int(
                live_item.get("live_run_count", 0)
            )
            existing["queued_run_count"] = int(existing.get("queued_run_count", 0)) + int(
                live_item.get("queued_run_count", 0)
            )
            existing["latest_activity_at"] = max(
                str(existing.get("latest_activity_at") or ""),
                str(live_item.get("latest_activity_at") or ""),
            )
            existing["aggregate_progress_percent"] = _average_progress(existing["runs"])

        experiment = experiment_by_id.get(experiment_id)
        if experiment is not None:
            experiment["has_live"] = True
            experiment["live_run_count"] = int(experiment.get("live_run_count", 0)) + int(
                live_item.get("live_run_count", 0)
            )
            experiment["latest_activity_at"] = max(
                str(experiment.get("latest_activity_at") or ""),
                str(live_item.get("latest_activity_at") or ""),
            )
            experiment["source_kind"] = source["kind"]
            experiment["source_id"] = source["id"]
            experiment["source_label"] = source["label"]

    recent_activity.extend(cloud_payload["recent_activity"])


def _recompute_summary(*, experiments: list[dict[str, Any]], live_experiments: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "total_experiments": len(experiments),
        "active_experiments": sum(1 for item in experiments if str(item.get("status")) == "active"),
        "completed_experiments": sum(1 for item in experiments if str(item.get("status")) in {"complete", "completed"}),
        "live_experiments": len(live_experiments),
        "live_runs": sum(int(item.get("live_run_count") or 0) for item in live_experiments),
        "queued_runs": sum(int(item.get("queued_run_count") or 0) for item in live_experiments),
        "attention_count": sum(1 for item in experiments if str(item.get("attention_state") or "none") != "none"),
    }


def _sort_experiments(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        items,
        key=lambda item: (
            int(bool(item.get("has_live"))),
            _ATTENTION_STATUS_RANK.get(str(item.get("attention_state") or "none"), 0),
            str(item.get("latest_activity_at") or item.get("updated_at") or item.get("created_at") or ""),
            str(item.get("experiment_id") or ""),
        ),
        reverse=True,
    )


def _sort_live_experiments(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        items,
        key=lambda item: (
            _ATTENTION_STATUS_RANK.get(str(item.get("attention_state") or "none"), 0),
            str(item.get("latest_activity_at") or ""),
            str(item.get("experiment_id") or ""),
        ),
        reverse=True,
    )


def _sort_recent_activity(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        items,
        key=lambda item: (
            str(item.get("updated_at") or item.get("finished_at") or ""),
            _ATTENTION_STATUS_RANK.get(str(item.get("status") or "none"), 0),
        ),
        reverse=True,
    )


def _flatten_live_runs(live_experiments: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for experiment in live_experiments:
        for run in experiment.get("runs", []):
            item = dict(run)
            item["experiment_id"] = experiment.get("experiment_id")
            item["experiment_name"] = experiment.get("name")
            rows.append(item)
    return rows


def _average_progress(runs: Iterable[dict[str, Any]]) -> float | None:
    values = [
        float(item["progress_percent"]) for item in runs if isinstance(item.get("progress_percent"), (int, float))
    ]
    if not values:
        return None
    return max(0.0, min(100.0, sum(values) / len(values)))


def _retag_remote_snapshot(
    snapshot: dict[str, Any],
    *,
    target: SshRemoteTargetProfile,
    state: str,
    bootstrap: Any = None,
) -> dict[str, Any]:
    result = copy.deepcopy(snapshot)
    result["source"] = _remote_source(target=target, state=state, bootstrap=bootstrap)
    for item in result.get("experiments", []):
        item["source_kind"] = "ssh"
        item["source_id"] = target.id
        item["source_label"] = target.label
        experiment_id = _to_non_empty_str(item.get("experiment_id"))
        item["detail_href"] = (
            _detail_href(experiment_id=experiment_id, source_kind="ssh", source_id=target.id)
            if experiment_id is not None
            else None
        )
    for item in result.get("live_experiments", []):
        item["source_kind"] = "ssh"
        item["source_id"] = target.id
        item["source_label"] = target.label
        experiment_id = _to_non_empty_str(item.get("experiment_id"))
        item["detail_href"] = (
            _detail_href(experiment_id=experiment_id, source_kind="ssh", source_id=target.id)
            if experiment_id is not None
            else None
        )
        for run in item.get("runs", []):
            run["source_kind"] = "ssh"
            run["source_id"] = target.id
            run["source_label"] = target.label
            run["detail_href"] = item["detail_href"]
    for item in result.get("recent_activity", []):
        item["source_kind"] = "ssh"
        item["source_id"] = target.id
        item["source_label"] = target.label
    return result


def _remote_source(*, target: SshRemoteTargetProfile, state: str, bootstrap: Any) -> dict[str, Any]:
    source = {
        "kind": "ssh",
        "id": target.id,
        "label": target.label,
        "host": target.ssh_config_host or target.host,
        "store_root": target.store_root,
        "state": state,
    }
    if bootstrap is None:
        return source
    source["bootstrap_status"] = getattr(bootstrap, "bootstrap_status", None)
    source["last_bootstrap_at"] = getattr(bootstrap, "last_bootstrap_at", None)
    source["last_bootstrap_error"] = getattr(bootstrap, "last_bootstrap_error", None)
    return source


def _detail_href(*, experiment_id: str, source_kind: str, source_id: str) -> str:
    if source_kind == "local" and source_id == "local":
        return f"/experiments/{experiment_id}"
    params = urlencode({"source_kind": source_kind, "source_id": source_id})
    return f"/experiments/{experiment_id}?{params}"


def _empty_remote_snapshot(*, target: SshRemoteTargetProfile, state: str, bootstrap: Any) -> dict[str, Any]:
    return {
        "generated_at": _utc_now_iso(),
        "source": _remote_source(target=target, state=state, bootstrap=bootstrap),
        "summary": {
            "total_experiments": 0,
            "active_experiments": 0,
            "completed_experiments": 0,
            "live_experiments": 0,
            "live_runs": 0,
            "queued_runs": 0,
            "attention_count": 0,
        },
        "experiments": [],
        "live_experiments": [],
        "live_runs": [],
        "recent_activity": [],
    }


def _extract_json_object(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


__all__ = [
    "RemoteSnapshotCoordinator",
    "build_monitor_snapshot",
    "merge_monitor_snapshots",
]
