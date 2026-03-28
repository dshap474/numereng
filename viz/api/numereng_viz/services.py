"""Service layer for viz API routes."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from numereng.features.telemetry import (
    get_run_lifecycle as get_run_lifecycle_record,
)
from numereng.features.telemetry import (
    reconcile_run_lifecycles,
)
from numereng_viz.monitor_snapshot import RemoteSnapshotCoordinator, build_monitor_snapshot, merge_monitor_snapshots
from numereng_viz.store_adapter import PerEraCorrLoadResult, VizStoreAdapter

TOP_METRIC_NAMES = [
    "bmc_last_200_eras_mean",
    "bmc_mean",
    "corr_sharpe",
    "corr_mean",
    "mmc_mean",
]


class VizService:
    """Thin orchestration layer over the store adapter."""

    def __init__(self, adapter: VizStoreAdapter) -> None:
        self.adapter = adapter
        self.remote_snapshots = RemoteSnapshotCoordinator(store_root=self.adapter.store_root)

    def list_experiments(self) -> list[dict[str, Any]]:
        return self.adapter.list_experiments()

    def get_experiments_overview(self) -> dict[str, Any]:
        local_snapshot = build_monitor_snapshot(store_root=self.adapter.store_root, refresh_cloud=True)
        remote_snapshots = self.remote_snapshots.fetch_snapshots()
        return merge_monitor_snapshots(local_snapshot, remote_snapshots)

    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        return self.adapter.get_experiment(experiment_id)

    def list_experiment_configs(
        self,
        experiment_id: str,
        *,
        q: str | None,
        include_incompatible: bool,
        limit: int,
        offset: int,
    ) -> dict[str, Any]:
        payload = self.adapter.list_experiment_configs(
            experiment_id,
            q=q,
            runnable_only=not include_incompatible,
            limit=limit,
            offset=offset,
        )
        for item in payload["items"]:
            item["mtime"] = self._mtime_to_iso(item.get("mtime"))
        return payload

    def list_configs(
        self,
        *,
        q: str | None,
        experiment_id: str | None,
        model_type: str | None,
        target: str | None,
        limit: int,
        offset: int,
    ) -> dict[str, Any]:
        payload = self.adapter.list_all_configs(
            q=q,
            experiment_id=experiment_id,
            model_type=model_type,
            target=target,
            limit=limit,
            offset=offset,
        )

        config_ids = [item["config_id"] for item in payload["items"] if isinstance(item.get("config_id"), str)]
        linked_runs = self.adapter.linked_runs_for_configs(config_ids)
        run_metrics = self.adapter.get_metrics_for_runs(list(linked_runs.values()), TOP_METRIC_NAMES)

        for item in payload["items"]:
            item["mtime"] = self._mtime_to_iso(item.get("mtime"))
            config_id = item.get("config_id")
            run_id = linked_runs.get(config_id) if isinstance(config_id, str) else None
            item["linked_run_id"] = run_id
            item["linked_metrics"] = run_metrics.get(run_id) if run_id else None
        return payload

    def compare_configs(self, config_ids: list[str]) -> dict[str, Any]:
        payload: list[dict[str, Any]] = []
        for config_id in config_ids:
            payload.append(
                {
                    "config_id": config_id,
                    "yaml": self.adapter.read_config_yaml(config_id),
                }
            )
        return {"configs": payload}

    def list_experiment_runs(self, experiment_id: str) -> list[dict[str, Any]]:
        return self.adapter.list_experiment_runs(experiment_id)

    def list_experiment_round_results(self, experiment_id: str) -> list[dict[str, Any]]:
        return self.adapter.list_experiment_round_results(experiment_id)

    def list_run_jobs(
        self,
        *,
        experiment_id: str | None,
        status: str | None,
        limit: int,
        offset: int,
        include_attempts: bool,
    ) -> dict[str, Any]:
        repair = reconcile_run_lifecycles(store_root=self.adapter.store_root, active_only=True)
        items = self.adapter.list_run_jobs(
            experiment_id=experiment_id,
            status=status,
            limit=limit,
            offset=offset,
            include_attempts=include_attempts,
        )
        total = self.adapter.count_run_jobs(
            experiment_id=experiment_id,
            status=status,
            include_attempts=include_attempts,
        )
        return {
            "items": items,
            "total": total,
            "reconciled_stale_count": repair.reconciled_stale_count,
            "reconciled_canceled_count": repair.reconciled_canceled_count,
        }

    def get_batch_jobs(self, batch_id: str) -> dict[str, Any]:
        items = self.adapter.get_run_job_batch(batch_id)
        return {
            "batch_id": batch_id,
            "items": items,
            "total": len(items),
        }

    def get_run_job(self, job_id: str) -> dict[str, Any] | None:
        reconcile_run_lifecycles(store_root=self.adapter.store_root, active_only=True)
        return self.adapter.get_run_job(job_id)

    def get_run_lifecycle(self, run_id: str) -> dict[str, Any] | None:
        reconcile_run_lifecycles(store_root=self.adapter.store_root, run_id=run_id, active_only=True)
        record = get_run_lifecycle_record(store_root=self.adapter.store_root, run_id=run_id)
        if record is None:
            return None
        return {
            "run_id": record.run_id,
            "run_hash": record.run_hash,
            "config_hash": record.config_hash,
            "job_id": record.job_id,
            "logical_run_id": record.logical_run_id,
            "attempt_id": record.attempt_id,
            "attempt_no": record.attempt_no,
            "source": record.source,
            "operation_type": record.operation_type,
            "job_type": record.job_type,
            "status": record.status,
            "experiment_id": record.experiment_id,
            "config_id": record.config_id,
            "config_source": record.config_source,
            "config_path": record.config_path,
            "config_sha256": record.config_sha256,
            "run_dir": record.run_dir,
            "runtime_path": record.runtime_path,
            "backend": record.backend,
            "worker_id": record.worker_id,
            "pid": record.pid,
            "host": record.host,
            "current_stage": record.current_stage,
            "completed_stages": list(record.completed_stages),
            "progress_percent": record.progress_percent,
            "progress_label": record.progress_label,
            "progress_current": record.progress_current,
            "progress_total": record.progress_total,
            "cancel_requested": record.cancel_requested,
            "cancel_requested_at": record.cancel_requested_at,
            "created_at": record.created_at,
            "queued_at": record.queued_at,
            "started_at": record.started_at,
            "last_heartbeat_at": record.last_heartbeat_at,
            "updated_at": record.updated_at,
            "finished_at": record.finished_at,
            "terminal_reason": record.terminal_reason,
            "terminal_detail": record.terminal_detail,
            "latest_metrics": record.latest_metrics,
            "latest_sample": record.latest_sample,
            "reconciled": record.reconciled,
        }

    def list_run_job_events(self, job_id: str, *, after_id: int | None, limit: int) -> list[dict[str, Any]]:
        return self.adapter.list_run_job_events(job_id, after_id=after_id, limit=limit)

    def list_run_job_logs(
        self,
        job_id: str,
        *,
        after_id: int | None,
        limit: int,
        stream: str,
    ) -> list[dict[str, Any]]:
        return self.adapter.list_run_job_logs(
            job_id,
            after_id=after_id,
            limit=limit,
            stream=stream,
        )

    def list_run_job_samples(self, job_id: str, *, after_id: int | None, limit: int) -> list[dict[str, Any]]:
        return self.adapter.list_run_job_samples(job_id, after_id=after_id, limit=limit)

    def list_experiment_operations(
        self,
        *,
        experiment_id: str,
        operation_type: str | None,
        status: str | None,
        limit: int,
        offset: int,
    ) -> dict[str, Any]:
        items = self.adapter.list_operations(
            experiment_id=experiment_id,
            operation_type=operation_type,
            status=status,
            limit=limit,
            offset=offset,
        )
        total = self.adapter.count_operations(
            experiment_id=experiment_id,
            operation_type=operation_type,
            status=status,
        )
        return {"items": items, "total": total}

    def list_operation_attempts(self, logical_run_id: str, *, limit: int, offset: int) -> dict[str, Any] | None:
        operation = self.adapter.get_operation(logical_run_id)
        if operation is None:
            return None
        items = self.adapter.list_operation_attempts(logical_run_id, limit=limit, offset=offset)
        total = self.adapter.count_operation_attempts(logical_run_id)
        return {
            "operation": operation,
            "items": items,
            "total": total,
        }

    def get_runpod_pods(self) -> dict[str, Any]:
        return self.adapter.get_runpod_pods()

    def get_run_manifest(self, run_id: str) -> dict[str, Any] | None:
        return self.adapter.get_run_manifest(run_id)

    def get_run_metrics(self, run_id: str) -> dict[str, Any] | None:
        return self.adapter.get_run_metrics(run_id)

    def get_run_events(self, run_id: str, *, limit: int) -> list[dict[str, Any]]:
        return self.adapter.list_run_events(run_id, limit=limit)

    def get_run_resources(self, run_id: str, *, limit: int) -> list[dict[str, Any]]:
        return self.adapter.list_run_resources(run_id, limit=limit)

    def get_per_era_corr(self, run_id: str) -> list[dict[str, Any]] | None:
        return self.adapter.get_per_era_corr(run_id)

    def get_per_era_corr_result(self, run_id: str) -> PerEraCorrLoadResult:
        return self.adapter.get_per_era_corr_result(run_id)

    def get_scoring_dashboard(self, run_id: str) -> dict[str, Any] | None:
        return self.adapter.get_scoring_dashboard(run_id)

    def get_trials(self, run_id: str) -> list[dict[str, Any]] | None:
        return self.adapter.get_trials(run_id)

    def get_best_params(self, run_id: str) -> dict[str, Any] | None:
        return self.adapter.get_best_params(run_id)

    def get_resolved_config(self, run_id: str) -> dict[str, Any] | None:
        return self.adapter.get_resolved_config(run_id)

    def get_diagnostics_sources(self, run_id: str) -> dict[str, Any] | None:
        return self.adapter.get_diagnostics_sources(run_id)

    async def get_run_bundle(self, run_id: str) -> dict[str, Any]:
        """Fetch bundle parts concurrently to reduce endpoint latency."""

        metrics_task = asyncio.to_thread(self.adapter.get_run_metrics, run_id)
        manifest_task = asyncio.to_thread(self.adapter.get_run_manifest, run_id)
        events_task = asyncio.to_thread(self.adapter.list_run_events, run_id, limit=50)
        resources_task = asyncio.to_thread(self.adapter.list_run_resources, run_id, limit=50)
        scoring_dashboard_task = asyncio.to_thread(self.adapter.get_scoring_dashboard, run_id)
        trials_task = asyncio.to_thread(self.adapter.get_trials, run_id)
        params_task = asyncio.to_thread(self.adapter.get_best_params, run_id)
        config_task = asyncio.to_thread(self.adapter.get_resolved_config, run_id)
        diagnostics_sources_task = asyncio.to_thread(self.adapter.get_diagnostics_sources, run_id)

        (
            metrics,
            manifest,
            events,
            resources,
            scoring_dashboard,
            trials,
            best_params,
            resolved_config,
            diagnostics_sources,
        ) = await asyncio.gather(
            metrics_task,
            manifest_task,
            events_task,
            resources_task,
            scoring_dashboard_task,
            trials_task,
            params_task,
            config_task,
            diagnostics_sources_task,
        )

        return {
            "metrics": metrics,
            "manifest": manifest,
            "scoring_dashboard": scoring_dashboard,
            "trials": trials,
            "best_params": best_params,
            "resolved_config": resolved_config,
            "events": events,
            "resources": resources,
            "diagnostics_sources": diagnostics_sources,
        }

    def list_studies(
        self,
        *,
        experiment_id: str | None,
        status: str | None,
        limit: int,
        offset: int,
    ) -> dict[str, Any]:
        items = self.adapter.list_studies(
            experiment_id=experiment_id,
            status=status,
            limit=limit,
            offset=offset,
        )
        total = self.adapter.count_studies(experiment_id=experiment_id)
        return {"items": items, "total": total}

    def get_study(self, study_id: str) -> dict[str, Any] | None:
        return self.adapter.get_study(study_id)

    def get_study_trials(self, study_id: str) -> list[dict[str, Any]] | None:
        return self.adapter.get_study_trials(study_id)

    def get_experiment_studies(self, experiment_id: str) -> list[dict[str, Any]]:
        return self.adapter.get_experiment_studies(experiment_id)

    def list_ensembles(self, *, experiment_id: str | None, limit: int, offset: int) -> dict[str, Any]:
        items = self.adapter.list_ensembles(experiment_id=experiment_id, limit=limit, offset=offset)
        total = self.adapter.count_ensembles(experiment_id=experiment_id)
        return {"items": items, "total": total}

    def get_ensemble(self, ensemble_id: str) -> dict[str, Any] | None:
        return self.adapter.get_ensemble(ensemble_id)

    def get_ensemble_correlations(self, ensemble_id: str) -> dict[str, Any] | None:
        return self.adapter.get_ensemble_correlations(ensemble_id)

    def get_ensemble_artifacts(self, ensemble_id: str) -> dict[str, Any] | None:
        return self.adapter.get_ensemble_artifacts(ensemble_id)

    def get_experiment_ensembles(self, experiment_id: str) -> list[dict[str, Any]]:
        return self.adapter.get_experiment_ensembles(experiment_id)

    def get_doc_tree(self, domain: str) -> dict[str, Any]:
        return self.adapter.get_doc_tree(domain)

    def get_doc_content(self, domain: str, path: str) -> dict[str, Any]:
        return self.adapter.get_doc_content(domain, path)

    def get_doc_asset_path(self, domain: str, path: str) -> Path:
        return self.adapter.get_doc_asset_path(domain, path)

    def get_experiment_doc(self, experiment_id: str, filename: str) -> dict[str, Any]:
        return self.adapter.get_experiment_doc(experiment_id, filename)

    def get_run_doc(self, run_id: str, filename: str) -> dict[str, Any]:
        return self.adapter.get_run_doc(run_id, filename)

    def get_notes_tree(self) -> dict[str, Any]:
        return self.adapter.get_notes_tree()

    def get_notes_content(self, path: str) -> dict[str, Any]:
        return self.adapter.get_note_content(path)

    def _mtime_to_iso(self, value: Any) -> str | Any:
        if isinstance(value, (int, float)):
            from datetime import UTC, datetime

            return datetime.fromtimestamp(float(value), tz=UTC).isoformat()
        return value
