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
from numereng_viz.remote_detail import RemoteDetailCoordinator
from numereng_viz.store_adapter import PerEraCorrLoadResult, VizStoreAdapter

TOP_METRIC_NAMES = [
    "bmc_last_200_eras_mean",
    "bmc_mean",
    "corr_sharpe",
    "corr_mean",
    "mmc_mean",
]

RUN_BUNDLE_SECTIONS = frozenset(
    {
        "manifest",
        "metrics",
        "scoring_dashboard",
        "events",
        "resources",
        "resolved_config",
        "trials",
        "best_params",
        "diagnostics_sources",
    }
)


class VizService:
    """Thin orchestration layer over the store adapter."""

    def __init__(self, adapter: VizStoreAdapter) -> None:
        self.adapter = adapter
        self.remote_snapshots = RemoteSnapshotCoordinator(store_root=self.adapter.store_root)
        self.remote_details = RemoteDetailCoordinator(store_root=self.adapter.store_root)

    def list_experiments(self) -> list[dict[str, Any]]:
        return self.adapter.list_experiments()

    def list_submissions(self) -> dict[str, Any]:
        return self.adapter.list_submissions()

    def get_submission(self, model_name: str) -> dict[str, Any] | None:
        return self.adapter.get_submission(model_name)

    def get_experiments_overview(self, *, include_remote: bool = True) -> dict[str, Any]:
        local_snapshot = build_monitor_snapshot(store_root=self.adapter.store_root, refresh_cloud=True)
        if not include_remote:
            return local_snapshot
        remote_snapshots = self.remote_snapshots.fetch_snapshots()
        return merge_monitor_snapshots(local_snapshot, remote_snapshots)

    def get_experiment(
        self,
        experiment_id: str,
        *,
        source_kind: str | None = None,
        source_id: str | None = None,
    ) -> dict[str, Any] | None:
        if self._is_remote_source(source_kind=source_kind, source_id=source_id):
            return self._remote_call(
                "get_experiment",
                args=(experiment_id,),
                source_kind=source_kind,
                source_id=source_id,
            )
        payload = self.adapter.get_experiment(experiment_id)
        if isinstance(payload, dict):
            return payload
        remote_source = self._unique_remote_source_for_experiment(experiment_id)
        if remote_source is None:
            return None
        remote_payload = self._remote_call(
            "get_experiment",
            args=(experiment_id,),
            source_kind=remote_source["kind"],
            source_id=remote_source["id"],
        )
        return remote_payload if isinstance(remote_payload, dict) else None

    def list_experiment_configs(
        self,
        experiment_id: str,
        *,
        q: str | None,
        include_incompatible: bool,
        limit: int,
        offset: int,
        source_kind: str | None = None,
        source_id: str | None = None,
    ) -> dict[str, Any]:
        if self._is_remote_source(source_kind=source_kind, source_id=source_id):
            payload = self._remote_call(
                "list_experiment_configs",
                args=(experiment_id,),
                kwargs={
                    "q": q,
                    "include_incompatible": include_incompatible,
                    "limit": limit,
                    "offset": offset,
                },
                source_kind=source_kind,
                source_id=source_id,
            )
            assert isinstance(payload, dict)
            return payload
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

    def list_experiment_runs(
        self,
        experiment_id: str,
        *,
        source_kind: str | None = None,
        source_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if self._is_remote_source(source_kind=source_kind, source_id=source_id):
            payload = self._remote_call(
                "list_experiment_runs",
                args=(experiment_id,),
                source_kind=source_kind,
                source_id=source_id,
            )
            assert isinstance(payload, list)
            return payload
        local_payload = self.adapter.list_experiment_runs(experiment_id)
        if local_payload:
            return local_payload
        if self.adapter.get_experiment(experiment_id) is not None:
            return local_payload
        remote_source = self._unique_remote_source_for_experiment(experiment_id)
        if remote_source is None:
            return local_payload
        remote_payload = self._remote_call(
            "list_experiment_runs",
            args=(experiment_id,),
            source_kind=remote_source["kind"],
            source_id=remote_source["id"],
        )
        if not isinstance(remote_payload, list):
            return local_payload
        return self._merge_local_and_remote_runs(local_payload, remote_payload, remote_source=remote_source)

    def list_experiment_round_results(
        self,
        experiment_id: str,
        *,
        source_kind: str | None = None,
        source_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if self._is_remote_source(source_kind=source_kind, source_id=source_id):
            payload = self._remote_call(
                "list_experiment_round_results",
                args=(experiment_id,),
                source_kind=source_kind,
                source_id=source_id,
            )
            assert isinstance(payload, list)
            return payload
        return self.adapter.list_experiment_round_results(experiment_id)

    def list_run_jobs(
        self,
        *,
        experiment_id: str | None,
        status: str | None,
        limit: int,
        offset: int,
        include_attempts: bool,
        source_kind: str | None = None,
        source_id: str | None = None,
    ) -> dict[str, Any]:
        if self._is_remote_source(source_kind=source_kind, source_id=source_id):
            payload = self._remote_call(
                "list_run_jobs",
                kwargs={
                    "experiment_id": experiment_id,
                    "status": status,
                    "limit": limit,
                    "offset": offset,
                    "include_attempts": include_attempts,
                },
                source_kind=source_kind,
                source_id=source_id,
            )
            assert isinstance(payload, dict)
            return payload
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

    def get_run_job(
        self,
        job_id: str,
        *,
        source_kind: str | None = None,
        source_id: str | None = None,
    ) -> dict[str, Any] | None:
        if self._is_remote_source(source_kind=source_kind, source_id=source_id):
            return self._remote_call(
                "get_run_job",
                args=(job_id,),
                source_kind=source_kind,
                source_id=source_id,
            )
        reconcile_run_lifecycles(store_root=self.adapter.store_root, active_only=True)
        return self.adapter.get_run_job(job_id)

    def get_run_lifecycle(
        self,
        run_id: str,
        *,
        source_kind: str | None = None,
        source_id: str | None = None,
    ) -> dict[str, Any] | None:
        if self._is_remote_source(source_kind=source_kind, source_id=source_id):
            return self._remote_call(
                "get_run_lifecycle",
                args=(run_id,),
                source_kind=source_kind,
                source_id=source_id,
            )
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

    def list_run_job_events(
        self,
        job_id: str,
        *,
        after_id: int | None,
        limit: int,
        source_kind: str | None = None,
        source_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if self._is_remote_source(source_kind=source_kind, source_id=source_id):
            payload = self._remote_call(
                "list_run_job_events",
                args=(job_id,),
                kwargs={"after_id": after_id, "limit": limit},
                source_kind=source_kind,
                source_id=source_id,
            )
            assert isinstance(payload, list)
            return payload
        return self.adapter.list_run_job_events(job_id, after_id=after_id, limit=limit)

    def list_run_job_logs(
        self,
        job_id: str,
        *,
        after_id: int | None,
        limit: int,
        stream: str,
        source_kind: str | None = None,
        source_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if self._is_remote_source(source_kind=source_kind, source_id=source_id):
            payload = self._remote_call(
                "list_run_job_logs",
                args=(job_id,),
                kwargs={"after_id": after_id, "limit": limit, "stream": stream},
                source_kind=source_kind,
                source_id=source_id,
            )
            assert isinstance(payload, list)
            return payload
        return self.adapter.list_run_job_logs(
            job_id,
            after_id=after_id,
            limit=limit,
            stream=stream,
        )

    def list_run_job_samples(
        self,
        job_id: str,
        *,
        after_id: int | None,
        limit: int,
        source_kind: str | None = None,
        source_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if self._is_remote_source(source_kind=source_kind, source_id=source_id):
            payload = self._remote_call(
                "list_run_job_samples",
                args=(job_id,),
                kwargs={"after_id": after_id, "limit": limit},
                source_kind=source_kind,
                source_id=source_id,
            )
            assert isinstance(payload, list)
            return payload
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

    def get_run_manifest(
        self,
        run_id: str,
        *,
        source_kind: str | None = None,
        source_id: str | None = None,
    ) -> dict[str, Any] | None:
        if self._is_remote_source(source_kind=source_kind, source_id=source_id):
            return self._remote_call(
                "get_run_manifest",
                args=(run_id,),
                source_kind=source_kind,
                source_id=source_id,
            )
        payload = self.adapter.get_run_manifest(run_id)
        if payload is not None:
            return payload
        return self._remote_run_fallback("get_run_manifest", run_id)

    def get_run_metrics(
        self,
        run_id: str,
        *,
        source_kind: str | None = None,
        source_id: str | None = None,
    ) -> dict[str, Any] | None:
        if self._is_remote_source(source_kind=source_kind, source_id=source_id):
            return self._remote_call(
                "get_run_metrics",
                args=(run_id,),
                source_kind=source_kind,
                source_id=source_id,
            )
        payload = self.adapter.get_run_metrics(run_id)
        if payload is not None:
            return payload
        return self._remote_run_fallback("get_run_metrics", run_id)

    def get_run_events(
        self,
        run_id: str,
        *,
        limit: int,
        source_kind: str | None = None,
        source_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if self._is_remote_source(source_kind=source_kind, source_id=source_id):
            payload = self._remote_call(
                "get_run_events",
                args=(run_id,),
                kwargs={"limit": limit},
                source_kind=source_kind,
                source_id=source_id,
            )
            assert isinstance(payload, list)
            return payload
        return self.adapter.list_run_events(run_id, limit=limit)

    def get_run_resources(
        self,
        run_id: str,
        *,
        limit: int,
        source_kind: str | None = None,
        source_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if self._is_remote_source(source_kind=source_kind, source_id=source_id):
            payload = self._remote_call(
                "get_run_resources",
                args=(run_id,),
                kwargs={"limit": limit},
                source_kind=source_kind,
                source_id=source_id,
            )
            assert isinstance(payload, list)
            return payload
        return self.adapter.list_run_resources(run_id, limit=limit)

    def get_per_era_corr(
        self,
        run_id: str,
        *,
        source_kind: str | None = None,
        source_id: str | None = None,
    ) -> list[dict[str, Any]] | None:
        if self._is_remote_source(source_kind=source_kind, source_id=source_id):
            return self._remote_call(
                "get_per_era_corr",
                args=(run_id,),
                source_kind=source_kind,
                source_id=source_id,
            )
        payload = self.adapter.get_per_era_corr(run_id)
        if payload is not None:
            return payload
        return self._remote_run_fallback("get_per_era_corr", run_id)

    def get_per_era_corr_result(
        self,
        run_id: str,
        *,
        source_kind: str | None = None,
        source_id: str | None = None,
    ) -> PerEraCorrLoadResult:
        if self._is_remote_source(source_kind=source_kind, source_id=source_id):
            payload = self.get_per_era_corr(run_id, source_kind=source_kind, source_id=source_id)
            return PerEraCorrLoadResult(payload=payload, persisted_read_ms=0.0, materialize_ms=0.0)
        return self.adapter.get_per_era_corr_result(run_id)

    def get_scoring_dashboard(
        self,
        run_id: str,
        *,
        source_kind: str | None = None,
        source_id: str | None = None,
    ) -> dict[str, Any] | None:
        if self._is_remote_source(source_kind=source_kind, source_id=source_id):
            return self._remote_call(
                "get_scoring_dashboard",
                args=(run_id,),
                source_kind=source_kind,
                source_id=source_id,
            )
        payload = self.adapter.get_scoring_dashboard(run_id)
        if payload is not None:
            return payload
        return self._remote_run_fallback("get_scoring_dashboard", run_id)

    def get_trials(
        self,
        run_id: str,
        *,
        source_kind: str | None = None,
        source_id: str | None = None,
    ) -> list[dict[str, Any]] | None:
        if self._is_remote_source(source_kind=source_kind, source_id=source_id):
            return self._remote_call(
                "get_trials",
                args=(run_id,),
                source_kind=source_kind,
                source_id=source_id,
            )
        payload = self.adapter.get_trials(run_id)
        if payload is not None:
            return payload
        return self._remote_run_fallback("get_trials", run_id)

    def get_best_params(
        self,
        run_id: str,
        *,
        source_kind: str | None = None,
        source_id: str | None = None,
    ) -> dict[str, Any] | None:
        if self._is_remote_source(source_kind=source_kind, source_id=source_id):
            return self._remote_call(
                "get_best_params",
                args=(run_id,),
                source_kind=source_kind,
                source_id=source_id,
            )
        payload = self.adapter.get_best_params(run_id)
        if payload is not None:
            return payload
        return self._remote_run_fallback("get_best_params", run_id)

    def get_resolved_config(
        self,
        run_id: str,
        *,
        source_kind: str | None = None,
        source_id: str | None = None,
    ) -> dict[str, Any] | None:
        if self._is_remote_source(source_kind=source_kind, source_id=source_id):
            return self._remote_call(
                "get_resolved_config",
                args=(run_id,),
                source_kind=source_kind,
                source_id=source_id,
            )
        payload = self.adapter.get_resolved_config(run_id)
        if payload is not None:
            return payload
        return self._remote_run_fallback("get_resolved_config", run_id)

    def get_diagnostics_sources(
        self,
        run_id: str,
        *,
        source_kind: str | None = None,
        source_id: str | None = None,
    ) -> dict[str, Any] | None:
        if self._is_remote_source(source_kind=source_kind, source_id=source_id):
            return self._remote_call(
                "get_diagnostics_sources",
                args=(run_id,),
                source_kind=source_kind,
                source_id=source_id,
            )
        payload = self.adapter.get_diagnostics_sources(run_id)
        if payload is not None:
            return payload
        return self._remote_run_fallback("get_diagnostics_sources", run_id)

    async def get_run_bundle(
        self,
        run_id: str,
        *,
        sections: set[str] | None = None,
        source_kind: str | None = None,
        source_id: str | None = None,
    ) -> dict[str, Any]:
        """Fetch bundle parts concurrently to reduce endpoint latency."""

        requested_sections = RUN_BUNDLE_SECTIONS if sections is None else set(sections)

        if self._is_remote_source(source_kind=source_kind, source_id=source_id):
            payload = self._remote_call(
                "get_run_bundle",
                args=(run_id,),
                kwargs={"sections": sorted(requested_sections)} if sections is not None else None,
                source_kind=source_kind,
                source_id=source_id,
            )
            assert isinstance(payload, dict)
            return payload

        manifest = self.adapter.get_run_manifest(run_id) if "manifest" in requested_sections else None
        metrics = self.adapter.get_run_metrics(run_id) if "metrics" in requested_sections else None
        scoring_dashboard = (
            self.adapter.get_scoring_dashboard(run_id) if "scoring_dashboard" in requested_sections else None
        )
        if (
            ("manifest" not in requested_sections or manifest is None)
            and ("metrics" not in requested_sections or metrics is None)
            and ("scoring_dashboard" not in requested_sections or scoring_dashboard is None)
            and not requested_sections.intersection(
                {"events", "resources", "resolved_config", "trials", "best_params", "diagnostics_sources"}
            )
        ):
            remote_payload = self._remote_run_fallback("get_run_bundle", run_id)
            if isinstance(remote_payload, dict):
                return {key: remote_payload.get(key) for key in requested_sections}

        tasks: dict[str, asyncio.Future[Any]] = {}
        if "metrics" in requested_sections:
            tasks["metrics"] = asyncio.to_thread(
                lambda: metrics if metrics is not None else self.adapter.get_run_metrics(run_id)
            )
        if "manifest" in requested_sections:
            tasks["manifest"] = asyncio.to_thread(
                lambda: manifest if manifest is not None else self.adapter.get_run_manifest(run_id)
            )
        if "events" in requested_sections:
            tasks["events"] = asyncio.to_thread(self.adapter.list_run_events, run_id, limit=50)
        if "resources" in requested_sections:
            tasks["resources"] = asyncio.to_thread(self.adapter.list_run_resources, run_id, limit=50)
        if "scoring_dashboard" in requested_sections:
            tasks["scoring_dashboard"] = asyncio.to_thread(
                lambda: (
                    scoring_dashboard if scoring_dashboard is not None else self.adapter.get_scoring_dashboard(run_id)
                )
            )
        if "trials" in requested_sections:
            tasks["trials"] = asyncio.to_thread(self.adapter.get_trials, run_id)
        if "best_params" in requested_sections:
            tasks["best_params"] = asyncio.to_thread(self.adapter.get_best_params, run_id)
        if "resolved_config" in requested_sections:
            tasks["resolved_config"] = asyncio.to_thread(self.adapter.get_resolved_config, run_id)
        if "diagnostics_sources" in requested_sections:
            tasks["diagnostics_sources"] = asyncio.to_thread(self.adapter.get_diagnostics_sources, run_id)

        results = await asyncio.gather(*tasks.values())
        return dict(zip(tasks.keys(), results, strict=True))

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

    def get_study(
        self,
        study_id: str,
        *,
        source_kind: str | None = None,
        source_id: str | None = None,
    ) -> dict[str, Any] | None:
        if self._is_remote_source(source_kind=source_kind, source_id=source_id):
            return self._remote_call(
                "get_study",
                args=(study_id,),
                source_kind=source_kind,
                source_id=source_id,
            )
        return self.adapter.get_study(study_id)

    def get_study_trials(
        self,
        study_id: str,
        *,
        source_kind: str | None = None,
        source_id: str | None = None,
    ) -> list[dict[str, Any]] | None:
        if self._is_remote_source(source_kind=source_kind, source_id=source_id):
            return self._remote_call(
                "get_study_trials",
                args=(study_id,),
                source_kind=source_kind,
                source_id=source_id,
            )
        return self.adapter.get_study_trials(study_id)

    def get_experiment_studies(
        self,
        experiment_id: str,
        *,
        source_kind: str | None = None,
        source_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if self._is_remote_source(source_kind=source_kind, source_id=source_id):
            payload = self._remote_call(
                "get_experiment_studies",
                args=(experiment_id,),
                source_kind=source_kind,
                source_id=source_id,
            )
            assert isinstance(payload, list)
            return payload
        return self.adapter.get_experiment_studies(experiment_id)

    def list_ensembles(self, *, experiment_id: str | None, limit: int, offset: int) -> dict[str, Any]:
        items = self.adapter.list_ensembles(experiment_id=experiment_id, limit=limit, offset=offset)
        total = self.adapter.count_ensembles(experiment_id=experiment_id)
        return {"items": items, "total": total}

    def get_ensemble(
        self,
        ensemble_id: str,
        *,
        source_kind: str | None = None,
        source_id: str | None = None,
    ) -> dict[str, Any] | None:
        if self._is_remote_source(source_kind=source_kind, source_id=source_id):
            return self._remote_call(
                "get_ensemble",
                args=(ensemble_id,),
                source_kind=source_kind,
                source_id=source_id,
            )
        return self.adapter.get_ensemble(ensemble_id)

    def get_ensemble_correlations(
        self,
        ensemble_id: str,
        *,
        source_kind: str | None = None,
        source_id: str | None = None,
    ) -> dict[str, Any] | None:
        if self._is_remote_source(source_kind=source_kind, source_id=source_id):
            return self._remote_call(
                "get_ensemble_correlations",
                args=(ensemble_id,),
                source_kind=source_kind,
                source_id=source_id,
            )
        return self.adapter.get_ensemble_correlations(ensemble_id)

    def get_ensemble_artifacts(
        self,
        ensemble_id: str,
        *,
        source_kind: str | None = None,
        source_id: str | None = None,
    ) -> dict[str, Any] | None:
        if self._is_remote_source(source_kind=source_kind, source_id=source_id):
            return self._remote_call(
                "get_ensemble_artifacts",
                args=(ensemble_id,),
                source_kind=source_kind,
                source_id=source_id,
            )
        return self.adapter.get_ensemble_artifacts(ensemble_id)

    def get_experiment_ensembles(
        self,
        experiment_id: str,
        *,
        source_kind: str | None = None,
        source_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if self._is_remote_source(source_kind=source_kind, source_id=source_id):
            payload = self._remote_call(
                "get_experiment_ensembles",
                args=(experiment_id,),
                source_kind=source_kind,
                source_id=source_id,
            )
            assert isinstance(payload, list)
            return payload
        return self.adapter.get_experiment_ensembles(experiment_id)

    def get_doc_tree(self, domain: str) -> dict[str, Any]:
        return self.adapter.get_doc_tree(domain)

    def get_doc_content(self, domain: str, path: str) -> dict[str, Any]:
        return self.adapter.get_doc_content(domain, path)

    def get_doc_asset_path(self, domain: str, path: str) -> Path:
        return self.adapter.get_doc_asset_path(domain, path)

    def get_experiment_doc(
        self,
        experiment_id: str,
        filename: str,
        *,
        source_kind: str | None = None,
        source_id: str | None = None,
    ) -> dict[str, Any]:
        if self._is_remote_source(source_kind=source_kind, source_id=source_id):
            payload = self._remote_call(
                "get_experiment_doc",
                args=(experiment_id, filename),
                source_kind=source_kind,
                source_id=source_id,
            )
            assert isinstance(payload, dict)
            return payload
        return self.adapter.get_experiment_doc(experiment_id, filename)

    def get_run_doc(
        self,
        run_id: str,
        filename: str,
        *,
        source_kind: str | None = None,
        source_id: str | None = None,
    ) -> dict[str, Any]:
        if self._is_remote_source(source_kind=source_kind, source_id=source_id):
            payload = self._remote_call(
                "get_run_doc",
                args=(run_id, filename),
                source_kind=source_kind,
                source_id=source_id,
            )
            assert isinstance(payload, dict)
            return payload
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

    def _remote_call(
        self,
        method: str,
        *,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        source_kind: str | None,
        source_id: str | None,
    ) -> Any:
        return self.remote_details.call(
            method,
            args=args,
            kwargs=kwargs,
            source_kind=str(source_kind or ""),
            source_id=str(source_id or ""),
        )

    def _is_remote_source(self, *, source_kind: str | None, source_id: str | None) -> bool:
        normalized_kind = (source_kind or "").strip()
        normalized_id = (source_id or "").strip()
        if not normalized_kind and not normalized_id:
            return False
        if normalized_kind in {"", "local"} and normalized_id in {"", "local"}:
            return False
        return True

    def _remote_run_fallback(self, method: str, run_id: str) -> Any:
        remote_source = self._unique_remote_source_for_run(run_id)
        if remote_source is None:
            return None
        return self._remote_call(
            method,
            args=(run_id,),
            source_kind=remote_source["kind"],
            source_id=remote_source["id"],
        )

    def _merge_local_and_remote_runs(
        self,
        local_payload: list[dict[str, Any]],
        remote_payload: list[dict[str, Any]],
        *,
        remote_source: dict[str, str],
    ) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = [dict(item) for item in local_payload]
        seen_run_ids = {str(item.get("run_id")) for item in merged if isinstance(item.get("run_id"), str)}
        for item in remote_payload:
            if not isinstance(item, dict):
                continue
            run_id = item.get("run_id")
            if not isinstance(run_id, str) or run_id in seen_run_ids:
                continue
            remote_item = dict(item)
            remote_item["source_kind"] = remote_source["kind"]
            remote_item["source_id"] = remote_source["id"]
            remote_item["source_label"] = remote_source["label"]
            merged.append(remote_item)
            seen_run_ids.add(run_id)
        merged.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return merged

    def _unique_remote_source_for_experiment(self, experiment_id: str) -> dict[str, str] | None:
        matches: dict[tuple[str, str], dict[str, str]] = {}
        for snapshot in self.remote_snapshots.fetch_snapshots():
            source = snapshot.get("source")
            if not isinstance(source, dict):
                continue
            source_kind = str(source.get("kind") or "")
            source_id = str(source.get("id") or "")
            source_label = str(source.get("label") or source_id or source_kind)
            if not source_kind or not source_id:
                continue
            if any(
                isinstance(item, dict) and item.get("experiment_id") == experiment_id
                for section_name in ("experiments", "live_experiments")
                for item in snapshot.get(section_name, [])
            ):
                matches[(source_kind, source_id)] = {
                    "kind": source_kind,
                    "id": source_id,
                    "label": source_label,
                }
        if len(matches) != 1:
            return None
        return next(iter(matches.values()))

    def _unique_remote_source_for_run(self, run_id: str) -> dict[str, str] | None:
        matches: dict[tuple[str, str], dict[str, str]] = {}
        for snapshot in self.remote_snapshots.fetch_snapshots():
            source = snapshot.get("source")
            if not isinstance(source, dict):
                continue
            source_kind = str(source.get("kind") or "")
            source_id = str(source.get("id") or "")
            source_label = str(source.get("label") or source_id or source_kind)
            if not source_kind or not source_id:
                continue
            has_run = any(
                isinstance(experiment, dict)
                and any(isinstance(item, dict) and item.get("run_id") == run_id for item in experiment.get("runs", []))
                for experiment in snapshot.get("live_experiments", [])
            ) or any(
                isinstance(item, dict) and item.get("run_id") == run_id for item in snapshot.get("recent_activity", [])
            )
            if has_run:
                matches[(source_kind, source_id)] = {
                    "kind": source_kind,
                    "id": source_id,
                    "label": source_label,
                }
        if len(matches) != 1:
            return None
        return next(iter(matches.values()))
