"""Service layer for viz API routes."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from numereng.features.viz.store_adapter import VizStoreAdapter

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

    def list_experiments(self) -> list[dict[str, Any]]:
        return self.adapter.list_experiments()

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
            "reconciled_stale_count": 0,
        }

    def get_batch_jobs(self, batch_id: str) -> dict[str, Any]:
        items = self.adapter.get_run_job_batch(batch_id)
        return {
            "batch_id": batch_id,
            "items": items,
            "total": len(items),
        }

    def get_run_job(self, job_id: str) -> dict[str, Any] | None:
        return self.adapter.get_run_job(job_id)

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

    def get_feature_importance(self, run_id: str, *, top_n: int) -> list[dict[str, Any]] | None:
        return self.adapter.get_feature_importance(run_id, top_n=top_n)

    def get_trials(self, run_id: str) -> list[dict[str, Any]] | None:
        return self.adapter.get_trials(run_id)

    def get_best_params(self, run_id: str) -> dict[str, Any] | None:
        return self.adapter.get_best_params(run_id)

    def get_resolved_config(self, run_id: str) -> dict[str, Any] | None:
        return self.adapter.get_resolved_config(run_id)

    async def get_run_bundle(self, run_id: str) -> dict[str, Any]:
        """Fetch bundle parts concurrently to reduce endpoint latency."""

        metrics_task = asyncio.to_thread(self.adapter.get_run_metrics, run_id)
        manifest_task = asyncio.to_thread(self.adapter.get_run_manifest, run_id)
        events_task = asyncio.to_thread(self.adapter.list_run_events, run_id, limit=50)
        resources_task = asyncio.to_thread(self.adapter.list_run_resources, run_id, limit=50)
        corr_task = asyncio.to_thread(self.adapter.get_per_era_corr, run_id)
        fi_task = asyncio.to_thread(self.adapter.get_feature_importance, run_id, top_n=30)
        trials_task = asyncio.to_thread(self.adapter.get_trials, run_id)
        params_task = asyncio.to_thread(self.adapter.get_best_params, run_id)
        config_task = asyncio.to_thread(self.adapter.get_resolved_config, run_id)
        diagnostics_sources_task = asyncio.to_thread(self.adapter.get_diagnostics_sources, run_id)

        (
            metrics,
            manifest,
            events,
            resources,
            per_era_corr,
            feature_importance,
            trials,
            best_params,
            resolved_config,
            diagnostics_sources,
        ) = await asyncio.gather(
            metrics_task,
            manifest_task,
            events_task,
            resources_task,
            corr_task,
            fi_task,
            trials_task,
            params_task,
            config_task,
            diagnostics_sources_task,
        )

        return {
            "metrics": metrics,
            "manifest": manifest,
            "per_era_corr": per_era_corr,
            "feature_importance": feature_importance,
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
