"""FastAPI routes for the viz API."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from numereng.features.viz.contracts import capabilities_payload
from numereng.features.viz.services import VizService

_VALID_EXPERIMENT_DOCS = {"EXPERIMENT.md", "REPORT.md"}
_VALID_RUN_DOCS = {"RUN.md"}


def _server_timing_header(metrics: dict[str, float]) -> str:
    parts = [f"{name};dur={value:.1f}" for name, value in metrics.items()]
    return ", ".join(parts)


def _cached_response(data: Any, *, max_age: int = 3600, extra_headers: dict[str, str] | None = None) -> JSONResponse:
    headers = {"Cache-Control": f"public, max-age={max_age}"}
    if extra_headers:
        headers.update(extra_headers)
    return JSONResponse(
        content=data,
        headers=headers,
    )


def _nocache_response(
    data: Any,
    *,
    status_code: int = 200,
    extra_headers: dict[str, str] | None = None,
) -> JSONResponse:
    headers = {"Cache-Control": "no-store"}
    if extra_headers:
        headers.update(extra_headers)
    return JSONResponse(
        content=data,
        status_code=status_code,
        headers=headers,
    )


def _bounded_limit(value: int, *, default: int, max_value: int) -> int:
    if value <= 0:
        return default
    return min(value, max_value)


def _validate_doc_filename(filename: str, *, valid: set[str]) -> str:
    if filename not in valid:
        raise HTTPException(400, f"Invalid doc filename: {filename}")
    return filename


def _validate_note_path(path: str) -> str:
    raw = path.strip().replace("\\", "/")
    if not raw or raw.startswith("/") or "\x00" in raw:
        raise HTTPException(400, f"Invalid note path: {path}")

    segments: list[str] = []
    for segment in raw.split("/"):
        if segment in {"", "."}:
            continue
        if segment == "..":
            raise HTTPException(400, f"Invalid note path: {path}")
        segments.append(segment)

    if not segments:
        raise HTTPException(400, f"Invalid note path: {path}")

    normalized = "/".join(segments)
    if not normalized.lower().endswith(".md"):
        raise HTTPException(400, f"Invalid note path: {path}")
    return normalized


def _sse_message(event: str, data: dict[str, Any], *, event_id: int | None = None) -> str:
    payload = json.dumps(data, default=str, separators=(",", ":"))
    parts = []
    if event_id is not None:
        parts.append(f"id: {event_id}")
    parts.append(f"event: {event}")
    parts.append(f"data: {payload}")
    return "\n".join(parts) + "\n\n"


def create_router(service: VizService) -> APIRouter:
    """Build `/api/*` router for a configured service instance."""

    router = APIRouter(prefix="/api")

    @router.get("/system/capabilities")
    def get_system_capabilities() -> JSONResponse:
        return _nocache_response(capabilities_payload())

    @router.get("/experiments")
    def list_experiments() -> JSONResponse:
        return _cached_response(service.list_experiments(), max_age=30)

    @router.get("/experiments/{experiment_id}")
    def get_experiment(experiment_id: str) -> JSONResponse:
        item = service.get_experiment(experiment_id)
        if item is None:
            raise HTTPException(404, f"Experiment {experiment_id} not found")
        return _cached_response(item, max_age=30)

    @router.get("/experiments/{experiment_id}/configs")
    def get_experiment_configs(
        experiment_id: str,
        q: str | None = None,
        include_incompatible: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> JSONResponse:
        try:
            payload = service.list_experiment_configs(
                experiment_id,
                q=q,
                include_incompatible=include_incompatible,
                limit=_bounded_limit(limit, default=100, max_value=1000),
                offset=max(0, offset),
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return _nocache_response(payload)

    @router.get("/configs")
    def list_configs(
        q: str | None = None,
        experiment_id: str | None = None,
        model_type: str | None = None,
        target: str | None = None,
        limit: int = 500,
        offset: int = 0,
    ) -> JSONResponse:
        try:
            payload = service.list_configs(
                q=q,
                experiment_id=experiment_id,
                model_type=model_type,
                target=target,
                limit=_bounded_limit(limit, default=500, max_value=5000),
                offset=max(0, offset),
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return _nocache_response(payload)

    @router.get("/configs/compare")
    def compare_configs(config_ids: str) -> JSONResponse:
        ids = [value.strip() for value in config_ids.split(",") if value.strip()]
        if len(ids) < 2 or len(ids) > 5:
            raise HTTPException(400, "Provide 2-5 comma-separated config_ids")
        try:
            payload = service.compare_configs(ids)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(404, str(exc)) from exc
        return _nocache_response(payload)

    @router.get("/experiments/{experiment_id}/runs")
    def get_experiment_runs(experiment_id: str) -> JSONResponse:
        try:
            payload = service.list_experiment_runs(experiment_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return _cached_response(payload, max_age=30)

    @router.get("/experiments/{experiment_id}/round-results")
    def get_experiment_round_results(experiment_id: str) -> JSONResponse:
        try:
            payload = service.list_experiment_round_results(experiment_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return _nocache_response(payload)

    @router.get("/run-jobs")
    def list_run_jobs(
        experiment_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
        include_attempts: bool = False,
    ) -> JSONResponse:
        payload = service.list_run_jobs(
            experiment_id=experiment_id,
            status=status,
            limit=_bounded_limit(limit, default=50, max_value=1000),
            offset=max(0, offset),
            include_attempts=include_attempts,
        )
        return _nocache_response(payload)

    @router.get("/experiments/{experiment_id}/operations")
    def list_experiment_operations(
        experiment_id: str,
        operation_type: str | None = None,
        status: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> JSONResponse:
        try:
            payload = service.list_experiment_operations(
                experiment_id=experiment_id,
                operation_type=operation_type,
                status=status,
                limit=_bounded_limit(limit, default=200, max_value=2000),
                offset=max(0, offset),
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return _nocache_response(payload)

    @router.get("/operations/{logical_run_id}/attempts")
    def list_operation_attempts(logical_run_id: str, limit: int = 100, offset: int = 0) -> JSONResponse:
        try:
            payload = service.list_operation_attempts(
                logical_run_id,
                limit=_bounded_limit(limit, default=100, max_value=2000),
                offset=max(0, offset),
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if payload is None:
            raise HTTPException(404, f"Operation {logical_run_id} not found")
        return _nocache_response(payload)

    @router.get("/runpod/pods")
    def list_runpod_pods() -> JSONResponse:
        return _nocache_response(service.get_runpod_pods())

    @router.get("/run-jobs/batches/{batch_id}")
    def get_batch_jobs(batch_id: str) -> JSONResponse:
        try:
            payload = service.get_batch_jobs(batch_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if payload["total"] == 0:
            raise HTTPException(404, f"Batch {batch_id} not found")
        return _nocache_response(payload)

    @router.get("/run-jobs/{job_id}")
    def get_run_job(job_id: str) -> JSONResponse:
        try:
            payload = service.get_run_job(job_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if payload is None:
            raise HTTPException(404, f"Job {job_id} not found")
        return _nocache_response(payload)

    @router.get("/run-jobs/{job_id}/events")
    def get_run_job_events(job_id: str, after_id: int | None = None, limit: int = 200) -> JSONResponse:
        try:
            job = service.get_run_job(job_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if job is None:
            raise HTTPException(404, f"Job {job_id} not found")
        payload = service.list_run_job_events(
            job_id,
            after_id=after_id,
            limit=_bounded_limit(limit, default=200, max_value=5000),
        )
        return _nocache_response(payload)

    @router.get("/run-jobs/{job_id}/logs")
    def get_run_job_logs(
        job_id: str,
        after_id: int | None = None,
        limit: int = 200,
        stream: str = "all",
    ) -> JSONResponse:
        try:
            job = service.get_run_job(job_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if job is None:
            raise HTTPException(404, f"Job {job_id} not found")
        payload = service.list_run_job_logs(
            job_id,
            after_id=after_id,
            limit=_bounded_limit(limit, default=200, max_value=5000),
            stream=stream,
        )
        return _nocache_response(payload)

    @router.get("/run-jobs/{job_id}/samples")
    def get_run_job_samples(job_id: str, after_id: int | None = None, limit: int = 200) -> JSONResponse:
        try:
            job = service.get_run_job(job_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if job is None:
            raise HTTPException(404, f"Job {job_id} not found")
        payload = service.list_run_job_samples(
            job_id,
            after_id=after_id,
            limit=_bounded_limit(limit, default=200, max_value=5000),
        )
        return _nocache_response(payload)

    @router.get("/run-jobs/{job_id}/stream")
    async def stream_run_job(
        job_id: str,
        request: Request,
        after_event_id: int = 0,
        after_log_id: int = 0,
        after_sample_id: int = 0,
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    ) -> StreamingResponse:
        try:
            job = service.get_run_job(job_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if job is None:
            raise HTTPException(404, f"Job {job_id} not found")

        if last_event_id and last_event_id.isdigit() and after_event_id == 0:
            after_event_id = int(last_event_id)

        async def event_stream() -> Any:
            event_cursor = after_event_id
            log_cursor = after_log_id
            sample_cursor = after_sample_id
            heartbeat_counter = 0
            while True:
                if await request.is_disconnected():
                    break

                emitted = False

                events = await asyncio.to_thread(
                    service.list_run_job_events,
                    job_id,
                    after_id=event_cursor,
                    limit=500,
                )
                for item in events:
                    event_cursor = int(item["id"])
                    emitted = True
                    yield _sse_message("job_event", item, event_id=event_cursor)

                logs = await asyncio.to_thread(
                    service.list_run_job_logs,
                    job_id,
                    after_id=log_cursor,
                    limit=500,
                    stream="all",
                )
                for item in logs:
                    log_cursor = int(item["id"])
                    emitted = True
                    yield _sse_message("log_line", item)

                samples = await asyncio.to_thread(
                    service.list_run_job_samples,
                    job_id,
                    after_id=sample_cursor,
                    limit=500,
                )
                for item in samples:
                    sample_cursor = int(item["id"])
                    emitted = True
                    yield _sse_message("resource_sample", item)

                if not emitted:
                    heartbeat_counter += 1
                    if heartbeat_counter >= 15:
                        heartbeat_counter = 0
                        yield _sse_message("heartbeat", {"job_id": job_id})

                try:
                    job = await asyncio.to_thread(service.get_run_job, job_id)
                except ValueError:
                    break
                if job is None:
                    break
                if job.get("status") in {"completed", "failed", "canceled", "stale"} and not emitted:
                    break

                await asyncio.sleep(1.0)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-store",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @router.get("/runs/{run_id}/manifest")
    def get_run_manifest(run_id: str) -> JSONResponse:
        started_at = time.perf_counter()
        try:
            payload = service.get_run_manifest(run_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if payload is None:
            raise HTTPException(404, f"Manifest for run {run_id} not found")
        return _cached_response(
            payload,
            extra_headers={
                "Server-Timing": _server_timing_header(
                    {"run_manifest": (time.perf_counter() - started_at) * 1000.0}
                )
            },
        )

    @router.get("/runs/{run_id}/metrics")
    def get_run_metrics(run_id: str) -> JSONResponse:
        started_at = time.perf_counter()
        try:
            payload = service.get_run_metrics(run_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if payload is None:
            raise HTTPException(404, f"Metrics for run {run_id} not found")
        return _cached_response(
            payload,
            extra_headers={
                "Server-Timing": _server_timing_header(
                    {"run_metrics": (time.perf_counter() - started_at) * 1000.0}
                )
            },
        )

    @router.get("/runs/{run_id}/events")
    def get_run_events(run_id: str, limit: int = 50) -> JSONResponse:
        started_at = time.perf_counter()
        try:
            payload = service.get_run_events(run_id, limit=_bounded_limit(limit, default=50, max_value=5000))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return _cached_response(
            payload,
            extra_headers={
                "Server-Timing": _server_timing_header(
                    {"run_events": (time.perf_counter() - started_at) * 1000.0}
                )
            },
        )

    @router.get("/runs/{run_id}/resources")
    def get_run_resources(run_id: str, limit: int = 50) -> JSONResponse:
        started_at = time.perf_counter()
        try:
            payload = service.get_run_resources(run_id, limit=_bounded_limit(limit, default=50, max_value=5000))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return _cached_response(
            payload,
            extra_headers={
                "Server-Timing": _server_timing_header(
                    {"run_resources": (time.perf_counter() - started_at) * 1000.0}
                )
            },
        )

    @router.get("/runs/{run_id}/per-era-corr")
    def get_per_era_corr(run_id: str) -> JSONResponse:
        started_at = time.perf_counter()
        try:
            result = service.get_per_era_corr_result(run_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        payload = result.payload
        if payload is None:
            raise HTTPException(404, f"Per-era correlation data for run {run_id} not found")
        return _cached_response(
            payload,
            extra_headers={
                "Server-Timing": _server_timing_header(
                    {
                        "run_per_era_corr_total": (time.perf_counter() - started_at) * 1000.0,
                        "run_per_era_corr_read": result.persisted_read_ms,
                        "run_per_era_corr_materialize": result.materialize_ms,
                    }
                )
            },
        )

    @router.get("/runs/{run_id}/scoring-dashboard")
    def get_scoring_dashboard(run_id: str) -> JSONResponse:
        started_at = time.perf_counter()
        try:
            payload = service.get_scoring_dashboard(run_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if payload is None:
            raise HTTPException(404, f"Scoring dashboard for run {run_id} not found")
        return _cached_response(
            payload,
            extra_headers={
                "Server-Timing": _server_timing_header(
                    {"run_scoring_dashboard": (time.perf_counter() - started_at) * 1000.0}
                )
            },
        )

    @router.get("/runs/{run_id}/trials")
    def get_trials(run_id: str) -> JSONResponse:
        started_at = time.perf_counter()
        try:
            payload = service.get_trials(run_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if payload is None:
            raise HTTPException(404, f"Trials data for run {run_id} not found")
        return _cached_response(
            payload,
            extra_headers={
                "Server-Timing": _server_timing_header(
                    {"run_trials": (time.perf_counter() - started_at) * 1000.0}
                )
            },
        )

    @router.get("/runs/{run_id}/best-params")
    def get_best_params(run_id: str) -> JSONResponse:
        started_at = time.perf_counter()
        try:
            payload = service.get_best_params(run_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if payload is None:
            raise HTTPException(404, f"Best params for run {run_id} not found")
        return _cached_response(
            payload,
            extra_headers={
                "Server-Timing": _server_timing_header(
                    {"run_best_params": (time.perf_counter() - started_at) * 1000.0}
                )
            },
        )

    @router.get("/runs/{run_id}/config")
    def get_resolved_config(run_id: str) -> JSONResponse:
        started_at = time.perf_counter()
        try:
            payload = service.get_resolved_config(run_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if payload is None:
            raise HTTPException(404, f"Resolved config for run {run_id} not found")
        return _cached_response(
            payload,
            extra_headers={
                "Server-Timing": _server_timing_header(
                    {"run_config": (time.perf_counter() - started_at) * 1000.0}
                )
            },
        )

    @router.get("/runs/{run_id}/diagnostics-sources")
    def get_diagnostics_sources(run_id: str) -> JSONResponse:
        started_at = time.perf_counter()
        try:
            payload = service.get_diagnostics_sources(run_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if payload is None:
            raise HTTPException(404, f"Diagnostics sources for run {run_id} not found")
        return _cached_response(
            payload,
            extra_headers={
                "Server-Timing": _server_timing_header(
                    {"run_diagnostics_sources": (time.perf_counter() - started_at) * 1000.0}
                )
            },
        )

    @router.get("/runs/{run_id}/bundle")
    async def get_run_bundle(run_id: str) -> JSONResponse:
        try:
            payload = await service.get_run_bundle(run_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return _cached_response(payload)

    @router.get("/studies")
    def list_studies(
        experiment_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> JSONResponse:
        payload = service.list_studies(
            experiment_id=experiment_id,
            status=status,
            limit=_bounded_limit(limit, default=50, max_value=1000),
            offset=max(0, offset),
        )
        return _nocache_response(payload)

    @router.get("/studies/{study_id}")
    def get_study(study_id: str) -> JSONResponse:
        payload = service.get_study(study_id)
        if payload is None:
            raise HTTPException(404, f"Study {study_id} not found")
        return _nocache_response(payload)

    @router.get("/studies/{study_id}/trials")
    def get_study_trials(study_id: str) -> JSONResponse:
        study = service.get_study(study_id)
        if study is None:
            raise HTTPException(404, f"Study {study_id} not found")
        payload = service.get_study_trials(study_id)
        if payload is None:
            raise HTTPException(404, "Study trials not found")
        return _nocache_response(payload)

    @router.get("/experiments/{experiment_id}/studies")
    def get_experiment_studies(experiment_id: str) -> JSONResponse:
        try:
            payload = service.get_experiment_studies(experiment_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return _nocache_response(payload)

    @router.get("/ensembles")
    def list_ensembles(experiment_id: str | None = None, limit: int = 50, offset: int = 0) -> JSONResponse:
        payload = service.list_ensembles(
            experiment_id=experiment_id,
            limit=_bounded_limit(limit, default=50, max_value=1000),
            offset=max(0, offset),
        )
        return _nocache_response(payload)

    @router.get("/ensembles/{ensemble_id}")
    def get_ensemble(ensemble_id: str) -> JSONResponse:
        try:
            payload = service.get_ensemble(ensemble_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if payload is None:
            raise HTTPException(404, f"Ensemble {ensemble_id} not found")
        return _nocache_response(payload)

    @router.get("/ensembles/{ensemble_id}/correlations")
    def get_ensemble_correlations(ensemble_id: str) -> JSONResponse:
        try:
            ensemble = service.get_ensemble(ensemble_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if ensemble is None:
            raise HTTPException(404, f"Ensemble {ensemble_id} not found")
        try:
            payload = service.get_ensemble_correlations(ensemble_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if payload is None:
            raise HTTPException(404, "correlation_matrix.parquet not found")
        return _nocache_response(payload)

    @router.get("/ensembles/{ensemble_id}/artifacts")
    def get_ensemble_artifacts(ensemble_id: str) -> JSONResponse:
        try:
            ensemble = service.get_ensemble(ensemble_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if ensemble is None:
            raise HTTPException(404, f"Ensemble {ensemble_id} not found")
        try:
            payload = service.get_ensemble_artifacts(ensemble_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if payload is None:
            raise HTTPException(404, "ensemble artifacts not found")
        return _nocache_response(payload)

    @router.get("/experiments/{experiment_id}/ensembles")
    def get_experiment_ensembles(experiment_id: str) -> JSONResponse:
        try:
            payload = service.get_experiment_ensembles(experiment_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return _nocache_response(payload)

    @router.get("/experiments/{experiment_id}/docs/{filename}")
    def get_experiment_doc(experiment_id: str, filename: str) -> JSONResponse:
        _validate_doc_filename(filename, valid=_VALID_EXPERIMENT_DOCS)
        payload = service.get_experiment_doc(experiment_id, filename)
        return _nocache_response(payload)

    @router.get("/runs/{run_id}/docs/{filename}")
    def get_run_doc(run_id: str, filename: str) -> JSONResponse:
        _validate_doc_filename(filename, valid=_VALID_RUN_DOCS)
        payload = service.get_run_doc(run_id, filename)
        return _nocache_response(payload)

    @router.get("/docs/numerai/tree")
    def get_numerai_doc_tree() -> JSONResponse:
        try:
            payload = service.get_doc_tree("numerai")
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return _nocache_response(payload)

    @router.get("/docs/numerai/content")
    def get_numerai_doc_content(path: str) -> JSONResponse:
        try:
            payload = service.get_doc_content("numerai", path)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return _nocache_response(payload)

    @router.get("/docs/numereng/tree")
    def get_numereng_doc_tree() -> JSONResponse:
        try:
            payload = service.get_doc_tree("numereng")
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return _nocache_response(payload)

    @router.get("/docs/numereng/content")
    def get_numereng_doc_content(path: str) -> JSONResponse:
        try:
            payload = service.get_doc_content("numereng", path)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return _nocache_response(payload)

    @router.get("/docs/{domain}/asset")
    def get_doc_asset(domain: str, path: str) -> FileResponse:
        try:
            asset_path = service.get_doc_asset_path(domain, path)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc

        return FileResponse(
            path=asset_path,
            headers={"Cache-Control": "public, max-age=3600"},
        )

    @router.get("/notes/tree")
    def get_notes_tree() -> JSONResponse:
        return _nocache_response(service.get_notes_tree())

    @router.get("/notes/content")
    def get_notes_content(path: str) -> JSONResponse:
        path = _validate_note_path(path)
        try:
            payload = service.get_notes_content(path)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return _nocache_response(payload)

    return router
