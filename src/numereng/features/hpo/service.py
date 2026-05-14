"""HPO orchestration service built on top of training + store features."""

from __future__ import annotations

import hashlib
import json
import re
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from numereng.config.hpo.contracts import canonicalize_hpo_sampler_payload, canonicalize_hpo_study_payload
from numereng.config.training import TrainingConfigLoaderError, ensure_json_config_path, load_training_config_json
from numereng.features.feature_neutralization import (
    NeutralizationDataError,
    NeutralizationValidationError,
    NeutralizePredictionsRequest,
    load_neutralizer_table,
    neutralize_predictions_file,
)
from numereng.features.hpo.artifacts import (
    ensure_safe_study_id,
    read_study_spec,
    resolve_study_storage_path,
    write_study_spec,
    write_study_summary,
    write_trial_config,
    write_trials_table,
)
from numereng.features.hpo.contracts import (
    HpoDirection,
    HpoStatus,
    HpoStopReason,
    HpoStudyCreateRequest,
    HpoStudyRecord,
    HpoStudyResult,
    HpoTrialRecord,
    HpoTrialResult,
)
from numereng.features.hpo.repo import get_study, list_studies, list_trials, save_study, save_trial
from numereng.features.hpo.runner_optuna import HpoOptunaError, HpoOptunaStudyResult, run_optuna_study
from numereng.features.hpo.search_space import HpoSearchSpaceError, apply_param_overrides, resolve_search_space
from numereng.features.scoring.metrics import DEFAULT_META_MODEL_COL
from numereng.features.scoring.models import CanonicalScoringStage, PostTrainingScoringRequest
from numereng.features.scoring.run_service import score_run as score_existing_run
from numereng.features.scoring.service import run_post_training_scoring
from numereng.features.store import StoreError, index_run, resolve_store_root
from numereng.features.telemetry import bind_launch_metadata, get_launch_metadata
from numereng.features.training import TrainingRunPreview, TrainingRunResult, preview_training_run, run_training
from numereng.features.training.client import TrainingDataClient, create_training_data_client
from numereng.features.training.repo import DEFAULT_DATASETS_DIR
from numereng.features.training.run_lock import RUN_LOCK_FILENAME
from numereng.features.training.service import resolve_benchmark_source

_SAFE_ID = re.compile(r"^[\w\-.]+$")
_DEFAULT_HPO_METRIC = "bmc_last_200_eras.mean"
_LEGACY_POST_FOLD_OBJECTIVE_METRIC = "post_fold_champion_objective"
_POST_FOLD_CORR_WEIGHT = 0.25
_POST_FOLD_BMC_WEIGHT = 2.25
_CORE_SCORING_METRIC_PREFIXES = ("bmc", "corr", "cwmm", "mmc")
_CORE_SCORING_METRIC_ROOTS = {"avg_corr_with_benchmark", "max_drawdown"}
_FULL_SCORING_METRIC_PREFIXES = ("feature_exposure", "feature_neutral", "fnc")
_HPO_METRIC_ALIASES = {
    "avg_corr_with_benchmark": (
        "bmc.avg_corr_with_benchmark",
        "bmc_last_200_eras.avg_corr_with_benchmark",
    ),
    "max_drawdown": ("corr.max_drawdown",),
}


class HpoError(Exception):
    """Base error for HPO workflows."""


class HpoValidationError(HpoError):
    """Raised when HPO inputs are invalid."""


class HpoNotFoundError(HpoError):
    """Raised when requested HPO study is missing."""


class HpoDependencyError(HpoError):
    """Raised when optional HPO runtime dependencies are unavailable."""


class HpoExecutionError(HpoError):
    """Raised when trials/studies fail to execute."""


def create_study(
    *,
    store_root: str | Path = ".numereng",
    request: HpoStudyCreateRequest,
) -> HpoStudyResult:
    """Create or resume one Optuna-backed HPO study."""

    _validate_create_request(request)

    try:
        _ = ensure_json_config_path(str(request.config_path), field_name="config_path")
    except TrainingConfigLoaderError as exc:
        raise HpoValidationError(str(exc)) from exc

    config_path = request.config_path.expanduser().resolve()
    if not config_path.is_file():
        raise HpoValidationError(f"hpo_config_not_found:{config_path}")

    try:
        base_config = load_training_config_json(config_path)
    except TrainingConfigLoaderError as exc:
        raise HpoValidationError(str(exc)) from exc

    try:
        param_specs = resolve_search_space(raw_search_space=request.search_space)
    except HpoSearchSpaceError as exc:
        raise HpoValidationError(str(exc)) from exc

    resolved_neutralizer_cols = _resolve_neutralization_cols(request)
    resolved_root = resolve_store_root(store_root)
    storage_path = resolve_study_storage_path(
        store_root=resolved_root,
        experiment_id=request.experiment_id,
        study_id=request.study_id,
    )

    full_spec = canonicalize_hpo_study_payload(
        _study_spec_payload(
            request=request,
            resolved_config_path=config_path,
            resolved_neutralizer_cols=resolved_neutralizer_cols,
        )
    )
    immutable_spec = _immutable_study_spec(
        full_spec=full_spec,
        base_config_hash=_hash_payload(base_config),
    )
    existing_immutable_spec = read_study_spec(storage_path=storage_path)
    if existing_immutable_spec is None:
        write_study_spec(storage_path=storage_path, payload=immutable_spec)
    else:
        existing_immutable_spec = canonicalize_hpo_study_payload(existing_immutable_spec)
        write_study_spec(storage_path=storage_path, payload=existing_immutable_spec)
        _validate_resume_spec(
            study_id=request.study_id,
            existing_immutable_spec=existing_immutable_spec,
            expected_immutable_spec=immutable_spec,
        )

    existing_record = get_study(store_root=resolved_root, study_id=request.study_id)
    existing_trials = list_trials(store_root=resolved_root, study_id=request.study_id)
    trial_results = [_trial_result_from_record(record) for record in existing_trials]
    trial_rows = [_trial_row_payload(record) for record in existing_trials]
    created_at = existing_record.created_at if existing_record is not None else _utc_now_iso()

    entry_stop_reason = _budget_stop_reason(
        trials=trial_results,
        direction=request.objective.direction,
        max_trials=request.stopping.max_trials,
        max_completed_trials=request.stopping.max_completed_trials,
        plateau_enabled=request.stopping.plateau.enabled,
        min_completed_trials=request.stopping.plateau.min_completed_trials,
        patience_completed_trials=request.stopping.plateau.patience_completed_trials,
        min_improvement_abs=request.stopping.plateau.min_improvement_abs,
    )
    if entry_stop_reason is not None:
        terminal_snapshot = _study_snapshot(
            request=request,
            spec=full_spec,
            storage_path=storage_path,
            trials=trial_results,
            status="failed" if entry_stop_reason == "all_trials_failed" else "completed",
            stop_reason=entry_stop_reason,
            created_at=created_at,
            updated_at=_utc_now_iso(),
            error_message="hpo_trials_all_failed" if entry_stop_reason == "all_trials_failed" else None,
        )
        _persist_study_snapshot(
            store_root=resolved_root,
            storage_path=storage_path,
            snapshot=terminal_snapshot,
            trial_rows=trial_rows,
        )
        return terminal_snapshot

    running_snapshot = _study_snapshot(
        request=request,
        spec=full_spec,
        storage_path=storage_path,
        trials=trial_results,
        status="running",
        stop_reason=None,
        created_at=created_at,
        updated_at=_utc_now_iso(),
        error_message=None,
    )
    _persist_study_snapshot(
        store_root=resolved_root,
        storage_path=storage_path,
        snapshot=running_snapshot,
        trial_rows=trial_rows,
    )

    neutralized_metric_client = create_training_data_client() if request.objective.neutralization.enabled else None

    def persist_running_snapshot() -> None:
        snapshot = _study_snapshot(
            request=request,
            spec=full_spec,
            storage_path=storage_path,
            trials=trial_results,
            status="running",
            stop_reason=None,
            created_at=created_at,
            updated_at=_utc_now_iso(),
            error_message=None,
        )
        _persist_study_snapshot(
            store_root=resolved_root,
            storage_path=storage_path,
            snapshot=snapshot,
            trial_rows=trial_rows,
        )

    def objective_callback(trial_number: int, params: dict[str, Any]) -> float:
        started_at = _utc_now_iso()
        trial_config = apply_param_overrides(base_config, params=params)
        trial_config_path = write_trial_config(
            storage_path=storage_path,
            trial_number=trial_number,
            config=trial_config,
        )
        training_result: TrainingRunResult | None = None

        try:
            reused_trial = _find_completed_trial_for_params(trials=trial_results, params=params)
            if reused_trial is not None:
                if reused_trial.value is None:
                    raise HpoExecutionError(
                        f"hpo_trial_reuse_value_missing:{request.study_id}:{reused_trial.trial_number}"
                    )
                trial_payload = HpoTrialResult(
                    study_id=request.study_id,
                    trial_number=trial_number,
                    status="completed",
                    params=params,
                    value=float(reused_trial.value),
                    run_id=reused_trial.run_id,
                    config_path=trial_config_path,
                    started_at=started_at,
                    finished_at=_utc_now_iso(),
                    error_message=None,
                )
                _record_trial_result(
                    store_root=resolved_root,
                    trial_results=trial_results,
                    trial_rows=trial_rows,
                    trial_payload=trial_payload,
                )
                persist_running_snapshot()
                return float(reused_trial.value)

            preview = preview_training_run(
                config_path=trial_config_path,
                output_dir=resolved_root,
                experiment_id=request.experiment_id,
            )
            reused_existing_run = _maybe_reuse_existing_run(
                request=request,
                params=params,
                resolved_neutralizer_cols=resolved_neutralizer_cols,
                neutralized_metric_client=neutralized_metric_client,
                trial_config_path=trial_config_path,
                store_root=resolved_root,
                preview=preview,
            )
            if reused_existing_run is not None:
                reused_run_id, metric_value = reused_existing_run
                trial_payload = HpoTrialResult(
                    study_id=request.study_id,
                    trial_number=trial_number,
                    status="completed",
                    params=params,
                    value=metric_value,
                    run_id=reused_run_id,
                    config_path=trial_config_path,
                    started_at=started_at,
                    finished_at=_utc_now_iso(),
                    error_message=None,
                )
                _record_trial_result(
                    store_root=resolved_root,
                    trial_results=trial_results,
                    trial_rows=trial_rows,
                    trial_payload=trial_payload,
                )
                persist_running_snapshot()
                return metric_value

            launch_scope = (
                nullcontext()
                if get_launch_metadata() is not None
                else bind_launch_metadata(source="api.hpo.create", operation_type="hpo", job_type="hpo")
            )
            with launch_scope:
                training_result = run_training(
                    config_path=trial_config_path,
                    output_dir=str(resolved_root),
                    experiment_id=request.experiment_id,
                )
            index_run(store_root=resolved_root, run_id=training_result.run_id)
            metric_value = _resolve_trial_metric(
                request=request,
                resolved_neutralizer_cols=resolved_neutralizer_cols,
                neutralized_metric_client=neutralized_metric_client,
                trial_config_path=trial_config_path,
                training_result=training_result,
                store_root=resolved_root,
            )
            trial_payload = HpoTrialResult(
                study_id=request.study_id,
                trial_number=trial_number,
                status="completed",
                params=params,
                value=metric_value,
                run_id=training_result.run_id,
                config_path=trial_config_path,
                started_at=started_at,
                finished_at=_utc_now_iso(),
                error_message=None,
            )
            _record_trial_result(
                store_root=resolved_root,
                trial_results=trial_results,
                trial_rows=trial_rows,
                trial_payload=trial_payload,
            )
            persist_running_snapshot()
            return metric_value
        except StoreError as exc:
            error_message = f"hpo_trial_index_failed:{trial_number}"
            trial_payload = HpoTrialResult(
                study_id=request.study_id,
                trial_number=trial_number,
                status="failed",
                params=params,
                value=None,
                run_id=training_result.run_id if training_result is not None else None,
                config_path=trial_config_path,
                started_at=started_at,
                finished_at=_utc_now_iso(),
                error_message=error_message,
            )
            _record_trial_result(
                store_root=resolved_root,
                trial_results=trial_results,
                trial_rows=trial_rows,
                trial_payload=trial_payload,
            )
            persist_running_snapshot()
            raise HpoExecutionError(error_message) from exc
        except Exception as exc:
            trial_payload = HpoTrialResult(
                study_id=request.study_id,
                trial_number=trial_number,
                status="failed",
                params=params,
                value=None,
                run_id=training_result.run_id if training_result is not None else None,
                config_path=trial_config_path,
                started_at=started_at,
                finished_at=_utc_now_iso(),
                error_message=str(exc),
            )
            _record_trial_result(
                store_root=resolved_root,
                trial_results=trial_results,
                trial_rows=trial_rows,
                trial_payload=trial_payload,
            )
            persist_running_snapshot()
            raise

    try:
        run_result = run_optuna_study(
            study_id=request.study_id,
            storage_path=storage_path,
            direction=request.objective.direction,
            sampler=request.sampler,
            stopping=request.stopping,
            specs=param_specs,
            objective_callback=objective_callback,
            summary_callback=persist_running_snapshot,
        )
    except HpoOptunaError as exc:
        failed_snapshot = _study_snapshot(
            request=request,
            spec=full_spec,
            storage_path=storage_path,
            trials=trial_results,
            status="failed",
            stop_reason=_budget_stop_reason(
                trials=trial_results,
                direction=request.objective.direction,
                max_trials=request.stopping.max_trials,
                max_completed_trials=request.stopping.max_completed_trials,
                plateau_enabled=request.stopping.plateau.enabled,
                min_completed_trials=request.stopping.plateau.min_completed_trials,
                patience_completed_trials=request.stopping.plateau.patience_completed_trials,
                min_improvement_abs=request.stopping.plateau.min_improvement_abs,
            ),
            created_at=created_at,
            updated_at=_utc_now_iso(),
            error_message=str(exc),
        )
        _persist_study_snapshot(
            store_root=resolved_root,
            storage_path=storage_path,
            snapshot=failed_snapshot,
            trial_rows=trial_rows,
        )
        raise HpoDependencyError(str(exc)) from exc

    final_stop_reason = run_result.stop_reason
    if final_stop_reason is None:
        final_stop_reason = _budget_stop_reason(
            trials=trial_results,
            direction=request.objective.direction,
            max_trials=request.stopping.max_trials,
            max_completed_trials=request.stopping.max_completed_trials,
            plateau_enabled=request.stopping.plateau.enabled,
            min_completed_trials=request.stopping.plateau.min_completed_trials,
            patience_completed_trials=request.stopping.plateau.patience_completed_trials,
            min_improvement_abs=request.stopping.plateau.min_improvement_abs,
        )
    final_snapshot = _study_snapshot(
        request=request,
        spec=full_spec,
        storage_path=storage_path,
        trials=trial_results,
        status="failed" if final_stop_reason == "all_trials_failed" else "completed",
        stop_reason=final_stop_reason,
        created_at=created_at,
        updated_at=_utc_now_iso(),
        error_message="hpo_trials_all_failed" if final_stop_reason == "all_trials_failed" else None,
        runner_result=run_result,
    )
    _persist_study_snapshot(
        store_root=resolved_root,
        storage_path=storage_path,
        snapshot=final_snapshot,
        trial_rows=trial_rows,
    )
    return final_snapshot


def list_studies_view(
    *,
    store_root: str | Path = ".numereng",
    experiment_id: str | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[HpoStudyRecord, ...]:
    """List persisted HPO studies."""

    return list_studies(
        store_root=store_root,
        experiment_id=experiment_id,
        status=status,
        limit=limit,
        offset=offset,
    )


def get_study_view(*, store_root: str | Path = ".numereng", study_id: str) -> HpoStudyRecord:
    """Load one persisted HPO study."""

    payload = get_study(store_root=store_root, study_id=study_id)
    if payload is None:
        raise HpoNotFoundError(f"hpo_study_not_found:{study_id}")
    return payload


def get_study_trials_view(*, store_root: str | Path = ".numereng", study_id: str) -> tuple[HpoTrialRecord, ...]:
    """Load trial rows for one persisted HPO study."""

    _ = get_study_view(store_root=store_root, study_id=study_id)
    return list_trials(store_root=store_root, study_id=study_id)


def _validate_create_request(request: HpoStudyCreateRequest) -> None:
    if not request.study_name.strip():
        raise HpoValidationError("hpo_study_name_invalid")
    try:
        _ = ensure_safe_study_id(request.study_id)
    except ValueError as exc:
        raise HpoValidationError("hpo_study_id_invalid") from exc
    if request.experiment_id is not None and not _SAFE_ID.match(request.experiment_id):
        raise HpoValidationError(f"hpo_experiment_id_invalid:{request.experiment_id}")
    if not request.objective.metric.strip():
        raise HpoValidationError("hpo_metric_invalid")
    if request.stopping.max_trials < 1:
        raise HpoValidationError("hpo_max_trials_invalid")
    if request.stopping.max_completed_trials is not None and request.stopping.max_completed_trials < 1:
        raise HpoValidationError("hpo_max_completed_trials_invalid")
    if request.stopping.timeout_seconds is not None and request.stopping.timeout_seconds < 1:
        raise HpoValidationError("hpo_timeout_seconds_invalid")
    if request.objective.neutralization.enabled and request.objective.neutralization.neutralizer_path is None:
        raise HpoValidationError("hpo_neutralizer_path_required")


def _resolve_neutralization_cols(request: HpoStudyCreateRequest) -> tuple[str, ...] | None:
    neutralization = request.objective.neutralization
    if not neutralization.enabled:
        return None
    if neutralization.neutralizer_path is None:
        raise HpoValidationError("hpo_neutralizer_path_required")
    try:
        _, resolved_cols = load_neutralizer_table(
            neutralizer_path=neutralization.neutralizer_path,
            neutralizer_cols=neutralization.neutralizer_cols,
        )
    except (NeutralizationValidationError, NeutralizationDataError) as exc:
        raise HpoValidationError(str(exc)) from exc
    return resolved_cols


def _study_spec_payload(
    *,
    request: HpoStudyCreateRequest,
    resolved_config_path: Path,
    resolved_neutralizer_cols: tuple[str, ...] | None,
) -> dict[str, Any]:
    neutralization = request.objective.neutralization
    search_space = request.search_space or {}
    return {
        "study_id": request.study_id,
        "study_name": request.study_name,
        "config_path": str(resolved_config_path),
        "experiment_id": request.experiment_id,
        "objective": {
            "metric": request.objective.metric,
            "direction": request.objective.direction,
            "neutralization": {
                "enabled": neutralization.enabled,
                "neutralizer_path": str(neutralization.neutralizer_path) if neutralization.neutralizer_path else None,
                "proportion": neutralization.proportion,
                "mode": neutralization.mode,
                "neutralizer_cols": list(resolved_neutralizer_cols) if resolved_neutralizer_cols is not None else None,
                "rank_output": neutralization.rank_output,
            },
        },
        "search_space": json.loads(json.dumps(search_space, sort_keys=True)),
        "sampler": canonicalize_hpo_sampler_payload(
            {
                "kind": request.sampler.kind,
                "seed": request.sampler.seed,
                "n_startup_trials": request.sampler.n_startup_trials,
                "multivariate": request.sampler.multivariate,
                "group": request.sampler.group,
            }
        ),
        "stopping": {
            "max_trials": request.stopping.max_trials,
            "max_completed_trials": request.stopping.max_completed_trials,
            "timeout_seconds": request.stopping.timeout_seconds,
            "plateau": {
                "enabled": request.stopping.plateau.enabled,
                "min_completed_trials": request.stopping.plateau.min_completed_trials,
                "patience_completed_trials": request.stopping.plateau.patience_completed_trials,
                "min_improvement_abs": request.stopping.plateau.min_improvement_abs,
            },
        },
    }


def _immutable_study_spec(*, full_spec: dict[str, Any], base_config_hash: str) -> dict[str, Any]:
    return {
        "study_id": full_spec["study_id"],
        "study_name": full_spec["study_name"],
        "config_path": full_spec["config_path"],
        "experiment_id": full_spec["experiment_id"],
        "base_config_hash": base_config_hash,
        "objective": full_spec["objective"],
        "search_space": full_spec["search_space"],
        "sampler": full_spec["sampler"],
    }


def _validate_resume_spec(
    *,
    study_id: str,
    existing_immutable_spec: dict[str, Any],
    expected_immutable_spec: dict[str, Any],
) -> None:
    if existing_immutable_spec != expected_immutable_spec:
        raise HpoValidationError(f"hpo_study_spec_mismatch:{study_id}")


def _hash_payload(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _record_trial_result(
    *,
    store_root: Path,
    trial_results: list[HpoTrialResult],
    trial_rows: list[dict[str, Any]],
    trial_payload: HpoTrialResult,
) -> None:
    save_trial(store_root=store_root, payload=trial_payload)
    _upsert_trial_result(trial_results, trial_payload)
    _upsert_trial_row(trial_rows, trial_payload)


def _upsert_trial_result(trial_results: list[HpoTrialResult], trial_payload: HpoTrialResult) -> None:
    for index, item in enumerate(trial_results):
        if item.trial_number == trial_payload.trial_number:
            trial_results[index] = trial_payload
            return
    trial_results.append(trial_payload)
    trial_results.sort(key=lambda item: item.trial_number)


def _upsert_trial_row(trial_rows: list[dict[str, Any]], trial_payload: HpoTrialResult) -> None:
    row = {
        "trial_number": trial_payload.trial_number,
        "status": trial_payload.status,
        "value": trial_payload.value,
        "run_id": trial_payload.run_id,
        "config_path": str(trial_payload.config_path),
        "params": json.dumps(trial_payload.params, sort_keys=True, ensure_ascii=True),
        "started_at": trial_payload.started_at,
        "finished_at": trial_payload.finished_at,
        "error_message": trial_payload.error_message,
    }
    for index, item in enumerate(trial_rows):
        if int(item.get("trial_number", -1)) == trial_payload.trial_number:
            trial_rows[index] = row
            return
    trial_rows.append(row)
    trial_rows.sort(key=lambda item: int(item["trial_number"]))


def _study_snapshot(
    *,
    request: HpoStudyCreateRequest,
    spec: dict[str, Any],
    storage_path: Path,
    trials: list[HpoTrialResult],
    status: HpoStatus,
    stop_reason: HpoStopReason | None,
    created_at: str,
    updated_at: str,
    error_message: str | None,
    runner_result: HpoOptunaStudyResult | None = None,
) -> HpoStudyResult:
    attempted_trials = runner_result.attempted_trials if runner_result is not None else len(trials)
    completed_trials = (
        runner_result.completed_trials
        if runner_result is not None
        else sum(1 for item in trials if item.status == "completed" and item.value is not None)
    )
    failed_trials = (
        runner_result.failed_trials
        if runner_result is not None
        else sum(1 for item in trials if item.status == "failed")
    )
    best_trial_number, best_value, best_run_id = _best_trial_fields(
        trials=trials,
        direction=request.objective.direction,
        fallback_trial_number=runner_result.best_trial_number if runner_result is not None else None,
        fallback_value=runner_result.best_value if runner_result is not None else None,
    )
    return HpoStudyResult(
        study_id=request.study_id,
        study_name=request.study_name,
        experiment_id=request.experiment_id,
        status=status,
        best_trial_number=best_trial_number,
        best_value=best_value,
        best_run_id=best_run_id,
        spec=spec,
        attempted_trials=attempted_trials,
        completed_trials=completed_trials,
        failed_trials=failed_trials,
        stop_reason=stop_reason,
        storage_path=storage_path,
        trials=tuple(sorted(trials, key=lambda item: item.trial_number)),
        created_at=created_at,
        updated_at=updated_at,
        error_message=error_message,
    )


def _persist_study_snapshot(
    *,
    store_root: Path,
    storage_path: Path,
    snapshot: HpoStudyResult,
    trial_rows: list[dict[str, Any]],
) -> None:
    save_study(store_root=store_root, payload=snapshot)
    write_study_summary(storage_path=storage_path, payload=_study_summary_payload(snapshot))
    write_trials_table(storage_path=storage_path, trials=trial_rows)


def _study_summary_payload(snapshot: HpoStudyResult) -> dict[str, Any]:
    return {
        "study_id": snapshot.study_id,
        "study_name": snapshot.study_name,
        "experiment_id": snapshot.experiment_id,
        "status": snapshot.status,
        "best_trial_number": snapshot.best_trial_number,
        "best_value": snapshot.best_value,
        "best_run_id": snapshot.best_run_id,
        "spec": snapshot.spec,
        "attempted_trials": snapshot.attempted_trials,
        "completed_trials": snapshot.completed_trials,
        "failed_trials": snapshot.failed_trials,
        "stop_reason": snapshot.stop_reason,
        "storage_path": str(snapshot.storage_path),
        "error_message": snapshot.error_message,
        "created_at": snapshot.created_at,
        "updated_at": snapshot.updated_at,
    }


def _trial_result_from_record(record: HpoTrialRecord) -> HpoTrialResult:
    config_path = (
        record.config_path if record.config_path is not None else Path(f"trial_{record.trial_number:04d}.json")
    )
    return HpoTrialResult(
        study_id=record.study_id,
        trial_number=record.trial_number,
        status=record.status,
        params=record.params,
        value=record.value,
        run_id=record.run_id,
        config_path=config_path,
        started_at=record.started_at or record.updated_at,
        finished_at=record.finished_at,
        error_message=record.error_message,
    )


def _trial_row_payload(record: HpoTrialRecord) -> dict[str, Any]:
    return {
        "trial_number": record.trial_number,
        "status": record.status,
        "value": record.value,
        "run_id": record.run_id,
        "config_path": str(record.config_path) if record.config_path else None,
        "params": json.dumps(record.params, sort_keys=True, ensure_ascii=True),
        "started_at": record.started_at,
        "finished_at": record.finished_at,
        "error_message": record.error_message,
    }


def _best_trial_fields(
    *,
    trials: list[HpoTrialResult],
    direction: HpoDirection,
    fallback_trial_number: int | None,
    fallback_value: float | None,
) -> tuple[int | None, float | None, str | None]:
    completed_trials = [trial for trial in trials if trial.status == "completed" and trial.value is not None]
    if not completed_trials:
        return None, None, None
    if fallback_trial_number is not None and fallback_value is not None:
        matched = next((trial for trial in completed_trials if trial.trial_number == fallback_trial_number), None)
        if matched is not None:
            return matched.trial_number, matched.value, matched.run_id
    winner = _best_trial_from_trials(completed_trials, direction=direction)
    return winner.trial_number, winner.value, winner.run_id


def _budget_stop_reason(
    *,
    trials: list[HpoTrialResult],
    direction: HpoDirection,
    max_trials: int,
    max_completed_trials: int | None,
    plateau_enabled: bool,
    min_completed_trials: int,
    patience_completed_trials: int,
    min_improvement_abs: float,
) -> HpoStopReason | None:
    attempted_trials = len(trials)
    completed_values = [float(item.value) for item in trials if item.status == "completed" and item.value is not None]
    failed_trials = sum(1 for item in trials if item.status == "failed")

    if (
        attempted_trials > 0
        and not completed_values
        and failed_trials == attempted_trials
        and attempted_trials >= max_trials
    ):
        return "all_trials_failed"
    if plateau_enabled and _plateau_reached(
        completed_values=completed_values,
        direction=direction,
        min_completed_trials=min_completed_trials,
        patience_completed_trials=patience_completed_trials,
        min_improvement_abs=min_improvement_abs,
    ):
        return "plateau_reached"
    if max_completed_trials is not None and len(completed_values) >= max_completed_trials:
        return "max_completed_trials_reached"
    if attempted_trials >= max_trials:
        return "max_trials_reached"
    return None


def _plateau_reached(
    *,
    completed_values: list[float],
    direction: HpoDirection,
    min_completed_trials: int,
    patience_completed_trials: int,
    min_improvement_abs: float,
) -> bool:
    if len(completed_values) < min_completed_trials:
        return False
    best_value: float | None = None
    last_improvement_index = 0
    for index, value in enumerate(completed_values, start=1):
        if best_value is None:
            best_value = value
            last_improvement_index = index
            continue
        if direction == "minimize":
            improved = value < (best_value - min_improvement_abs)
        else:
            improved = value > (best_value + min_improvement_abs)
        if improved:
            best_value = value
            last_improvement_index = index
    return len(completed_values) - last_improvement_index >= patience_completed_trials


def _best_trial_from_trials(trials: list[HpoTrialResult], *, direction: HpoDirection) -> HpoTrialResult:
    if direction == "minimize":
        return min(trials, key=lambda item: float(item.value if item.value is not None else float("inf")))
    return max(trials, key=lambda item: float(item.value if item.value is not None else float("-inf")))


def _find_completed_trial_for_params(
    *,
    trials: list[HpoTrialResult],
    params: dict[str, Any],
) -> HpoTrialResult | None:
    params_key = _canonical_params_key(params)
    for trial in trials:
        if trial.status != "completed" or trial.value is None or trial.run_id is None:
            continue
        if _canonical_params_key(trial.params) == params_key:
            return trial
    return None


def _canonical_params_key(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _maybe_reuse_existing_run(
    *,
    request: HpoStudyCreateRequest,
    params: dict[str, Any],
    resolved_neutralizer_cols: tuple[str, ...] | None,
    neutralized_metric_client: TrainingDataClient | None,
    trial_config_path: Path,
    store_root: Path,
    preview: TrainingRunPreview,
) -> tuple[str, float] | None:
    run_dir = preview.run_dir
    if not run_dir.exists():
        return None
    preexisting_entries = _run_dir_entries_without_lock(run_dir)
    if not preexisting_entries:
        return None

    manifest = _load_existing_run_manifest(preview.run_manifest_path)
    if manifest is None:
        raise HpoExecutionError(f"hpo_trial_existing_run_not_reusable:{preview.run_id}:manifest_missing:reset_required")
    status = str(manifest.get("status", "")).strip().upper() or "UNKNOWN"
    if status != "FINISHED":
        raise HpoExecutionError(
            f"hpo_trial_existing_run_not_reusable:{preview.run_id}:status={status.lower()}:reset_required"
        )

    try:
        index_run(store_root=store_root, run_id=preview.run_id)
        metric_value = _resolve_trial_metric(
            request=request,
            resolved_neutralizer_cols=resolved_neutralizer_cols,
            neutralized_metric_client=neutralized_metric_client,
            trial_config_path=trial_config_path,
            training_result=TrainingRunResult(
                run_id=preview.run_id,
                predictions_path=preview.predictions_path,
                results_path=preview.results_path,
            ),
            store_root=store_root,
        )
    except (StoreError, HpoExecutionError) as exc:
        raise HpoExecutionError(f"hpo_trial_existing_run_not_reusable:{preview.run_id}:{exc}:reset_required") from exc

    return preview.run_id, metric_value


def _run_dir_entries_without_lock(run_dir: Path) -> list[str]:
    entries = [entry.name for entry in run_dir.iterdir() if entry.name != RUN_LOCK_FILENAME]
    entries.sort()
    return entries


def _load_existing_run_manifest(manifest_path: Path) -> dict[str, Any] | None:
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _resolve_trial_metric(
    *,
    request: HpoStudyCreateRequest,
    resolved_neutralizer_cols: tuple[str, ...] | None,
    neutralized_metric_client: TrainingDataClient | None,
    trial_config_path: Path,
    training_result: TrainingRunResult,
    store_root: Path,
) -> float:
    neutralization = request.objective.neutralization
    if neutralization.enabled:
        if neutralization.neutralizer_path is None:
            raise HpoValidationError("hpo_neutralizer_path_required")
        neutralized_result = neutralize_predictions_file(
            request=NeutralizePredictionsRequest(
                predictions_path=training_result.predictions_path,
                neutralizer_path=neutralization.neutralizer_path,
                proportion=neutralization.proportion,
                mode=neutralization.mode,
                neutralizer_cols=resolved_neutralizer_cols,
                rank_output=neutralization.rank_output,
            ),
            run_id=training_result.run_id,
        )
        if neutralized_metric_client is None:  # pragma: no cover - defensive gate
            raise HpoExecutionError("hpo_neutralization_client_missing")
        return _extract_metric_value_from_predictions(
            predictions_path=neutralized_result.output_path,
            trial_config_path=trial_config_path,
            metric=request.objective.metric,
            scoring_client=neutralized_metric_client,
        )
    if request.objective.metric == _LEGACY_POST_FOLD_OBJECTIVE_METRIC:
        return _extract_post_fold_objective_value(
            store_root=store_root,
            run_id=training_result.run_id,
            results_path=training_result.results_path,
        )
    return _extract_metric_value_with_auto_scoring(
        metric=request.objective.metric,
        training_result=training_result,
        store_root=store_root,
    )


def _extract_metric_value_with_auto_scoring(
    *,
    metric: str,
    training_result: TrainingRunResult,
    store_root: Path,
) -> float:
    try:
        return _extract_metric_value(results_path=training_result.results_path, metric=metric)
    except HpoExecutionError as exc:
        if str(exc) != f"hpo_metric_not_found:{metric}":
            raise
        required_stage = _auto_scoring_stage_for_metric(metric)
        if required_stage is None:
            raise

    try:
        score_existing_run(run_id=training_result.run_id, store_root=store_root, stage=required_stage)
    except Exception as exc:
        raise HpoExecutionError(f"hpo_auto_scoring_failed:{training_result.run_id}:{required_stage}:{exc}") from exc

    return _extract_metric_value(results_path=training_result.results_path, metric=metric)


def _auto_scoring_stage_for_metric(metric: str) -> CanonicalScoringStage | None:
    root = metric.strip()
    while root.startswith("metrics."):
        root = root.removeprefix("metrics.")
    root = root.split(".", maxsplit=1)[0]
    if root.startswith(_FULL_SCORING_METRIC_PREFIXES):
        return "post_training_full"
    if root in _CORE_SCORING_METRIC_ROOTS or root.startswith(_CORE_SCORING_METRIC_PREFIXES):
        return "post_training_core"
    return None


def _extract_metric_value(*, results_path: Path, metric: str) -> float:
    if not results_path.exists():
        raise HpoExecutionError(f"hpo_results_not_found:{results_path}")

    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HpoExecutionError("hpo_results_invalid") from exc

    value = _lookup_hpo_metric_value(payload=payload, metric=metric)
    if value is None:
        raise HpoExecutionError(f"hpo_metric_not_found:{metric}")
    return value


def _extract_post_fold_objective_value(
    *,
    store_root: Path,
    run_id: str,
    results_path: Path | None = None,
) -> float:
    snapshot_path = store_root / "runs" / run_id / "artifacts" / "scoring" / "post_fold_snapshots.parquet"
    if snapshot_path.exists():
        try:
            snapshot = pd.read_parquet(snapshot_path)
            if snapshot.empty:
                raise HpoExecutionError(f"hpo_post_fold_snapshots_empty:{snapshot_path}")
            corr_mean = _snapshot_metric_mean(
                snapshot=snapshot,
                columns=("corr_ender20_fold_mean", "corr_native_fold_mean"),
            )
            bmc_mean = _snapshot_metric_mean(
                snapshot=snapshot,
                columns=("bmc_fold_mean", "bmc_ender20_fold_mean"),
            )
            objective_value = _combine_default_hpo_metric(corr_value=corr_mean, bmc_value=bmc_mean)
            if objective_value is not None:
                return objective_value
            raise HpoExecutionError(
                "hpo_post_fold_metric_missing:"
                "corr_ender20_fold_mean,corr_native_fold_mean,bmc_fold_mean,bmc_ender20_fold_mean"
            )
        except HpoExecutionError:
            if results_path is None:
                raise
        except Exception as exc:
            if results_path is None:
                raise HpoExecutionError(f"hpo_post_fold_snapshots_invalid:{snapshot_path}") from exc
    if results_path is None:
        raise HpoExecutionError(f"hpo_post_fold_snapshots_not_found:{snapshot_path}")
    return _extract_legacy_post_fold_objective_from_results(results_path=results_path)


def _extract_metric_value_from_predictions(
    *,
    predictions_path: Path,
    trial_config_path: Path,
    metric: str,
    scoring_client: TrainingDataClient,
) -> float:
    try:
        trial_config = load_training_config_json(trial_config_path)
    except TrainingConfigLoaderError as exc:
        raise HpoExecutionError(f"hpo_trial_config_invalid:{trial_config_path}") from exc

    data_config = _as_mapping(trial_config.get("data"))
    data_version = str(data_config.get("data_version", "v5.2"))
    dataset_variant = str(data_config.get("dataset_variant", ""))
    if dataset_variant not in {"non_downsampled", "downsampled"}:
        raise HpoExecutionError("hpo_trial_dataset_variant_invalid")
    feature_set = str(data_config.get("feature_set", "small"))
    target_col = str(data_config.get("target_col", "target"))
    era_col = str(data_config.get("era_col", "era"))
    id_col = str(data_config.get("id_col", "id"))
    dataset_scope = str(data_config.get("dataset_scope", "train_only"))
    meta_model_data_path = _optional_path(data_config.get("meta_model_data_path"))
    meta_model_col = str(data_config.get("meta_model_col", DEFAULT_META_MODEL_COL))
    try:
        scoring_result = run_post_training_scoring(
            request=PostTrainingScoringRequest(
                run_id=predictions_path.stem,
                config_hash="",
                seed=None,
                predictions_path=predictions_path,
                pred_cols=("prediction",),
                target_col=target_col,
                scoring_target_cols=(target_col, "target_ender_20"),
                scoring_targets_explicit=False,
                data_version=data_version,
                dataset_variant=dataset_variant,
                feature_set=feature_set,
                feature_source_paths=None,
                dataset_scope=dataset_scope,
                benchmark_source=resolve_benchmark_source(data_config=data_config, data_root=DEFAULT_DATASETS_DIR),
                meta_model_col=meta_model_col,
                meta_model_data_path=meta_model_data_path,
                era_col=era_col,
                id_col=id_col,
                data_root=DEFAULT_DATASETS_DIR,
                stage="all",
            ),
            client=scoring_client,
        )
    except Exception as exc:
        raise HpoExecutionError(f"hpo_neutralized_metric_compute_failed:{exc}") from exc

    metrics_payload = _metrics_payload_from_summaries(summaries=scoring_result.summaries)
    if metric == _LEGACY_POST_FOLD_OBJECTIVE_METRIC:
        value = _legacy_post_fold_objective_from_payload(metrics_payload)
    else:
        value = _lookup_hpo_metric_value(payload={"metrics": metrics_payload}, metric=metric)
    if value is None:
        raise HpoExecutionError(f"hpo_metric_not_found:{metric}")
    return value


def _metrics_payload_from_summaries(*, summaries: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in ("corr", "fnc", "mmc", "cwmm", "bmc", "bmc_last_200_eras"):
        summary = summaries.get(key)
        if not isinstance(summary, pd.DataFrame):
            continue
        if "prediction" not in summary.index:
            continue
        payload[key] = summary.loc["prediction"].to_dict()
    return payload


def _extract_legacy_post_fold_objective_from_results(*, results_path: Path) -> float:
    if not results_path.exists():
        raise HpoExecutionError(f"hpo_results_not_found:{results_path}")

    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HpoExecutionError("hpo_results_invalid") from exc

    metrics_payload = payload.get("metrics")
    if not isinstance(metrics_payload, dict):
        raise HpoExecutionError(f"hpo_metric_not_found:{_LEGACY_POST_FOLD_OBJECTIVE_METRIC}")

    value = _legacy_post_fold_objective_from_payload(metrics_payload)
    if value is None:
        raise HpoExecutionError(f"hpo_metric_not_found:{_LEGACY_POST_FOLD_OBJECTIVE_METRIC}")
    return value


def _legacy_post_fold_objective_from_payload(metrics_payload: dict[str, Any]) -> float | None:
    payload = {"metrics": metrics_payload}
    corr_value = _metric_lookup(payload, "metrics.corr.mean")
    bmc_value = _metric_lookup(payload, "metrics.bmc_last_200_eras.mean")
    if bmc_value is None:
        bmc_value = _metric_lookup(payload, "metrics.bmc.mean")
    if bmc_value is None:
        bmc_value = _metric_lookup(payload, _aliased_metric_path(metrics_payload, "bmc_last_200_eras"))
    if bmc_value is None:
        bmc_value = _metric_lookup(payload, _aliased_metric_path(metrics_payload, "bmc"))

    return _combine_default_hpo_metric(corr_value=corr_value, bmc_value=bmc_value)


def _snapshot_metric_mean(*, snapshot: pd.DataFrame, columns: tuple[str, ...]) -> float | None:
    for column in columns:
        if column not in snapshot.columns:
            continue
        value = float(pd.Series(snapshot[column], dtype="float64").dropna().mean())
        if not pd.isna(value):
            return value
    return None


def _combine_default_hpo_metric(*, corr_value: float | None, bmc_value: float | None) -> float | None:
    if corr_value is None and bmc_value is None:
        return None
    if corr_value is None:
        return bmc_value
    if bmc_value is None:
        return corr_value
    return (_POST_FOLD_CORR_WEIGHT * corr_value) + (_POST_FOLD_BMC_WEIGHT * bmc_value)


def _aliased_metric_path(metrics_payload: dict[str, Any], metric_name: str) -> str:
    if metric_name == "bmc":
        alias_keys = [
            key for key in metrics_payload if key.startswith("bmc_") and not key.startswith("bmc_last_200_eras_")
        ]
    else:
        alias_keys = [key for key in metrics_payload if key.startswith(f"{metric_name}_")]
    if len(alias_keys) != 1:
        return ""
    return f"metrics.{alias_keys[0]}.mean"


def _as_mapping(value: object) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}


def _optional_path(value: object) -> str | Path | None:
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        return value
    return None


def _metric_lookup(payload: dict[str, Any], metric: str) -> float | None:
    cursor: Any = payload
    for token in metric.split("."):
        if not isinstance(cursor, dict):
            return None
        cursor = cursor.get(token)
    if isinstance(cursor, bool):
        return None
    if isinstance(cursor, (int, float)):
        return float(cursor)
    return None


def _lookup_hpo_metric_value(*, payload: dict[str, Any], metric: str) -> float | None:
    value = _metric_lookup(payload, metric)
    if value is not None:
        return value
    if not metric.startswith("metrics."):
        value = _metric_lookup(payload, f"metrics.{metric}")
        if value is not None:
            return value
    for alias in _hpo_metric_aliases(metric):
        value = _metric_lookup(payload, alias)
        if value is not None:
            return value
        value = _metric_lookup(payload, f"metrics.{alias}")
        if value is not None:
            return value
    return None


def _hpo_metric_aliases(metric: str) -> tuple[str, ...]:
    normalized = metric.strip()
    if "." in normalized:
        return ()
    return _HPO_METRIC_ALIASES.get(normalized, ())


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()
