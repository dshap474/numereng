"""Internal staged orchestration for local training runs."""

from __future__ import annotations

import gc
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pandas as pd

from numereng.config.training.contracts import PostTrainingScoringPolicy
from numereng.features.scoring.metrics import (
    append_post_fold_artifacts,
    attach_benchmark_predictions,
    load_custom_benchmark_predictions,
)
from numereng.features.scoring.models import BenchmarkSource, CanonicalScoringStage, PostTrainingScoringRequest
from numereng.features.scoring.service import run_scoring
from numereng.features.store import StoreError, index_run
from numereng.features.telemetry import (
    LocalResourceSampler,
    LocalRunTelemetrySession,
    begin_local_training_session,
    emit_stage_event,
    get_launch_metadata,
    get_run_lifecycle,
    mark_job_running,
    mark_job_starting,
    start_local_resource_sampler,
    stop_local_resource_sampler,
)
from numereng.features.training.client import TrainingDataClient, create_training_data_client
from numereng.features.training.cv import (
    build_full_history_predictions,
    build_oof_predictions,
    era_cv_splits,
    fit_full_history_model,
)
from numereng.features.training.errors import TrainingCanceledError, TrainingConfigError, TrainingError
from numereng.features.training.mlflow_tracking import maybe_log_training_run
from numereng.features.training.model_artifacts import (
    ModelArtifactError,
    ModelArtifactManifest,
    save_model_artifact,
)
from numereng.features.training.models import (
    ModelDataLoaderProtocol,
    TrainingRunPreview,
    TrainingRunResult,
    build_model_data_loader,
    build_x_cols,
    normalize_x_groups,
)
from numereng.features.training.repo import (
    DEFAULT_DATASETS_DIR,
    apply_missing_all_twos_as_nan,
    ensure_split_dataset_paths,
    list_lazy_source_eras,
    load_config,
    load_features,
    load_full_data,
    resolve_metrics_path,
    resolve_output_locations,
    resolve_predictions_path,
    resolve_resolved_config_path,
    resolve_results_path,
    resolve_run_manifest_path,
    resolve_runtime_snapshot_path,
    resolve_score_provenance_path,
    save_metrics,
    save_predictions,
    save_resolved_config,
    save_results,
    save_run_manifest,
    save_score_provenance,
    save_scoring_artifacts,
    select_prediction_columns,
)
from numereng.features.training.run_lock import (
    RunLock,
    acquire_run_lock,
    build_local_attempt_id,
    release_run_lock,
)
from numereng.features.training.run_log import initialize_run_log, log_error, log_info, resolve_run_log_path
from numereng.features.training.run_store import (
    build_run_manifest,
    compute_config_hash,
    compute_run_hash,
    error_payload,
    run_id_from_hash,
)
from numereng.features.training.service import (
    _SIMPLE_PROFILE,
    _apply_resource_policy,
    _as_dict,
    _coerce_float,
    _coerce_int,
    _coerce_optional_int,
    _ensure_run_dir_is_fresh,
    _extract_metrics_payload,
    _is_full_history_refit_profile,
    _lifecycle_manifest_payload,
    _mark_telemetry_canceled,
    _mark_telemetry_completed,
    _mark_telemetry_failed,
    _optional_path,
    _raise_if_cancel_requested,
    _record_telemetry_metrics,
    _record_telemetry_stage,
    _resolve_baseline_path,
    _resolve_cache_policy,
    _resolve_dataset_scope,
    _resolve_dataset_scope_for_profile,
    _resolve_dataset_variant,
    _resolve_resource_policy,
    _resolve_scoring_target_cols,
    _resolve_store_root_for_run,
    _scoring_targets_explicit,
    build_results_payload,
    is_round_post_training_scoring_policy,
    post_training_scoring_requested_stage,
    resolve_benchmark_source,
    resolve_model_config,
    resolve_post_training_scoring_policy,
)
from numereng.features.training.strategies import TrainingEnginePlan, TrainingProfile, resolve_training_engine
from numereng.platform.run_execution import build_local_run_execution, merge_run_execution

logger = logging.getLogger(__name__)
_EMITTED_STAGE_FILE_ORDER = (
    "run_metric_series",
    "post_fold_per_era",
    "post_fold_snapshots",
    "post_training_core_summary",
    "post_training_full_summary",
)
_STAGE_PROGRESS_STARTS: dict[str, float] = {
    "queued": 0.0,
    "initializing": 5.0,
    "load_data": 15.0,
    "train_model": 15.0,
    "score_predictions": 80.0,
    "persist_artifacts": 85.0,
    "index_run": 92.0,
    "score_predictions_post_run": 95.0,
    "finalize_manifest": 99.0,
}
_TRAIN_MODEL_PROGRESS_START = 15.0
_TRAIN_MODEL_PROGRESS_END = 80.0
_PROGRESS_LABELS: dict[str, str] = {
    "queued": "Queued",
    "initializing": "Initializing",
    "load_data": "Loading data",
    "train_model": "Training model",
    "score_predictions": "Scoring predictions",
    "persist_artifacts": "Persisting artifacts",
    "index_run": "Indexing run",
    "score_predictions_post_run": "Post-run scoring",
    "finalize_manifest": "Finalizing manifest",
}


@dataclass
class TrainingPipelineState:
    """Mutable state shared across staged local training execution."""

    config_path: Path
    output_dir: Path | None
    client: TrainingDataClient | None
    profile: TrainingProfile | None
    post_training_scoring: PostTrainingScoringPolicy | None
    engine_mode: str | None
    window_size_eras: int | None
    embargo_eras: int | None
    experiment_id: str | None
    allow_round_batch_post_training_scoring: bool = False
    config: dict[str, object] | None = None
    data_config: dict[str, object] = field(default_factory=dict)
    preprocessing_config: dict[str, object] = field(default_factory=dict)
    training_config: dict[str, object] = field(default_factory=dict)
    model_config: dict[str, object] = field(default_factory=dict)
    data_version: str = "v5.2"
    dataset_variant: str = "non_downsampled"
    feature_set: str = "small"
    target_col: str = "target"
    era_col: str = "era"
    id_col: str = "id"
    benchmark_source: BenchmarkSource | None = None
    meta_model_data_path: str | Path | None = None
    meta_model_col: str = "numerai_meta_model"
    data_embargo_eras: int | None = None
    dataset_scope: str = "train_only"
    nan_missing_all_twos: bool = False
    missing_value: float = 2.0
    resource_policy: dict[str, object] = field(default_factory=dict)
    cache_policy: dict[str, object] = field(default_factory=dict)
    engine_plan: TrainingEnginePlan | None = None
    model_type: str = "LGBMRegressor"
    model_params: dict[str, object] = field(default_factory=dict)
    x_groups: list[str] = field(default_factory=list)
    run_hash: str = ""
    run_id: str = ""
    config_hash: str = ""
    output_root: Path | None = None
    baselines_dir: Path | None = None
    results_dir: Path | None = None
    predictions_dir: Path | None = None
    scoring_dir: Path | None = None
    run_manifest_path: Path | None = None
    runtime_path: Path | None = None
    resolved_snapshot_path: Path | None = None
    metrics_path: Path | None = None
    score_provenance_path: Path | None = None
    training_runtime_metadata: dict[str, object] = field(default_factory=dict)
    created_at: str | None = None
    run_log_path: Path | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    predictions_path: Path | None = None
    predictions_relative: Path | None = None
    results_path: Path | None = None
    score_provenance_relative: Path | None = None
    run_store_root: Path | None = None
    telemetry_session: LocalRunTelemetrySession | None = None
    telemetry_sampler: LocalResourceSampler | None = None
    completed_stages: list[str] = field(default_factory=list)
    run_attempt_id: str = ""
    run_manifest_written: bool = False
    run_lock: RunLock | None = None
    training_client: TrainingDataClient | None = None
    features: list[str] = field(default_factory=list)
    full: pd.DataFrame | None = None
    baseline: pd.DataFrame | None = None
    baseline_col: str | None = None
    baseline_name: str | None = None
    baseline_predictions_path: str | None = None
    baseline_pred_col: str = "prediction"
    x_cols: list[str] = field(default_factory=list)
    data_loader: ModelDataLoaderProtocol | None = None
    all_eras: list[object] = field(default_factory=list)
    full_rows: int = 0
    full_eras: int = 0
    cv_enabled: bool = True
    cv_meta: dict[str, object] = field(default_factory=dict)
    effective_embargo_eras: int = 0
    predictions: pd.DataFrame | None = None
    fitted_model: object | None = None
    oof_rows: int = 0
    oof_eras: int = 0
    scoring_feature_source_paths: tuple[Path, ...] | None = None
    summaries: dict[str, pd.DataFrame] | None = None
    metrics_status: dict[str, object] | None = None
    metrics_payload: dict[str, object] = field(default_factory=dict)
    post_training_scoring_policy: PostTrainingScoringPolicy = "none"
    post_training_requested_stage: CanonicalScoringStage | None = None
    cancel_requested_at: str | None = None
    train_model_progress_total: int | None = None
    train_model_progress_current: int = 0
    run_execution: dict[str, object] = field(default_factory=dict)
    model_artifact_path: Path | None = None
    model_manifest_path: Path | None = None


def _stage_progress_payload(
    state: TrainingPipelineState,
    *,
    stage_name: str,
) -> dict[str, object]:
    if stage_name == "train_model":
        return _train_model_progress_payload(state)
    return {
        "progress_percent": _STAGE_PROGRESS_STARTS.get(stage_name),
        "progress_label": _PROGRESS_LABELS.get(stage_name),
        "progress_current": None,
        "progress_total": None,
    }


def _train_model_progress_payload(state: TrainingPipelineState) -> dict[str, object]:
    total = state.train_model_progress_total or 1
    current = max(0, min(state.train_model_progress_current, total))
    percent = _TRAIN_MODEL_PROGRESS_START
    if total > 0:
        percent = _TRAIN_MODEL_PROGRESS_START + (
            (_TRAIN_MODEL_PROGRESS_END - _TRAIN_MODEL_PROGRESS_START) * (current / total)
        )
    engine_mode = state.engine_plan.mode if state.engine_plan is not None else None
    if _is_full_history_refit_profile(engine_mode):
        label = "Full fit"
    elif current >= total:
        label = f"Fold {total} of {total}"
    else:
        label = f"Fold {current + 1} of {total}"
    return {
        "progress_percent": percent,
        "progress_label": label,
        "progress_current": current,
        "progress_total": total,
    }


def _resolve_train_model_progress_total(
    *,
    state: TrainingPipelineState,
    cv_config: dict[str, object],
) -> int:
    engine_mode = state.engine_plan.mode if state.engine_plan is not None else None
    if _is_full_history_refit_profile(engine_mode):
        return 1
    cv_mode_raw = cv_config.get("mode")
    cv_mode = str(cv_mode_raw).strip() if isinstance(cv_mode_raw, str) else ""
    try:
        splits = era_cv_splits(
            state.all_eras,
            embargo=_coerce_int(cv_config.get("embargo"), default=0),
            mode=cv_mode or "official_walkforward",
            min_train_size=_coerce_int(cv_config.get("min_train_size"), default=0),
            chunk_size=_coerce_int(cv_config.get("chunk_size"), default=156),
            holdout_train_eras=cast(list[object] | None, cv_config.get("train_eras")),
            holdout_val_eras=cast(list[object] | None, cv_config.get("val_eras")),
        )
    except TrainingConfigError:
        return 1
    total = 0
    for train_eras, val_eras in splits:
        if train_eras and val_eras:
            total += 1
    return max(total, 1)


def _emit_train_model_progress(state: TrainingPipelineState) -> None:
    session = state.telemetry_session
    if session is None:
        return
    try:
        emit_stage_event(
            session,
            current_stage="train_model",
            completed_stages=list(state.completed_stages),
            extra_payload=_train_model_progress_payload(state),
        )
    except Exception:
        logger.exception("failed to emit train_model progress for run_id=%s", state.run_id)


def prepare_training_run(
    *,
    config_path: str | Path,
    output_dir: str | Path | None = None,
    client: TrainingDataClient | None = None,
    profile: TrainingProfile | None = None,
    post_training_scoring: PostTrainingScoringPolicy | None = None,
    engine_mode: str | None = None,
    window_size_eras: int | None = None,
    embargo_eras: int | None = None,
    experiment_id: str | None = None,
    allow_round_batch_post_training_scoring: bool = False,
    state: TrainingPipelineState | None = None,
) -> TrainingPipelineState:
    """Resolve config, run identity, runtime policy, and persistent run scaffolding."""
    if state is None:
        state = TrainingPipelineState(
            config_path=Path(config_path).expanduser().resolve(),
            output_dir=Path(output_dir).expanduser() if output_dir is not None else None,
            client=client,
            profile=profile,
            post_training_scoring=post_training_scoring,
            engine_mode=engine_mode,
            window_size_eras=window_size_eras,
            embargo_eras=embargo_eras,
            experiment_id=experiment_id,
            allow_round_batch_post_training_scoring=allow_round_batch_post_training_scoring,
        )

    state = _resolve_training_preview_state(state)
    output_root = cast(Path, state.output_root)
    state.run_attempt_id = build_local_attempt_id(state.run_id)
    state.run_lock = acquire_run_lock(
        run_dir=output_root,
        run_id=state.run_id,
        attempt_id=state.run_attempt_id,
    )

    _ensure_run_dir_is_fresh(output_root)
    launch_metadata = get_launch_metadata()
    if launch_metadata is None:
        raise TrainingError("training_launch_metadata_missing")
    state.run_execution = (
        merge_run_execution(None, launch_metadata.execution)
        if launch_metadata.execution is not None
        else build_local_run_execution(source=launch_metadata.source)
    )
    state.run_store_root = _resolve_store_root_for_run(output_root)
    state.telemetry_session = begin_local_training_session(
        store_root=state.run_store_root,
        config_path=state.config_path,
        run_id=state.run_id,
        run_hash=state.run_hash,
        config_hash=state.config_hash,
        run_dir=output_root,
        runtime_path=cast(Path, state.runtime_path),
        source=launch_metadata.source,
        experiment_id=state.experiment_id,
        operation_type=launch_metadata.operation_type,
        job_type=launch_metadata.job_type,
        request_payload={
            "run_hash": state.run_hash,
            "engine_mode": state.engine_plan.mode,
            "data_version": state.data_version,
            "feature_set": state.feature_set,
            "target_col": state.target_col,
        },
    )
    if state.telemetry_session is None:
        raise TrainingError(f"training_lifecycle_bootstrap_failed:{state.run_id}")
    mark_job_starting(state.telemetry_session, pid=os.getpid(), worker_id="local")
    mark_job_running(state.telemetry_session)
    state.telemetry_sampler = start_local_resource_sampler(
        state.telemetry_session,
        interval_seconds=5.0,
    )
    _record_telemetry_stage(
        state.telemetry_session,
        completed_stages=state.completed_stages,
        stage_name="initializing",
        message="Training session initialized.",
        run_log_path=None,
        attempt_id=state.run_attempt_id,
        extra_payload=_stage_progress_payload(state, stage_name="initializing"),
    )

    running_manifest = build_run_manifest(
        run_id=state.run_id,
        run_hash=state.run_hash,
        status="RUNNING",
        config_path=state.config_path,
        config_hash=state.config_hash,
        data_version=state.data_version,
        feature_set=state.feature_set,
        target_col=state.target_col,
        model_type=state.model_type,
        engine_mode=state.engine_plan.mode,
        experiment_id=state.experiment_id,
        artifacts=state.artifacts,
        training_metadata=state.training_runtime_metadata,
        execution=state.run_execution,
    )
    save_run_manifest(running_manifest, state.run_manifest_path)
    state.run_manifest_written = True
    try:
        initialize_run_log(output_root)
    except Exception:
        logger.exception("failed to initialize run-local log for run_id=%s", state.run_id)
    log_info(
        state.run_log_path,
        event="run_started",
        message=f"run_id={state.run_id} engine_mode={state.engine_plan.mode}",
        attempt_id=state.run_attempt_id,
    )
    save_resolved_config(state.config, state.resolved_snapshot_path)
    state.created_at = str(running_manifest["created_at"])
    state.training_client = create_training_data_client() if state.client is None else state.client
    return state


def preview_training_run(
    *,
    config_path: str | Path,
    output_dir: str | Path | None = None,
    client: TrainingDataClient | None = None,
    profile: TrainingProfile | None = None,
    post_training_scoring: PostTrainingScoringPolicy | None = None,
    engine_mode: str | None = None,
    window_size_eras: int | None = None,
    embargo_eras: int | None = None,
    experiment_id: str | None = None,
    allow_round_batch_post_training_scoring: bool = False,
) -> TrainingRunPreview:
    """Resolve deterministic training run identity and artifact paths without executing."""

    state = TrainingPipelineState(
        config_path=Path(config_path).expanduser().resolve(),
        output_dir=Path(output_dir).expanduser() if output_dir is not None else None,
        client=client,
        profile=profile,
        post_training_scoring=post_training_scoring,
        engine_mode=engine_mode,
        window_size_eras=window_size_eras,
        embargo_eras=embargo_eras,
        experiment_id=experiment_id,
        allow_round_batch_post_training_scoring=allow_round_batch_post_training_scoring,
    )
    state = _resolve_training_preview_state(state)
    output_root = cast(Path, state.output_root)
    predictions_dir = cast(Path, state.predictions_dir)
    results_dir = cast(Path, state.results_dir)
    return TrainingRunPreview(
        run_id=state.run_id,
        run_hash=state.run_hash,
        config_hash=state.config_hash,
        run_dir=output_root,
        predictions_path=resolve_predictions_path(state.config, state.config_path, predictions_dir),
        results_path=resolve_results_path(state.config, state.config_path, results_dir),
        scoring_dir=cast(Path, state.scoring_dir),
        run_manifest_path=cast(Path, state.run_manifest_path),
    )


def _resolve_training_preview_state(state: TrainingPipelineState) -> TrainingPipelineState:
    """Resolve config, run identity, and artifact paths without side effects."""

    state.config = load_config(state.config_path)
    state.data_config = _as_dict(state.config.get("data"))
    state.preprocessing_config = _as_dict(state.config.get("preprocessing"))
    state.training_config = _as_dict(state.config.get("training"))
    state.model_config = _as_dict(state.config.get("model"))
    config = state.config

    state.data_version = str(state.data_config.get("data_version", "v5.2"))
    state.dataset_variant = _resolve_dataset_variant(state.data_config.get("dataset_variant"))
    state.feature_set = str(state.data_config.get("feature_set", "small"))
    state.target_col = str(state.data_config.get("target_col", "target"))
    state.era_col = str(state.data_config.get("era_col", "era"))
    state.id_col = str(state.data_config.get("id_col", "id"))
    state.benchmark_source = resolve_benchmark_source(
        data_config=state.data_config,
        data_root=DEFAULT_DATASETS_DIR,
    )
    state.meta_model_data_path = _optional_path(state.data_config.get("meta_model_data_path"))
    state.meta_model_col = str(state.data_config.get("meta_model_col", "numerai_meta_model"))
    state.data_embargo_eras = _coerce_optional_int(state.data_config.get("embargo_eras"))
    dataset_scope_config = _resolve_dataset_scope(state.data_config.get("dataset_scope"))

    state.nan_missing_all_twos = bool(state.preprocessing_config.get("nan_missing_all_twos", False))
    state.missing_value = _coerce_float(state.preprocessing_config.get("missing_value"), default=2.0)
    state.resource_policy = _resolve_resource_policy(_as_dict(state.training_config.get("resources")))
    state.cache_policy = _resolve_cache_policy(_as_dict(state.training_config.get("cache")))
    _apply_resource_policy(state.resource_policy)
    state.post_training_scoring_policy = resolve_post_training_scoring_policy(
        training_config=state.training_config,
        override=state.post_training_scoring,
    )
    state.post_training_requested_stage = post_training_scoring_requested_stage(state.post_training_scoring_policy)
    if is_round_post_training_scoring_policy(state.post_training_scoring_policy) and not (
        state.allow_round_batch_post_training_scoring
    ):
        raise TrainingError("training_post_training_scoring_round_requires_experiment_workflow")

    state.engine_plan = resolve_training_engine(
        training_config=state.training_config,
        data_config=state.data_config,
        profile=state.profile,
        engine_mode=state.engine_mode,
        window_size_eras=state.window_size_eras,
        embargo_eras=state.embargo_eras,
    )
    state.dataset_scope = _resolve_dataset_scope_for_profile(
        profile=state.engine_plan.mode,
        configured_scope=dataset_scope_config,
        dataset_variant=state.dataset_variant,
    )
    state.model_type, state.model_params = resolve_model_config(state.model_config)

    raw_x_groups = state.model_config.get("x_groups") or state.model_config.get("data_needed")
    if raw_x_groups is None:
        x_groups_input: list[str] | None = None
    elif isinstance(raw_x_groups, (list, tuple)):
        x_groups_input = [str(item) for item in raw_x_groups]
    else:
        raise TrainingConfigError("training_model_x_groups_invalid")
    try:
        state.x_groups = normalize_x_groups(x_groups_input)
    except ValueError as exc:
        raise TrainingConfigError(str(exc)) from exc

    state.run_hash = compute_run_hash(
        config=config,
        data_version=state.data_version,
        feature_set=state.feature_set,
        target_col=state.target_col,
        model_type=state.model_type,
        engine_mode=state.engine_plan.mode,
        engine_settings=state.engine_plan.resolved_config,
    )
    state.run_id = run_id_from_hash(state.run_hash)
    state.config_hash = compute_config_hash(config)

    output_root, baselines_dir, results_dir, predictions_dir, scoring_dir = resolve_output_locations(
        config,
        state.output_dir,
        state.run_id,
    )
    state.output_root = output_root
    state.baselines_dir = baselines_dir
    state.results_dir = results_dir
    state.predictions_dir = predictions_dir
    state.scoring_dir = scoring_dir
    state.run_manifest_path = resolve_run_manifest_path(output_root)
    state.runtime_path = resolve_runtime_snapshot_path(output_root)
    state.resolved_snapshot_path = resolve_resolved_config_path(output_root)
    state.metrics_path = resolve_metrics_path(output_root)
    state.score_provenance_path = resolve_score_provenance_path(output_root)
    state.training_runtime_metadata = {
        "data": {
            "dataset_scope": state.dataset_scope,
            "dataset_variant": state.dataset_variant,
        },
        "scoring": _initial_scoring_metadata(
            policy=state.post_training_scoring_policy,
            requested_stage=state.post_training_requested_stage,
        ),
        "resources": state.resource_policy,
        "cache": state.cache_policy,
    }
    state.run_log_path = resolve_run_log_path(output_root)
    state.artifacts = {
        "runtime": str(cast(Path, state.runtime_path).relative_to(output_root)),
        "resolved_config": str(state.resolved_snapshot_path.relative_to(output_root)),
        "log": str(state.run_log_path.relative_to(output_root)),
    }
    return state


def load_training_data(state: TrainingPipelineState) -> TrainingPipelineState:
    """Load feature metadata and prepare training data providers."""
    _raise_if_cancel_requested(state.telemetry_session, stage_name="load_data")
    _record_telemetry_stage(
        state.telemetry_session,
        completed_stages=state.completed_stages,
        stage_name="load_data",
        message="Loading training datasets and feature metadata.",
        run_log_path=state.run_log_path,
        attempt_id=state.run_attempt_id,
        extra_payload=_stage_progress_payload(state, stage_name="load_data"),
    )

    state.features = load_features(
        cast(TrainingDataClient, state.training_client),
        state.data_version,
        state.feature_set,
        dataset_variant=state.dataset_variant,
    )
    state.full = load_full_data(
        cast(TrainingDataClient, state.training_client),
        state.data_version,
        state.dataset_variant,
        state.features,
        state.era_col,
        state.target_col,
        state.id_col,
        dataset_scope=state.dataset_scope,
    )
    if state.nan_missing_all_twos:
        state.full = apply_missing_all_twos_as_nan(
            state.full,
            state.features,
            state.era_col,
            state.missing_value,
        )
    state.all_eras = sorted(set(state.full[state.era_col].tolist()), key=_era_sort_key)
    state.full_rows = int(state.full.shape[0])
    state.full_eras = int(state.full[state.era_col].nunique())

    if "baseline" in state.x_groups:
        baseline_spec = _as_dict(state.model_config.get("baseline"))
        baseline_name = baseline_spec.get("name")
        baseline_path = baseline_spec.get("predictions_path")
        pred_col = str(baseline_spec.get("pred_col", "prediction"))
        if not baseline_name or not baseline_path:
            raise TrainingConfigError("training_baseline_config_missing_name_or_predictions_path")
        if not state.id_col:
            raise TrainingConfigError("training_id_col_required_for_baseline")
        resolved_baseline_path = _resolve_baseline_path(str(baseline_path), cast(Path, state.baselines_dir))
        state.baseline_name = str(baseline_name)
        state.baseline_predictions_path = str(resolved_baseline_path)
        state.baseline_pred_col = pred_col
        state.baseline, state.baseline_col = load_custom_benchmark_predictions(
            resolved_baseline_path,
            str(baseline_name),
            pred_col=pred_col,
            era_col=state.era_col,
            id_col=state.id_col,
        )
        if state.baseline_col is None:
            raise TrainingConfigError("training_baseline_column_missing")
        if state.full is None:
            raise TrainingConfigError("training_data_loading_materialized_missing_frame")
        state.full = attach_benchmark_predictions(
            state.full,
            state.baseline,
            state.baseline_col,
            era_col=state.era_col,
            id_col=state.id_col,
        )

    state.x_cols = build_x_cols(
        x_groups=state.x_groups,
        features=state.features,
        era_col=state.era_col,
        id_col=state.id_col,
        baseline_col=state.baseline_col,
    )

    if state.full is None:
        raise TrainingConfigError("training_data_loading_materialized_missing_frame")
    state.data_loader = build_model_data_loader(
        full=state.full,
        x_cols=state.x_cols,
        era_col=state.era_col,
        target_col=state.target_col,
        id_col=state.id_col,
    )
    return state


def train_model(state: TrainingPipelineState) -> TrainingPipelineState:
    """Build predictions and persist pre-scoring artifacts."""
    _raise_if_cancel_requested(state.telemetry_session, stage_name="train_model")
    if state.config is None:
        raise TrainingError("training_config_uninitialized")
    config = state.config
    engine_plan = state.engine_plan
    if engine_plan is None:
        raise TrainingError("training_engine_plan_uninitialized")
    simple_train_eras: list[object] | None = None
    simple_val_eras: list[object] | None = None
    if engine_plan.mode == _SIMPLE_PROFILE:
        train_path, validation_path = ensure_split_dataset_paths(
            cast(TrainingDataClient, state.training_client),
            state.data_version,
            dataset_variant=state.dataset_variant,
            data_root=DEFAULT_DATASETS_DIR,
        )
        simple_train_eras = list_lazy_source_eras((train_path,), era_col=state.era_col, include_validation_only=False)
        simple_val_eras = list_lazy_source_eras((validation_path,), era_col=state.era_col, include_validation_only=True)
        if not simple_train_eras or not simple_val_eras:
            raise TrainingConfigError("training_profile_simple_requires_nonempty_split_eras")

    cv_config = dict(engine_plan.cv_config)
    if engine_plan.mode == _SIMPLE_PROFILE:
        cv_config["embargo"] = 0
        cv_config["train_eras"] = simple_train_eras
        cv_config["val_eras"] = simple_val_eras
    elif state.data_embargo_eras is not None:
        cv_config.setdefault("embargo", state.data_embargo_eras)
    state.cv_enabled = bool(cv_config.get("enabled", True))
    state.train_model_progress_total = _resolve_train_model_progress_total(state=state, cv_config=cv_config)
    state.train_model_progress_current = 0

    _record_telemetry_stage(
        state.telemetry_session,
        completed_stages=state.completed_stages,
        stage_name="train_model",
        message="Building model predictions.",
        run_log_path=state.run_log_path,
        attempt_id=state.run_attempt_id,
        extra_payload=_stage_progress_payload(state, stage_name="train_model"),
    )
    on_fold_complete = None
    on_fold_start = None
    if (
        state.scoring_dir is not None
        and state.training_client is not None
        and state.benchmark_source is not None
        and not _is_full_history_refit_profile(engine_plan.mode)
    ):

        def _format_val_interval(fold_metadata: dict[str, object]) -> str:
            val_interval = fold_metadata.get("val_interval")
            if not isinstance(val_interval, dict):
                return "unknown"
            start = val_interval.get("start")
            end = val_interval.get("end")
            return f"{start}->{end}"

        def _on_fold_start(fold_metadata: dict[str, object]) -> None:
            _raise_if_cancel_requested(state.telemetry_session, stage_name="train_model.fold_start")
            _emit_train_model_progress(state)
            log_info(
                state.run_log_path,
                event="fold_started",
                message=(
                    f"fold={fold_metadata.get('fold')} "
                    f"train_eras={fold_metadata.get('train_eras')} "
                    f"val_eras={fold_metadata.get('val_eras')} "
                    f"val_interval={_format_val_interval(fold_metadata)}"
                ),
                attempt_id=state.run_attempt_id,
            )

        def _on_fold_complete(fold_predictions: pd.DataFrame, fold_metadata: dict[str, object]) -> None:
            _raise_if_cancel_requested(state.telemetry_session, stage_name="train_model.fold_complete")
            state.train_model_progress_current += 1
            _emit_train_model_progress(state)
            if state.benchmark_source is None or state.training_client is None or state.scoring_dir is None:
                return
            try:
                append_post_fold_artifacts(
                    predictions=fold_predictions,
                    run_id=state.run_id,
                    config_hash=state.config_hash,
                    seed=None,
                    target_col=state.target_col,
                    benchmark_source=state.benchmark_source,
                    client=state.training_client,
                    data_version=state.data_version,
                    dataset_variant=state.dataset_variant,
                    feature_source_paths=state.scoring_feature_source_paths,
                    dataset_scope=state.dataset_scope,
                    era_col=state.era_col,
                    id_col=state.id_col,
                    data_root=DEFAULT_DATASETS_DIR,
                    scoring_dir=state.scoring_dir,
                )
            except Exception as exc:
                log_error(
                    state.run_log_path,
                    event="post_fold_scoring_append_failed",
                    message=str(exc),
                    attempt_id=state.run_attempt_id,
                )
            log_info(
                state.run_log_path,
                event="fold_completed",
                message=(
                    f"fold={fold_metadata.get('fold')} "
                    f"train_rows={fold_metadata.get('train_rows')} "
                    f"val_rows={fold_metadata.get('val_rows')} "
                    f"val_interval={_format_val_interval(fold_metadata)} "
                    f"prediction_rows={len(fold_predictions)}"
                ),
                attempt_id=state.run_attempt_id,
            )

        on_fold_complete = _on_fold_complete
        on_fold_start = _on_fold_start

    if _is_full_history_refit_profile(engine_plan.mode):
        # Preserve the long-standing service-module monkeypatch seam used by
        # training service tests. Real runs always pass a callable or loader
        # protocol here and keep the fitted model for artifact persistence.
        if isinstance(state.data_loader, ModelDataLoaderProtocol) or callable(state.data_loader):
            full_history_result = fit_full_history_model(
                eras=state.all_eras,
                data_loader=cast(ModelDataLoaderProtocol, state.data_loader),
                model_type=state.model_type,
                model_params=state.model_params,
                model_config=state.model_config,
                id_col=state.id_col,
                era_col=state.era_col,
                target_col=state.target_col,
                store_root=state.run_store_root or _resolve_store_root_for_run(state.output_root),
                feature_cols=state.features,
            )
            state.predictions = full_history_result.predictions
            state.cv_meta = full_history_result.meta
            state.fitted_model = full_history_result.model
        else:
            state.predictions, state.cv_meta = build_full_history_predictions(
                state.all_eras,
                state.data_loader,
                state.model_type,
                state.model_params,
                state.model_config,
                state.id_col,
                state.era_col,
                state.target_col,
                store_root=state.run_store_root or _resolve_store_root_for_run(state.output_root),
                feature_cols=state.features,
            )
            state.fitted_model = None
    else:
        if not state.cv_enabled:
            raise TrainingConfigError("training_cv_required")
        state.predictions, state.cv_meta = build_oof_predictions(
            state.all_eras,
            cast(ModelDataLoaderProtocol, state.data_loader),
            state.model_type,
            state.model_params,
            state.model_config,
            cv_config,
            state.id_col,
            state.era_col,
            state.target_col,
            store_root=state.run_store_root or _resolve_store_root_for_run(state.output_root),
            feature_cols=state.features,
            parallel_folds=cast(int, state.resource_policy["parallel_folds"]),
            parallel_backend=str(state.resource_policy["parallel_backend"]),
            memmap_enabled=bool(state.resource_policy["memmap_enabled"]),
            on_fold_start=on_fold_start,
            on_fold_complete=on_fold_complete,
        )

    state.effective_embargo_eras = _coerce_int(
        state.cv_meta.get("embargo"),
        default=0 if state.data_embargo_eras is None else state.data_embargo_eras,
    )
    state.predictions = select_prediction_columns(state.predictions, state.id_col, state.era_col, state.target_col)
    if state.predictions_dir is None or state.output_root is None:
        raise TrainingError("training_output_paths_uninitialized")
    state.predictions_path, state.predictions_relative = save_predictions(
        state.predictions,
        config,
        state.config_path,
        state.predictions_dir,
        state.output_root,
    )
    state.artifacts["predictions"] = str(state.predictions_relative)
    state.oof_rows = int(state.predictions.shape[0])
    state.oof_eras = int(state.predictions[state.era_col].nunique())
    state.scoring_feature_source_paths = None
    if _is_full_history_refit_profile(engine_plan.mode) and state.fitted_model is not None:
        _persist_full_history_model_artifact(state)

    _raise_if_cancel_requested(state.telemetry_session, stage_name="score_predictions")
    _record_telemetry_stage(
        state.telemetry_session,
        completed_stages=state.completed_stages,
        stage_name="score_predictions",
        message="Deferring scoring to post-run metrics phase.",
        run_log_path=state.run_log_path,
        attempt_id=state.run_attempt_id,
        extra_payload=_stage_progress_payload(state, stage_name="score_predictions"),
    )
    state.metrics_status = _initial_metrics_status(state.post_training_scoring_policy)
    state.training_runtime_metadata["scoring"] = _initial_scoring_metadata(
        policy=state.post_training_scoring_policy,
        requested_stage=state.post_training_requested_stage,
    )

    _raise_if_cancel_requested(state.telemetry_session, stage_name="persist_artifacts")
    _record_telemetry_stage(
        state.telemetry_session,
        completed_stages=state.completed_stages,
        stage_name="persist_artifacts",
        message="Persisting run artifacts and metrics.",
        run_log_path=state.run_log_path,
        attempt_id=state.run_attempt_id,
        extra_payload=_stage_progress_payload(state, stage_name="persist_artifacts"),
    )
    if (
        state.results_dir is None
        or state.predictions_relative is None
        or state.metrics_path is None
        or state.benchmark_source is None
    ):
        raise TrainingError("training_artifact_paths_uninitialized")
    state.results_path = resolve_results_path(config, state.config_path, state.results_dir)
    results = build_results_payload(
        model_type=state.model_type,
        model_params=state.model_params,
        model_config=state.model_config,
        nan_missing_all_twos=state.nan_missing_all_twos,
        missing_value=state.missing_value,
        data_version=state.data_version,
        dataset_variant=state.dataset_variant,
        feature_set=state.feature_set,
        target_col=state.target_col,
        dataset_scope=state.dataset_scope,
        full_rows=state.full_rows,
        full_eras=state.full_eras,
        oof_rows=state.oof_rows,
        oof_eras=state.oof_eras,
        configured_embargo_eras=state.data_embargo_eras,
        effective_embargo_eras=state.effective_embargo_eras,
        benchmark_source=state.benchmark_source,
        meta_model_col=state.meta_model_col,
        meta_model_data_path=state.meta_model_data_path,
        output_dir=state.output_root,
        predictions_relative=state.predictions_relative,
        score_provenance_relative=state.score_provenance_relative,
        summaries=None,
        cv_meta=state.cv_meta,
        engine_plan=engine_plan,
        cv_enabled=state.cv_enabled,
        resource_policy=state.resource_policy,
        cache_policy=state.cache_policy,
        scoring_metadata=cast(dict[str, object], state.training_runtime_metadata["scoring"]),
        metrics_status=state.metrics_status,
    )
    if state.results_path is None:
        raise TrainingError("training_results_path_uninitialized")
    save_results(results, state.results_path)
    state.artifacts["results"] = str(state.results_path.relative_to(state.output_root))
    state.metrics_payload = _extract_metrics_payload(results)
    save_metrics(state.metrics_payload, state.metrics_path)
    state.artifacts["metrics"] = str(state.metrics_path.relative_to(state.output_root))
    _record_telemetry_stage(
        state.telemetry_session,
        completed_stages=state.completed_stages,
        stage_name="index_run",
        message="Indexing run artifacts in store.",
        run_log_path=state.run_log_path,
        attempt_id=state.run_attempt_id,
        extra_payload=_stage_progress_payload(state, stage_name="index_run"),
    )
    state.run_store_root = _resolve_store_root_for_run(state.output_root)
    try:
        index_run(store_root=state.run_store_root, run_id=state.run_id)
    except StoreError as exc:
        raise TrainingError(f"training_store_index_failed:{state.run_id}") from exc
    return state


def score_predictions(state: TrainingPipelineState) -> TrainingPipelineState:
    """Compute post-run scoring artifacts for the persisted predictions."""
    _raise_if_cancel_requested(state.telemetry_session, stage_name="score_predictions_post_run")
    engine_plan = state.engine_plan
    if engine_plan is None:
        raise TrainingError("training_engine_plan_uninitialized")
    if (
        state.output_root is None
        or state.predictions_path is None
        or state.training_client is None
        or state.benchmark_source is None
    ):
        raise TrainingError("training_scoring_paths_uninitialized")

    _record_telemetry_stage(
        state.telemetry_session,
        completed_stages=state.completed_stages,
        stage_name="score_predictions_post_run",
        message="Computing metrics in post-run phase.",
        run_log_path=state.run_log_path,
        attempt_id=state.run_attempt_id,
        extra_payload=_stage_progress_payload(state, stage_name="score_predictions_post_run"),
    )
    state.full = None
    state.baseline = None
    state.all_eras = []
    state.data_loader = None
    state.predictions = pd.DataFrame()
    gc.collect()

    requested_stage = state.post_training_requested_stage
    if requested_stage is None:
        log_info(
            state.run_log_path,
            event="post_run_scoring_skipped",
            message="policy=none",
            attempt_id=state.run_attempt_id,
        )
        return state
    if is_round_post_training_scoring_policy(state.post_training_scoring_policy):
        log_info(
            state.run_log_path,
            event="post_run_scoring_deferred",
            message=f"policy={state.post_training_scoring_policy} stage={requested_stage}",
            attempt_id=state.run_attempt_id,
        )
        return state

    log_info(
        state.run_log_path,
        event="post_run_scoring_started",
        message=f"stage={requested_stage}",
        attempt_id=state.run_attempt_id,
    )

    try:
        scoring_result = run_scoring(
            request=PostTrainingScoringRequest(
                run_id=state.run_id,
                config_hash=state.config_hash,
                seed=None,
                predictions_path=state.predictions_path,
                pred_cols=("prediction",),
                target_col=state.target_col,
                scoring_target_cols=_resolve_scoring_target_cols(
                    data_config=state.data_config,
                    target_col=state.target_col,
                ),
                scoring_targets_explicit=_scoring_targets_explicit(data_config=state.data_config),
                data_version=state.data_version,
                dataset_variant=state.dataset_variant,
                feature_set=state.feature_set,
                feature_source_paths=state.scoring_feature_source_paths,
                dataset_scope=state.dataset_scope,
                benchmark_source=state.benchmark_source,
                meta_model_col=state.meta_model_col,
                meta_model_data_path=state.meta_model_data_path,
                era_col=state.era_col,
                id_col=state.id_col,
                data_root=DEFAULT_DATASETS_DIR,
                stage=requested_stage,
            ),
            client=state.training_client,
        )
    except Exception as exc:
        logger.exception("post-run scoring failed for run_id=%s", state.run_id)
        log_error(
            state.run_log_path,
            event="post_run_scoring_failed",
            message=str(exc),
            attempt_id=state.run_attempt_id,
        )
        state.metrics_status = {
            "status": "failed",
            "reason": "post_training_scoring_failed",
            "error": str(exc),
        }
        state.training_runtime_metadata["scoring"] = _failed_scoring_metadata(
            policy=state.post_training_scoring_policy,
            requested_stage=requested_stage,
            error=str(exc),
            reason="post_training_scoring_failed",
        )
        _persist_results_with_current_scoring_state(state)
        return state

    state.summaries = scoring_result.summaries
    score_provenance = scoring_result.score_provenance
    stage_metadata = _as_dict(score_provenance.get("stages"))
    if (
        state.score_provenance_path is None
        or state.scoring_dir is None
        or state.results_path is None
        or state.predictions_relative is None
        or state.metrics_path is None
    ):
        raise TrainingError("training_scoring_paths_uninitialized")
    persisted_scoring = save_scoring_artifacts(
        scoring_result.artifacts,
        scoring_dir=state.scoring_dir,
        output_dir=state.output_root,
        selected_stage=scoring_result.requested_stage,
    )
    manifest_payload = _refresh_persisted_scoring_provenance(
        score_provenance=score_provenance,
        persisted_scoring=persisted_scoring,
    )
    state.training_runtime_metadata["scoring"] = _scoring_metadata_from_manifest(
        policy=state.post_training_scoring_policy,
        status="succeeded",
        requested_stage=scoring_result.requested_stage,
        refreshed_stages=scoring_result.refreshed_stages,
        stage_metadata=stage_metadata,
        manifest_payload=manifest_payload,
    )
    save_score_provenance(score_provenance, state.score_provenance_path)
    state.score_provenance_relative = state.score_provenance_path.relative_to(state.output_root)
    state.artifacts["score_provenance"] = str(state.score_provenance_relative)
    state.artifacts["scoring_manifest"] = str(persisted_scoring.manifest_relative)
    state.metrics_status = None
    log_info(
        state.run_log_path,
        event="post_run_scoring_succeeded",
        message=(
            f"requested_stage={scoring_result.requested_stage} "
            f"refreshed_stages={','.join(scoring_result.refreshed_stages)}"
        ),
        attempt_id=state.run_attempt_id,
    )

    results = build_results_payload(
        model_type=state.model_type,
        model_params=state.model_params,
        model_config=state.model_config,
        nan_missing_all_twos=state.nan_missing_all_twos,
        missing_value=state.missing_value,
        data_version=state.data_version,
        dataset_variant=state.dataset_variant,
        feature_set=state.feature_set,
        target_col=state.target_col,
        dataset_scope=state.dataset_scope,
        full_rows=state.full_rows,
        full_eras=state.full_eras,
        oof_rows=state.oof_rows,
        oof_eras=state.oof_eras,
        configured_embargo_eras=state.data_embargo_eras,
        effective_embargo_eras=state.effective_embargo_eras,
        benchmark_source=state.benchmark_source,
        meta_model_col=state.meta_model_col,
        meta_model_data_path=state.meta_model_data_path,
        output_dir=state.output_root,
        predictions_relative=state.predictions_relative,
        score_provenance_relative=state.score_provenance_relative,
        summaries=state.summaries,
        cv_meta=state.cv_meta,
        engine_plan=engine_plan,
        cv_enabled=state.cv_enabled,
        resource_policy=state.resource_policy,
        cache_policy=state.cache_policy,
        scoring_metadata=cast(dict[str, object], state.training_runtime_metadata["scoring"]),
        metrics_status=None,
    )
    save_results(results, state.results_path)
    state.metrics_payload = _extract_metrics_payload(results)
    save_metrics(state.metrics_payload, state.metrics_path)
    _record_telemetry_metrics(state.telemetry_session, state.metrics_payload)
    return state


def _persist_results_with_current_scoring_state(state: TrainingPipelineState) -> None:
    engine_plan = state.engine_plan
    if engine_plan is None:
        raise TrainingError("training_engine_plan_uninitialized")
    if (
        state.results_path is None
        or state.output_root is None
        or state.predictions_relative is None
        or state.metrics_path is None
        or state.benchmark_source is None
    ):
        raise TrainingError("training_scoring_paths_uninitialized")

    results = build_results_payload(
        model_type=state.model_type,
        model_params=state.model_params,
        model_config=state.model_config,
        nan_missing_all_twos=state.nan_missing_all_twos,
        missing_value=state.missing_value,
        data_version=state.data_version,
        dataset_variant=state.dataset_variant,
        feature_set=state.feature_set,
        target_col=state.target_col,
        dataset_scope=state.dataset_scope,
        full_rows=state.full_rows,
        full_eras=state.full_eras,
        oof_rows=state.oof_rows,
        oof_eras=state.oof_eras,
        configured_embargo_eras=state.data_embargo_eras,
        effective_embargo_eras=state.effective_embargo_eras,
        benchmark_source=state.benchmark_source,
        meta_model_col=state.meta_model_col,
        meta_model_data_path=state.meta_model_data_path,
        output_dir=state.output_root,
        predictions_relative=state.predictions_relative,
        score_provenance_relative=state.score_provenance_relative,
        summaries=None,
        cv_meta=state.cv_meta,
        engine_plan=engine_plan,
        cv_enabled=state.cv_enabled,
        resource_policy=state.resource_policy,
        cache_policy=state.cache_policy,
        scoring_metadata=cast(dict[str, object], state.training_runtime_metadata["scoring"]),
        metrics_status=state.metrics_status,
    )
    save_results(results, state.results_path)
    state.metrics_payload = _extract_metrics_payload(results)
    save_metrics(state.metrics_payload, state.metrics_path)


def _persist_full_history_model_artifact(state: TrainingPipelineState) -> None:
    if state.output_root is None:
        raise TrainingError("training_output_paths_uninitialized")
    if state.fitted_model is None:
        raise TrainingError("training_model_artifact_missing_fitted_model")
    if not state.id_col:
        raise TrainingError("training_model_artifact_missing_id_col")
    model_upload_compatible = (
        state.baseline_predictions_path is None
        and not bool(state.model_config.get("module_path"))
        and state.model_type == "LGBMRegressor"
    )
    manifest = ModelArtifactManifest(
        run_id=state.run_id,
        model_type=state.model_type,
        data_version=state.data_version,
        dataset_variant=state.dataset_variant,
        feature_set=state.feature_set,
        target_col=state.target_col,
        era_col=state.era_col,
        id_col=state.id_col,
        feature_cols=tuple(state.x_cols),
        baseline_col=state.baseline_col,
        baseline_name=state.baseline_name,
        baseline_predictions_path=state.baseline_predictions_path,
        baseline_pred_col=state.baseline_pred_col,
        model_upload_compatible=model_upload_compatible,
        uses_custom_module=bool(state.model_config.get("module_path")) or state.model_type != "LGBMRegressor",
    )
    try:
        artifact_path, manifest_path = save_model_artifact(
            run_dir=state.output_root,
            model=state.fitted_model,
            manifest=manifest,
        )
    except ModelArtifactError as exc:
        raise TrainingError(str(exc)) from exc
    state.model_artifact_path = artifact_path
    state.model_manifest_path = manifest_path
    state.artifacts["model_artifact"] = str(artifact_path.relative_to(state.output_root))
    state.artifacts["model_manifest"] = str(manifest_path.relative_to(state.output_root))


def finalize_training_run(state: TrainingPipelineState) -> TrainingRunResult:
    """Persist final manifest state, refresh index, and return run outputs."""
    _raise_if_cancel_requested(state.telemetry_session, stage_name="finalize_manifest")
    if state.config is None:
        raise TrainingError("training_config_uninitialized")
    config = state.config
    engine_plan = state.engine_plan
    if engine_plan is None:
        raise TrainingError("training_engine_plan_uninitialized")
    state.full = None
    state.baseline = None
    state.all_eras = []
    state.data_loader = None
    state.predictions = pd.DataFrame()
    gc.collect()

    _record_telemetry_stage(
        state.telemetry_session,
        completed_stages=state.completed_stages,
        stage_name="finalize_manifest",
        message="Writing final run manifest.",
        run_log_path=state.run_log_path,
        attempt_id=state.run_attempt_id,
        extra_payload=_stage_progress_payload(state, stage_name="finalize_manifest"),
    )
    finished_manifest = build_run_manifest(
        run_id=state.run_id,
        run_hash=state.run_hash,
        status="FINISHED",
        config_path=state.config_path,
        config_hash=state.config_hash,
        data_version=state.data_version,
        feature_set=state.feature_set,
        target_col=state.target_col,
        model_type=state.model_type,
        engine_mode=engine_plan.mode,
        experiment_id=state.experiment_id,
        created_at=state.created_at,
        artifacts=state.artifacts,
        metrics_summary=state.metrics_payload,
        training_metadata=state.training_runtime_metadata,
        lifecycle_metadata=_lifecycle_manifest_payload(terminal_reason="completed"),
        execution=state.run_execution,
    )
    if state.run_manifest_path is None or state.run_store_root is None or state.output_root is None:
        raise TrainingError("training_finalize_paths_uninitialized")
    save_run_manifest(finished_manifest, state.run_manifest_path)
    try:
        index_run(store_root=state.run_store_root, run_id=state.run_id)
    except StoreError:
        pass
    _ = maybe_log_training_run(
        run_id=state.run_id,
        config=config,
        metrics_payload=state.metrics_payload,
        artifacts=state.artifacts,
        output_root=state.output_root,
    )
    _mark_telemetry_completed(
        state.telemetry_session,
        run_id=state.run_id,
        run_dir=state.output_root,
        run_log_path=state.run_log_path,
        attempt_id=state.run_attempt_id,
    )
    return TrainingRunResult(
        run_id=state.run_id,
        predictions_path=cast(Path, state.predictions_path),
        results_path=cast(Path, state.results_path),
        model_artifact_path=state.model_artifact_path,
        model_manifest_path=state.model_manifest_path,
    )


def fail_training_run(state: TrainingPipelineState, exc: Exception) -> None:
    """Persist failure manifest and telemetry for an interrupted run."""
    engine_plan = state.engine_plan
    if state.run_manifest_written:
        is_canceled = isinstance(exc, TrainingCanceledError)
        cancel_requested_at: str | None = None
        terminal_reason = "failed"
        terminal_detail: dict[str, object] | None = None
        if state.telemetry_session is not None:
            store_root_for_lookup = state.run_store_root
            if store_root_for_lookup is None and state.output_root is not None:
                store_root_for_lookup = _resolve_store_root_for_run(state.output_root)
            lifecycle_record = (
                get_run_lifecycle(store_root=store_root_for_lookup, run_id=state.run_id)
                if store_root_for_lookup is not None
                else None
            )
            if lifecycle_record is not None:
                cancel_requested_at = lifecycle_record.cancel_requested_at
        if is_canceled:
            status = "CANCELED"
            error_payload_value = None
            terminal_reason = "cancel_requested"
            terminal_detail = {"message": str(exc)}
        else:
            status = "FAILED"
            error_payload_value = error_payload(exc)
            terminal_detail = {"message": str(exc), "error": error_payload_value}
        failed_manifest = build_run_manifest(
            run_id=state.run_id,
            run_hash=state.run_hash,
            status=status,
            config_path=state.config_path,
            config_hash=state.config_hash,
            data_version=state.data_version,
            feature_set=state.feature_set,
            target_col=state.target_col,
            model_type=state.model_type,
            engine_mode=engine_plan.mode if engine_plan is not None else state.engine_mode or "unknown",
            experiment_id=state.experiment_id,
            created_at=state.created_at,
            artifacts=state.artifacts,
            error=error_payload_value,
            training_metadata=state.training_runtime_metadata,
            lifecycle_metadata=_lifecycle_manifest_payload(
                terminal_reason=terminal_reason,
                cancel_requested_at=cancel_requested_at,
                terminal_detail=terminal_detail,
            ),
            execution=state.run_execution,
        )
        try:
            save_run_manifest(failed_manifest, cast(Path, state.run_manifest_path))
        except Exception:
            pass
        if state.run_store_root is not None:
            try:
                index_run(store_root=state.run_store_root, run_id=state.run_id)
            except Exception:
                pass
        if is_canceled:
            _mark_telemetry_canceled(
                state.telemetry_session,
                run_id=state.run_id,
                message=str(exc),
                run_log_path=state.run_log_path,
                attempt_id=state.run_attempt_id,
                cancel_requested_at=cancel_requested_at,
                terminal_reason=terminal_reason,
                terminal_detail=terminal_detail,
            )
        else:
            _mark_telemetry_failed(
                state.telemetry_session,
                run_id=state.run_id,
                error=error_payload(exc),
                message=str(exc),
                run_log_path=state.run_log_path,
                attempt_id=state.run_attempt_id,
                terminal_reason=terminal_reason,
                terminal_detail=terminal_detail,
            )


def cleanup_training_run(state: TrainingPipelineState) -> None:
    """Release background samplers and run lock."""
    try:
        stop_local_resource_sampler(state.telemetry_sampler)
    finally:
        release_run_lock(state.run_lock)


def run_training_pipeline(
    *,
    config_path: str | Path,
    output_dir: str | Path | None = None,
    client: TrainingDataClient | None = None,
    profile: TrainingProfile | None = None,
    post_training_scoring: PostTrainingScoringPolicy | None = None,
    engine_mode: str | None = None,
    window_size_eras: int | None = None,
    embargo_eras: int | None = None,
    experiment_id: str | None = None,
    allow_round_batch_post_training_scoring: bool = False,
) -> TrainingRunResult:
    """Execute the full local train pipeline and persist canonical scoring artifacts."""
    state = TrainingPipelineState(
        config_path=Path(config_path).expanduser().resolve(),
        output_dir=Path(output_dir).expanduser() if output_dir is not None else None,
        client=client,
        profile=profile,
        post_training_scoring=post_training_scoring,
        engine_mode=engine_mode,
        window_size_eras=window_size_eras,
        embargo_eras=embargo_eras,
        experiment_id=experiment_id,
        allow_round_batch_post_training_scoring=allow_round_batch_post_training_scoring,
    )
    try:
        state = prepare_training_run(
            config_path=config_path,
            output_dir=output_dir,
            client=client,
            profile=profile,
            post_training_scoring=post_training_scoring,
            engine_mode=engine_mode,
            window_size_eras=window_size_eras,
            embargo_eras=embargo_eras,
            experiment_id=experiment_id,
            allow_round_batch_post_training_scoring=allow_round_batch_post_training_scoring,
            state=state,
        )
        state = load_training_data(state)
        state = train_model(state)
        state = score_predictions(state)
        return finalize_training_run(state)
    except Exception as exc:
        fail_training_run(state, exc)
        raise
    finally:
        cleanup_training_run(state)


def _initial_metrics_status(policy: PostTrainingScoringPolicy) -> dict[str, object]:
    if policy == "none":
        return {
            "status": "deferred",
            "reason": "post_training_scoring_disabled",
        }
    if is_round_post_training_scoring_policy(policy):
        return {
            "status": "deferred",
            "reason": "experiment_round_batch_pending",
        }
    return {
        "status": "pending",
        "reason": "post_training_metrics_pending",
    }


def _initial_scoring_metadata(
    *,
    policy: PostTrainingScoringPolicy,
    requested_stage: CanonicalScoringStage | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "policy": policy,
        "status": "pending",
        "requested_stage": requested_stage,
        "refreshed_stages": [],
    }
    if policy == "none":
        payload["status"] = "deferred"
        payload["reason"] = "post_training_scoring_disabled"
    elif is_round_post_training_scoring_policy(policy):
        payload["status"] = "deferred"
        payload["reason"] = "experiment_round_batch_pending"
    return payload


def _failed_scoring_metadata(
    *,
    policy: PostTrainingScoringPolicy,
    requested_stage: CanonicalScoringStage | None,
    error: str,
    reason: str,
) -> dict[str, object]:
    return {
        "policy": policy,
        "status": "failed",
        "requested_stage": requested_stage,
        "refreshed_stages": [],
        "reason": reason,
        "error": error,
    }


def _era_sort_key(era: object) -> int | str:
    if isinstance(era, str) and era.isdigit():
        return int(era)
    if isinstance(era, int):
        return era
    return str(era)


def _scoring_metadata_from_manifest(
    *,
    policy: PostTrainingScoringPolicy,
    status: str,
    requested_stage: str,
    refreshed_stages: tuple[str, ...],
    stage_metadata: dict[str, object],
    manifest_payload: dict[str, object],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "policy": policy,
        "status": status,
        "requested_stage": requested_stage,
        "refreshed_stages": list(refreshed_stages),
    }
    refreshed_stage_files = _as_dict(manifest_payload.get("refreshed_stage_files"))
    emitted = refreshed_stage_files.keys() if refreshed_stage_files else stage_metadata.get("emitted")
    omissions = _as_dict(_as_dict(manifest_payload.get("stages")).get("omissions")) or stage_metadata.get("omissions")
    if isinstance(emitted, list):
        payload["emitted_stage_files"] = _ordered_emitted_stage_files(emitted)
    elif refreshed_stage_files:
        payload["emitted_stage_files"] = _ordered_emitted_stage_files(refreshed_stage_files.keys())
    if isinstance(omissions, dict):
        payload["omissions"] = {str(key): str(value) for key, value in omissions.items()}
    return payload


def _refresh_persisted_scoring_provenance(
    *,
    score_provenance: dict[str, object],
    persisted_scoring: object,
) -> dict[str, object]:
    if not hasattr(persisted_scoring, "manifest_path") or not hasattr(persisted_scoring, "manifest_relative"):
        return {}
    manifest_path = cast(Path, getattr(persisted_scoring, "manifest_path"))
    manifest_relative = cast(Path, getattr(persisted_scoring, "manifest_relative"))
    try:
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(manifest_payload, dict):
        return {}
    chart_files = _as_dict(manifest_payload.get("chart_files"))
    stage_files = _as_dict(manifest_payload.get("stage_files"))
    refreshed = manifest_payload.get("refreshed_canonical_stages")
    refreshed_stages = [str(value) for value in refreshed] if isinstance(refreshed, list) else []
    score_provenance["artifacts"] = {
        "scoring_manifest": str(manifest_relative),
        "charts": sorted(chart_files.keys()),
        "stage_files": sorted(stage_files.keys()),
        "requested_stage": manifest_payload.get("requested_stage"),
        "refreshed_canonical_stages": refreshed_stages,
    }
    return manifest_payload


def _ordered_emitted_stage_files(values: object) -> list[str]:
    if not isinstance(values, list) and not hasattr(values, "__iter__"):
        return []
    resolved = [str(value) for value in values]
    remaining = set(resolved)
    ordered: list[str] = []
    for item in _EMITTED_STAGE_FILE_ORDER:
        if item in remaining:
            ordered.append(item)
            remaining.remove(item)
    ordered.extend(sorted(remaining))
    return ordered


__all__ = [
    "TrainingPipelineState",
    "cleanup_training_run",
    "fail_training_run",
    "finalize_training_run",
    "load_training_data",
    "prepare_training_run",
    "run_training_pipeline",
    "score_predictions",
    "train_model",
]
