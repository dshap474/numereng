"""Internal staged orchestration for local training runs."""

from __future__ import annotations

import gc
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pandas as pd

from numereng.features.scoring.metrics import attach_benchmark_predictions, load_custom_benchmark_predictions
from numereng.features.scoring.models import PostTrainingScoringRequest, default_scoring_policy
from numereng.features.scoring.service import run_post_training_scoring
from numereng.features.store import StoreError, index_run
from numereng.features.telemetry import (
    LocalResourceSampler,
    LocalRunTelemetrySession,
    begin_local_training_session,
    get_launch_metadata,
    mark_job_running,
    mark_job_starting,
    start_local_resource_sampler,
    stop_local_resource_sampler,
)
from numereng.features.training.client import TrainingDataClient, create_training_data_client
from numereng.features.training.cv import build_full_history_predictions, build_oof_predictions
from numereng.features.training.errors import TrainingConfigError, TrainingError
from numereng.features.training.mlflow_tracking import maybe_log_training_run
from numereng.features.training.models import (
    FoldJoinColumn,
    ModelDataLoaderProtocol,
    TrainingRunResult,
    build_lazy_parquet_data_loader,
    build_model_data_loader,
    build_x_cols,
    normalize_x_groups,
)
from numereng.features.training.repo import (
    DEFAULT_BENCHMARK_MODEL,
    DEFAULT_DATASETS_DIR,
    apply_missing_all_twos_as_nan,
    ensure_split_dataset_paths,
    list_lazy_source_eras,
    load_config,
    load_features,
    load_fold_data_lazy,
    load_full_data,
    resolve_fold_lazy_source_paths,
    resolve_metrics_path,
    resolve_output_locations,
    resolve_resolved_config_path,
    resolve_results_path,
    resolve_run_manifest_path,
    resolve_score_provenance_path,
    save_metrics,
    save_per_era_corr_artifacts,
    save_predictions,
    save_resolved_config,
    save_results,
    save_run_manifest,
    save_score_provenance,
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
    _FOLD_LAZY_LOADING_MODE,
    _MATERIALIZED_LOADING_MODE,
    _SIMPLE_PROFILE,
    _apply_resource_policy,
    _as_dict,
    _coerce_float,
    _coerce_int,
    _coerce_optional_int,
    _ensure_run_dir_is_fresh,
    _extract_metrics_payload,
    _is_full_history_refit_profile,
    _mark_telemetry_completed,
    _mark_telemetry_failed,
    _optional_path,
    _record_telemetry_metrics,
    _record_telemetry_stage,
    _resolve_baseline_path,
    _resolve_cache_policy,
    _resolve_dataset_scope,
    _resolve_dataset_scope_for_profile,
    _resolve_dataset_variant,
    _resolve_loading_mode,
    _resolve_resource_policy,
    _resolve_scoring_mode,
    _resolve_scoring_target_cols,
    _resolve_store_root_for_run,
    _scoring_policy_payload,
    build_results_payload,
    resolve_model_config,
)
from numereng.features.training.strategies import TrainingProfile, resolve_training_engine

logger = logging.getLogger(__name__)


@dataclass
class TrainingPipelineState:
    """Mutable state shared across staged local training execution."""

    config_path: Path
    output_dir: Path | None
    client: TrainingDataClient | None
    profile: TrainingProfile | None
    engine_mode: str | None
    window_size_eras: int | None
    embargo_eras: int | None
    experiment_id: str | None
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
    benchmark_data_path: str | Path | None = None
    meta_model_data_path: str | Path | None = None
    meta_model_col: str = "numerai_meta_model"
    data_embargo_eras: int | None = None
    benchmark_model: str = DEFAULT_BENCHMARK_MODEL
    dataset_scope: str = "train_only"
    loading_mode: str = _MATERIALIZED_LOADING_MODE
    scoring_mode: str = "materialized"
    scoring_era_chunk_size: int = 64
    include_feature_neutral_metrics: bool = True
    nan_missing_all_twos: bool = False
    missing_value: float = 2.0
    resource_policy: dict[str, object] = field(default_factory=dict)
    cache_policy: dict[str, object] = field(default_factory=dict)
    engine_plan: object | None = None
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
    run_manifest_path: Path | None = None
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
    source_paths: tuple[Path, ...] | None = None
    profile_frame: pd.DataFrame | None = None
    lazy_include_validation_only: bool = False
    baseline: pd.DataFrame | None = None
    baseline_col: str | None = None
    join_columns: list[FoldJoinColumn] = field(default_factory=list)
    x_cols: list[str] = field(default_factory=list)
    data_loader: ModelDataLoaderProtocol | None = None
    all_eras: list[object] = field(default_factory=list)
    full_rows: int = 0
    full_eras: int = 0
    cv_enabled: bool = True
    cv_meta: dict[str, object] = field(default_factory=dict)
    effective_embargo_eras: int = 0
    predictions: pd.DataFrame | None = None
    oof_rows: int = 0
    oof_eras: int = 0
    scoring_feature_source_paths: tuple[Path, ...] | None = None
    summaries: dict[str, pd.DataFrame] | None = None
    metrics_status: dict[str, object] | None = None
    effective_scoring_backend: str = ""
    metrics_payload: dict[str, object] = field(default_factory=dict)


def prepare_training_run(
    *,
    config_path: str | Path,
    output_dir: str | Path | None = None,
    client: TrainingDataClient | None = None,
    profile: TrainingProfile | None = None,
    engine_mode: str | None = None,
    window_size_eras: int | None = None,
    embargo_eras: int | None = None,
    experiment_id: str | None = None,
    state: TrainingPipelineState | None = None,
) -> TrainingPipelineState:
    """Resolve config, run identity, runtime policy, and persistent run scaffolding."""
    if state is None:
        state = TrainingPipelineState(
            config_path=Path(config_path).expanduser().resolve(),
            output_dir=Path(output_dir).expanduser() if output_dir is not None else None,
            client=client,
            profile=profile,
            engine_mode=engine_mode,
            window_size_eras=window_size_eras,
            embargo_eras=embargo_eras,
            experiment_id=experiment_id,
        )

    state.config = load_config(state.config_path)
    state.data_config = _as_dict(state.config.get("data"))
    state.preprocessing_config = _as_dict(state.config.get("preprocessing"))
    state.training_config = _as_dict(state.config.get("training"))
    state.model_config = _as_dict(state.config.get("model"))

    state.data_version = str(state.data_config.get("data_version", "v5.2"))
    state.dataset_variant = _resolve_dataset_variant(state.data_config.get("dataset_variant"))
    state.feature_set = str(state.data_config.get("feature_set", "small"))
    state.target_col = str(state.data_config.get("target_col", "target"))
    state.era_col = str(state.data_config.get("era_col", "era"))
    state.id_col = str(state.data_config.get("id_col", "id"))
    state.benchmark_data_path = _optional_path(state.data_config.get("benchmark_data_path"))
    state.meta_model_data_path = _optional_path(state.data_config.get("meta_model_data_path"))
    state.meta_model_col = str(state.data_config.get("meta_model_col", "numerai_meta_model"))
    state.data_embargo_eras = _coerce_optional_int(state.data_config.get("embargo_eras"))
    state.benchmark_model = str(state.data_config.get("benchmark_model", DEFAULT_BENCHMARK_MODEL))
    dataset_scope_config = _resolve_dataset_scope(state.data_config.get("dataset_scope"))
    loading_config = _as_dict(state.data_config.get("loading"))
    state.loading_mode = _resolve_loading_mode(loading_config.get("mode"))
    state.scoring_mode = _resolve_scoring_mode(loading_config.get("scoring_mode"))
    state.scoring_era_chunk_size = _coerce_int(loading_config.get("era_chunk_size"), default=64)
    state.include_feature_neutral_metrics = bool(loading_config.get("include_feature_neutral_metrics", True))
    if state.scoring_era_chunk_size < 1:
        raise TrainingConfigError("training_data_loading_era_chunk_size_invalid")

    state.nan_missing_all_twos = bool(state.preprocessing_config.get("nan_missing_all_twos", False))
    state.missing_value = _coerce_float(state.preprocessing_config.get("missing_value"), default=2.0)
    state.resource_policy = _resolve_resource_policy(_as_dict(state.training_config.get("resources")))
    state.cache_policy = _resolve_cache_policy(_as_dict(state.training_config.get("cache")))
    _apply_resource_policy(state.resource_policy)

    state.engine_plan = resolve_training_engine(
        training_config=state.training_config,
        data_config=state.data_config,
        profile=state.profile,
        engine_mode=state.engine_mode,
        window_size_eras=state.window_size_eras,
        embargo_eras=state.embargo_eras,
    )
    state.dataset_scope = _resolve_dataset_scope_for_profile(
        profile=cast(TrainingProfile, state.engine_plan.mode),
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
        config=cast(dict[str, object], state.config),
        data_version=state.data_version,
        feature_set=state.feature_set,
        target_col=state.target_col,
        model_type=state.model_type,
        engine_mode=state.engine_plan.mode,
        engine_settings=state.engine_plan.resolved_config,
    )
    state.run_id = run_id_from_hash(state.run_hash)
    state.config_hash = compute_config_hash(cast(dict[str, object], state.config))

    output_root, baselines_dir, results_dir, predictions_dir = resolve_output_locations(
        cast(dict[str, object], state.config),
        state.output_dir,
        state.run_id,
    )
    state.output_root = output_root
    state.baselines_dir = baselines_dir
    state.results_dir = results_dir
    state.predictions_dir = predictions_dir
    state.run_manifest_path = resolve_run_manifest_path(output_root)
    state.resolved_snapshot_path = resolve_resolved_config_path(output_root)
    state.metrics_path = resolve_metrics_path(output_root)
    state.score_provenance_path = resolve_score_provenance_path(output_root)
    state.training_runtime_metadata = {
        "data": {
            "dataset_scope": state.dataset_scope,
            "dataset_variant": state.dataset_variant,
        },
        "loading": {"mode": state.loading_mode},
        "scoring": {
            "mode": state.scoring_mode,
            "era_chunk_size": state.scoring_era_chunk_size,
            "effective_backend": state.scoring_mode,
            "policy": _scoring_policy_payload(default_scoring_policy(state.include_feature_neutral_metrics)),
        },
        "resources": state.resource_policy,
        "cache": state.cache_policy,
    }
    state.run_log_path = resolve_run_log_path(output_root)
    state.artifacts = {
        "resolved_config": str(state.resolved_snapshot_path.relative_to(output_root)),
        "log": str(state.run_log_path.relative_to(output_root)),
    }
    state.run_attempt_id = build_local_attempt_id(state.run_id)
    state.run_lock = acquire_run_lock(
        run_dir=output_root,
        run_id=state.run_id,
        attempt_id=state.run_attempt_id,
    )

    _ensure_run_dir_is_fresh(output_root)
    launch_metadata = get_launch_metadata()
    if launch_metadata is not None:
        try:
            state.run_store_root = _resolve_store_root_for_run(output_root)
            state.telemetry_session = begin_local_training_session(
                store_root=state.run_store_root,
                config_path=state.config_path,
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
            if state.telemetry_session is not None:
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
                )
        except Exception:
            logger.exception("training telemetry bootstrap failed for run_id=%s", state.run_id)

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
        message=(
            f"run_id={state.run_id} engine_mode={state.engine_plan.mode} "
            f"loading_mode={state.loading_mode} scoring_mode={state.scoring_mode}"
        ),
        attempt_id=state.run_attempt_id,
    )
    save_resolved_config(cast(dict[str, object], state.config), state.resolved_snapshot_path)
    state.created_at = str(running_manifest["created_at"])
    state.training_client = create_training_data_client() if state.client is None else state.client
    return state


def load_training_data(state: TrainingPipelineState) -> TrainingPipelineState:
    """Load feature metadata and prepare training data providers."""
    _record_telemetry_stage(
        state.telemetry_session,
        completed_stages=state.completed_stages,
        stage_name="load_data",
        message="Loading training datasets and feature metadata.",
        run_log_path=state.run_log_path,
        attempt_id=state.run_attempt_id,
    )

    state.features = load_features(
        cast(TrainingDataClient, state.training_client),
        state.data_version,
        state.feature_set,
        dataset_variant=state.dataset_variant,
    )
    if state.loading_mode == _MATERIALIZED_LOADING_MODE:
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
    else:
        if state.nan_missing_all_twos:
            raise TrainingConfigError("training_data_loading_fold_lazy_disallows_nan_missing_all_twos")
        state.lazy_include_validation_only = state.dataset_scope == "train_plus_validation"
        state.source_paths = resolve_fold_lazy_source_paths(
            cast(TrainingDataClient, state.training_client),
            state.data_version,
            dataset_variant=state.dataset_variant,
            dataset_scope=state.dataset_scope,
            data_root=DEFAULT_DATASETS_DIR,
        )
        state.all_eras = list_lazy_source_eras(
            state.source_paths,
            era_col=state.era_col,
            include_validation_only=state.lazy_include_validation_only,
        )
        state.profile_frame = load_fold_data_lazy(
            state.source_paths,
            eras=state.all_eras,
            columns=[state.era_col],
            era_col=state.era_col,
            id_col=None,
            include_validation_only=state.lazy_include_validation_only,
        )
        state.all_eras = sorted(set(state.profile_frame[state.era_col].tolist()), key=_era_sort_key)
        state.full_rows = int(state.profile_frame.shape[0])
        state.full_eras = (
            int(state.profile_frame[state.era_col].nunique())
            if state.era_col in state.profile_frame.columns
            else len(state.all_eras)
        )

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
        state.baseline, state.baseline_col = load_custom_benchmark_predictions(
            resolved_baseline_path,
            str(baseline_name),
            pred_col=pred_col,
            era_col=state.era_col,
            id_col=state.id_col,
        )
        if state.loading_mode == _MATERIALIZED_LOADING_MODE:
            if state.full is None:
                raise TrainingConfigError("training_data_loading_materialized_missing_frame")
            state.full = attach_benchmark_predictions(
                state.full,
                state.baseline,
                cast(str, state.baseline_col),
                era_col=state.era_col,
                id_col=state.id_col,
            )
        else:
            state.join_columns.append(FoldJoinColumn(frame=state.baseline, column=cast(str, state.baseline_col)))

    state.x_cols = build_x_cols(
        x_groups=state.x_groups,
        features=state.features,
        era_col=state.era_col,
        id_col=state.id_col,
        baseline_col=state.baseline_col,
    )

    if state.loading_mode == _MATERIALIZED_LOADING_MODE:
        if state.full is None:
            raise TrainingConfigError("training_data_loading_materialized_missing_frame")
        state.data_loader = build_model_data_loader(
            full=state.full,
            x_cols=state.x_cols,
            era_col=state.era_col,
            target_col=state.target_col,
            id_col=state.id_col,
        )
    else:
        if state.source_paths is None:
            raise TrainingConfigError("training_data_loading_fold_lazy_missing_sources")
        state.data_loader = build_lazy_parquet_data_loader(
            source_paths=state.source_paths,
            x_cols=state.x_cols,
            era_col=state.era_col,
            target_col=state.target_col,
            id_col=state.id_col,
            include_validation_only=state.lazy_include_validation_only,
            join_columns=state.join_columns,
        )
    return state


def train_model(state: TrainingPipelineState) -> TrainingPipelineState:
    """Build predictions and persist pre-scoring artifacts."""
    simple_train_eras: list[object] | None = None
    simple_val_eras: list[object] | None = None
    if state.engine_plan.mode == _SIMPLE_PROFILE:
        if (
            state.loading_mode == _FOLD_LAZY_LOADING_MODE
            and state.source_paths is not None
            and len(state.source_paths) >= 2
        ):
            train_path, validation_path = state.source_paths[0], state.source_paths[1]
        else:
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

    cv_config = dict(state.engine_plan.cv_config)
    if state.engine_plan.mode == _SIMPLE_PROFILE:
        cv_config["embargo"] = 0
        cv_config["train_eras"] = simple_train_eras
        cv_config["val_eras"] = simple_val_eras
    elif state.data_embargo_eras is not None:
        cv_config.setdefault("embargo", state.data_embargo_eras)
    state.cv_enabled = bool(cv_config.get("enabled", True))

    _record_telemetry_stage(
        state.telemetry_session,
        completed_stages=state.completed_stages,
        stage_name="train_model",
        message="Building model predictions.",
        run_log_path=state.run_log_path,
        attempt_id=state.run_attempt_id,
    )
    if _is_full_history_refit_profile(state.engine_plan.mode):
        state.predictions, state.cv_meta = build_full_history_predictions(
            state.all_eras,
            cast(ModelDataLoaderProtocol, state.data_loader),
            state.model_type,
            state.model_params,
            state.model_config,
            state.id_col,
            state.era_col,
            state.target_col,
            feature_cols=state.features,
        )
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
            feature_cols=state.features,
            parallel_folds=cast(int, state.resource_policy["parallel_folds"]),
            parallel_backend=str(state.resource_policy["parallel_backend"]),
            memmap_enabled=bool(state.resource_policy["memmap_enabled"]),
        )

    state.effective_embargo_eras = _coerce_int(
        state.cv_meta.get("embargo"),
        default=0 if state.data_embargo_eras is None else state.data_embargo_eras,
    )
    state.predictions = select_prediction_columns(state.predictions, state.id_col, state.era_col, state.target_col)
    state.predictions_path, state.predictions_relative = save_predictions(
        state.predictions,
        cast(dict[str, object], state.config),
        state.config_path,
        cast(Path, state.predictions_dir),
        cast(Path, state.output_root),
    )
    state.artifacts["predictions"] = str(state.predictions_relative)
    state.oof_rows = int(state.predictions.shape[0])
    state.oof_eras = int(state.predictions[state.era_col].nunique())
    state.scoring_feature_source_paths = state.source_paths
    state.effective_scoring_backend = state.scoring_mode

    _record_telemetry_stage(
        state.telemetry_session,
        completed_stages=state.completed_stages,
        stage_name="score_predictions",
        message="Deferring scoring to post-run metrics phase.",
        run_log_path=state.run_log_path,
        attempt_id=state.run_attempt_id,
    )
    if _is_full_history_refit_profile(state.engine_plan.mode):
        state.metrics_status = {
            "status": "not_applicable",
            "reason": "training_profile_full_history_refit_no_validation_metrics",
        }
        state.effective_scoring_backend = "not_applicable"
    else:
        state.metrics_status = {
            "status": "pending",
            "reason": "post_run_metrics_pending",
        }
        state.effective_scoring_backend = "pending"
    state.training_runtime_metadata["scoring"] = {
        "mode": state.scoring_mode,
        "era_chunk_size": state.scoring_era_chunk_size,
        "effective_backend": state.effective_scoring_backend,
        "policy": _scoring_policy_payload(default_scoring_policy()),
    }

    _record_telemetry_stage(
        state.telemetry_session,
        completed_stages=state.completed_stages,
        stage_name="persist_artifacts",
        message="Persisting run artifacts and metrics.",
        run_log_path=state.run_log_path,
        attempt_id=state.run_attempt_id,
    )
    state.results_path = resolve_results_path(
        cast(dict[str, object], state.config),
        state.config_path,
        cast(Path, state.results_dir),
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
        benchmark_model=state.benchmark_model,
        benchmark_data_path=state.benchmark_data_path,
        meta_model_col=state.meta_model_col,
        meta_model_data_path=state.meta_model_data_path,
        output_dir=cast(Path, state.output_root),
        predictions_relative=cast(Path, state.predictions_relative),
        score_provenance_relative=state.score_provenance_relative,
        summaries=None,
        cv_meta=state.cv_meta,
        engine_plan=state.engine_plan,
        cv_enabled=state.cv_enabled,
        loading_mode=state.loading_mode,
        scoring_mode=state.scoring_mode,
        scoring_era_chunk_size=state.scoring_era_chunk_size,
        resource_policy=state.resource_policy,
        cache_policy=state.cache_policy,
        scoring_backend=state.effective_scoring_backend,
        scoring_policy=default_scoring_policy(),
        metrics_status=state.metrics_status,
    )
    save_results(results, cast(Path, state.results_path))
    state.artifacts["results"] = str(cast(Path, state.results_path).relative_to(cast(Path, state.output_root)))
    state.metrics_payload = _extract_metrics_payload(results)
    save_metrics(state.metrics_payload, cast(Path, state.metrics_path))
    state.artifacts["metrics"] = str(cast(Path, state.metrics_path).relative_to(cast(Path, state.output_root)))
    _record_telemetry_stage(
        state.telemetry_session,
        completed_stages=state.completed_stages,
        stage_name="index_run",
        message="Indexing run artifacts in store.",
        run_log_path=state.run_log_path,
        attempt_id=state.run_attempt_id,
    )
    state.run_store_root = _resolve_store_root_for_run(cast(Path, state.output_root))
    try:
        index_run(store_root=state.run_store_root, run_id=state.run_id)
    except StoreError as exc:
        raise TrainingError(f"training_store_index_failed:{state.run_id}") from exc
    return state


def score_predictions(state: TrainingPipelineState) -> TrainingPipelineState:
    """Compute post-run scoring artifacts for the persisted predictions."""
    if _is_full_history_refit_profile(state.engine_plan.mode):
        return state

    _record_telemetry_stage(
        state.telemetry_session,
        completed_stages=state.completed_stages,
        stage_name="score_predictions_post_run",
        message="Computing metrics in post-run phase.",
        run_log_path=state.run_log_path,
        attempt_id=state.run_attempt_id,
    )
    log_info(
        state.run_log_path,
        event="post_run_scoring_started",
        message=f"mode={state.scoring_mode} era_chunk_size={state.scoring_era_chunk_size}",
        attempt_id=state.run_attempt_id,
    )
    state.full = None
    state.source_paths = None
    state.profile_frame = None
    state.join_columns = []
    state.baseline = None
    state.all_eras = []
    state.data_loader = None
    state.predictions = pd.DataFrame()
    gc.collect()

    try:
        scoring_result = run_post_training_scoring(
            request=PostTrainingScoringRequest(
                predictions_path=cast(Path, state.predictions_path),
                pred_cols=("prediction",),
                target_col=state.target_col,
                scoring_target_cols=_resolve_scoring_target_cols(
                    data_config=state.data_config,
                    target_col=state.target_col,
                ),
                data_version=state.data_version,
                dataset_variant=state.dataset_variant,
                feature_set=state.feature_set,
                feature_source_paths=state.scoring_feature_source_paths,
                dataset_scope=state.dataset_scope,
                benchmark_model=state.benchmark_model,
                benchmark_data_path=state.benchmark_data_path,
                meta_model_col=state.meta_model_col,
                meta_model_data_path=state.meta_model_data_path,
                era_col=state.era_col,
                id_col=state.id_col,
                data_root=DEFAULT_DATASETS_DIR,
                scoring_mode=state.scoring_mode,
                era_chunk_size=state.scoring_era_chunk_size,
                include_feature_neutral_metrics=state.include_feature_neutral_metrics,
            ),
            client=cast(TrainingDataClient, state.training_client),
        )
    except Exception as exc:
        logger.exception("post-run scoring failed for run_id=%s", state.run_id)
        state.effective_scoring_backend = "failed"
        state.training_runtime_metadata["scoring"] = {
            "mode": state.scoring_mode,
            "era_chunk_size": state.scoring_era_chunk_size,
            "effective_backend": state.effective_scoring_backend,
            "policy": _scoring_policy_payload(default_scoring_policy()),
        }
        log_error(
            state.run_log_path,
            event="post_run_scoring_failed",
            message=str(exc),
            attempt_id=state.run_attempt_id,
        )
        raise TrainingError(f"training_post_run_scoring_failed:{state.run_id}") from exc

    state.summaries = scoring_result.summaries
    score_provenance = scoring_result.score_provenance
    state.effective_scoring_backend = scoring_result.effective_scoring_backend
    state.training_runtime_metadata["scoring"] = {
        "mode": state.scoring_mode,
        "era_chunk_size": state.scoring_era_chunk_size,
        "effective_backend": state.effective_scoring_backend,
        "policy": _scoring_policy_payload(scoring_result.policy),
    }
    save_score_provenance(score_provenance, cast(Path, state.score_provenance_path))
    state.score_provenance_relative = cast(Path, state.score_provenance_path).relative_to(cast(Path, state.output_root))
    state.artifacts["score_provenance"] = str(state.score_provenance_relative)
    if scoring_result.per_era_corr is not None:
        _, _, per_era_corr_relative, per_era_corr_csv_relative = save_per_era_corr_artifacts(
            scoring_result.per_era_corr,
            predictions_dir=cast(Path, state.predictions_dir),
            output_dir=cast(Path, state.output_root),
        )
        state.artifacts["per_era_corr"] = str(per_era_corr_relative)
        state.artifacts["per_era_corr_csv"] = str(per_era_corr_csv_relative)
    state.metrics_status = None
    log_info(
        state.run_log_path,
        event="post_run_scoring_succeeded",
        message=f"effective_backend={state.effective_scoring_backend}",
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
        benchmark_model=state.benchmark_model,
        benchmark_data_path=state.benchmark_data_path,
        meta_model_col=state.meta_model_col,
        meta_model_data_path=state.meta_model_data_path,
        output_dir=cast(Path, state.output_root),
        predictions_relative=cast(Path, state.predictions_relative),
        score_provenance_relative=state.score_provenance_relative,
        summaries=state.summaries,
        cv_meta=state.cv_meta,
        engine_plan=state.engine_plan,
        cv_enabled=state.cv_enabled,
        loading_mode=state.loading_mode,
        scoring_mode=state.scoring_mode,
        scoring_era_chunk_size=state.scoring_era_chunk_size,
        resource_policy=state.resource_policy,
        cache_policy=state.cache_policy,
        scoring_backend=state.effective_scoring_backend,
        scoring_policy=scoring_result.policy,
        metrics_status=None,
    )
    save_results(results, cast(Path, state.results_path))
    state.metrics_payload = _extract_metrics_payload(results)
    save_metrics(state.metrics_payload, cast(Path, state.metrics_path))
    _record_telemetry_metrics(state.telemetry_session, state.metrics_payload)
    return state


def finalize_training_run(state: TrainingPipelineState) -> TrainingRunResult:
    """Persist final manifest state, refresh index, and return run outputs."""
    state.full = None
    state.source_paths = None
    state.profile_frame = None
    state.join_columns = []
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
        engine_mode=state.engine_plan.mode,
        experiment_id=state.experiment_id,
        created_at=state.created_at,
        artifacts=state.artifacts,
        metrics_summary=state.metrics_payload,
        training_metadata=state.training_runtime_metadata,
    )
    save_run_manifest(finished_manifest, cast(Path, state.run_manifest_path))
    try:
        index_run(store_root=state.run_store_root, run_id=state.run_id)
    except StoreError:
        pass
    _ = maybe_log_training_run(
        run_id=state.run_id,
        config=cast(dict[str, object], state.config),
        metrics_payload=state.metrics_payload,
        artifacts=state.artifacts,
        output_root=cast(Path, state.output_root),
    )
    _mark_telemetry_completed(
        state.telemetry_session,
        run_id=state.run_id,
        run_dir=cast(Path, state.output_root),
        run_log_path=state.run_log_path,
        attempt_id=state.run_attempt_id,
    )
    return TrainingRunResult(
        run_id=state.run_id,
        predictions_path=cast(Path, state.predictions_path),
        results_path=cast(Path, state.results_path),
    )


def fail_training_run(state: TrainingPipelineState, exc: Exception) -> None:
    """Persist failure manifest and telemetry for an interrupted run."""
    if state.run_manifest_written:
        failed_manifest = build_run_manifest(
            run_id=state.run_id,
            run_hash=state.run_hash,
            status="FAILED",
            config_path=state.config_path,
            config_hash=state.config_hash,
            data_version=state.data_version,
            feature_set=state.feature_set,
            target_col=state.target_col,
            model_type=state.model_type,
            engine_mode=state.engine_plan.mode,
            experiment_id=state.experiment_id,
            created_at=state.created_at,
            artifacts=state.artifacts,
            error=error_payload(exc),
            training_metadata=state.training_runtime_metadata,
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
        _mark_telemetry_failed(
            state.telemetry_session,
            run_id=state.run_id,
            error=error_payload(exc),
            message=str(exc),
            run_log_path=state.run_log_path,
            attempt_id=state.run_attempt_id,
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
    engine_mode: str | None = None,
    window_size_eras: int | None = None,
    embargo_eras: int | None = None,
    experiment_id: str | None = None,
) -> TrainingRunResult:
    """Execute the full local train -> score pipeline as explicit stages."""
    state = TrainingPipelineState(
        config_path=Path(config_path).expanduser().resolve(),
        output_dir=Path(output_dir).expanduser() if output_dir is not None else None,
        client=client,
        profile=profile,
        engine_mode=engine_mode,
        window_size_eras=window_size_eras,
        embargo_eras=embargo_eras,
        experiment_id=experiment_id,
    )
    try:
        state = prepare_training_run(
            config_path=config_path,
            output_dir=output_dir,
            client=client,
            profile=profile,
            engine_mode=engine_mode,
            window_size_eras=window_size_eras,
            embargo_eras=embargo_eras,
            experiment_id=experiment_id,
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


def _era_sort_key(era: object) -> int | str:
    if isinstance(era, str) and era.isdigit():
        return int(era)
    if isinstance(era, int):
        return era
    return str(era)


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
