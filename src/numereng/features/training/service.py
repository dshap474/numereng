"""Business logic for core training pipeline workflows."""

from __future__ import annotations

import gc
import logging
import math
import os
from importlib import import_module
from pathlib import Path
from typing import cast

import pandas as pd

from numereng.features.store import StoreError, index_run
from numereng.features.telemetry import (
    LocalResourceSampler,
    LocalRunTelemetrySession,
    append_log_line,
    append_resource_sample,
    begin_local_training_session,
    capture_local_resource_sample,
    emit_metric_event,
    emit_stage_event,
    get_launch_metadata,
    mark_job_completed,
    mark_job_failed,
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
    resolve_variant_dataset_filename,
    save_metrics,
    save_predictions,
    save_resolved_config,
    save_results,
    save_run_manifest,
    save_score_provenance,
    select_prediction_columns,
)
from numereng.features.training.run_log import (
    initialize_run_log,
    log_error,
    log_info,
    log_stage,
    resolve_run_log_path,
)
from numereng.features.training.run_store import (
    build_run_manifest,
    compute_config_hash,
    compute_run_hash,
    error_payload,
    run_id_from_hash,
)
from numereng.features.training.scoring.metrics import (
    attach_benchmark_predictions,
    load_custom_benchmark_predictions,
)
from numereng.features.training.scoring.models import PostTrainingScoringRequest
from numereng.features.training.scoring.service import run_post_training_scoring
from numereng.features.training.strategies import (
    TrainingEnginePlan,
    TrainingProfile,
    resolve_training_engine,
)

_MATERIALIZED_LOADING_MODE = "materialized"
_FOLD_LAZY_LOADING_MODE = "fold_lazy"
_SIMPLE_PROFILE: TrainingProfile = "simple"
_PURGED_WALK_FORWARD_PROFILE: TrainingProfile = "purged_walk_forward"
_SUBMISSION_PROFILE: TrainingProfile = "submission"
_MATERIALIZED_SCORING_MODE = "materialized"
_ERA_STREAM_SCORING_MODE = "era_stream"

logger = logging.getLogger(__name__)


def resolve_model_config(model_config: dict[str, object]) -> tuple[str, dict[str, object]]:
    """Resolve required model type/params from model config block."""
    model_type = str(model_config.get("type", "LGBMRegressor"))
    model_params = model_config.get("params")
    if model_params is None:
        raise TrainingConfigError("training_model_params_missing")
    if not isinstance(model_params, dict):
        raise TrainingConfigError("training_model_params_not_mapping")
    return model_type, model_params


def load_and_prepare_data(
    client: TrainingDataClient,
    data_version: str,
    dataset_variant: str,
    feature_set: str,
    target_col: str,
    era_col: str,
    id_col: str,
    full_data_path: str | Path | None,
    dataset_scope: str,
    nan_missing_all_twos: bool,
    missing_value: float,
) -> tuple[pd.DataFrame, list[str]]:
    """Load feature metadata and modeling dataset, then apply preprocessing."""
    features = load_features(client, data_version, feature_set, dataset_variant=dataset_variant)
    full = load_full_data(
        client,
        data_version,
        dataset_variant,
        features,
        era_col,
        target_col,
        id_col,
        full_data_path=full_data_path,
        dataset_scope=dataset_scope,
    )

    if nan_missing_all_twos:
        full = apply_missing_all_twos_as_nan(full, features, era_col, missing_value)

    return full, features


def build_results_payload(
    *,
    model_type: str,
    model_params: dict[str, object],
    model_config: dict[str, object],
    nan_missing_all_twos: bool,
    missing_value: float,
    data_version: str,
    dataset_variant: str,
    feature_set: str,
    target_col: str,
    full_data_path: str | Path | None,
    dataset_scope: str,
    full_rows: int,
    full_eras: int,
    oof_rows: int,
    oof_eras: int,
    configured_embargo_eras: int,
    effective_embargo_eras: int,
    benchmark_model: str,
    benchmark_data_path: str | Path | None,
    meta_model_col: str,
    meta_model_data_path: str | Path | None,
    output_dir: Path,
    predictions_relative: Path,
    score_provenance_relative: Path | None,
    summaries: dict[str, pd.DataFrame] | None,
    cv_meta: dict[str, object],
    engine_plan: TrainingEnginePlan,
    cv_enabled: bool,
    loading_mode: str,
    scoring_mode: str,
    scoring_era_chunk_size: int,
    resource_policy: dict[str, object],
    cache_policy: dict[str, object],
    scoring_backend: str,
    metrics_status: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build canonical results payload persisted for each training run."""
    model_meta: dict[str, object] = {
        "type": model_type,
        "params": model_params,
    }

    for key in (
        "x_groups",
        "data_needed",
        "target_transform",
        "prediction_transform",
        "era_weighting",
        "prediction_batch_size",
        "benchmark",
        "baseline",
    ):
        if key in model_config:
            model_meta[key] = model_config[key]

    metrics_payload: dict[str, object]
    if summaries is None:
        metrics_payload = metrics_status or {"status": "not_applicable"}
    else:
        metrics_corr = summaries["corr"].loc["prediction"].to_dict()
        metrics_fnc = summaries["fnc"].loc["prediction"].to_dict()
        metrics_mmc = summaries["mmc"].loc["prediction"].to_dict()
        metrics_cwmm = summaries["cwmm"].loc["prediction"].to_dict()
        metrics_bmc = summaries["bmc"].loc["prediction"].to_dict()
        metrics_bmc_200 = summaries["bmc_last_200_eras"].loc["prediction"].to_dict()
        metrics_payload = {
            "corr": metrics_corr,
            "fnc": metrics_fnc,
            "mmc": metrics_mmc,
            "cwmm": metrics_cwmm,
            "bmc": metrics_bmc,
            "bmc_last_200_eras": metrics_bmc_200,
        }

    return {
        "model": model_meta,
        "preprocessing": {
            "nan_missing_all_twos": nan_missing_all_twos,
            "missing_value": missing_value,
        },
        "data": {
            "data_version": data_version,
            "dataset_variant": dataset_variant,
            "feature_set": feature_set,
            "target": target_col,
            "full_data_path": str(full_data_path) if full_data_path else None,
            "dataset_scope": dataset_scope,
            "full_rows": full_rows,
            "full_eras": full_eras,
            "oof_rows": oof_rows,
            "oof_eras": oof_eras,
            "embargo_eras": effective_embargo_eras,
            "configured_embargo_eras": configured_embargo_eras,
            "effective_embargo_eras": effective_embargo_eras,
        },
        "benchmark": {
            "model": benchmark_model,
            "file": str(
                benchmark_data_path
                or _default_variant_data_path(
                    data_version,
                    dataset_variant,
                    "full_benchmark_models.parquet",
                )
            ),
        },
        "meta_model": {
            "column": meta_model_col,
            "file": str(
                meta_model_data_path
                or _default_variant_data_path(
                    data_version,
                    dataset_variant,
                    "meta_model.parquet",
                )
            ),
        },
        "output": {
            "output_dir": str(output_dir),
            "predictions_file": str(predictions_relative),
            "score_provenance_file": str(score_provenance_relative) if score_provenance_relative else None,
        },
        "metrics": metrics_payload,
        "cv": cv_meta,
        "training": {
            "engine": {
                "profile": engine_plan.mode,
                "mode": engine_plan.mode,
                "resolved": engine_plan.resolved_config,
                "override_sources": engine_plan.override_sources,
            },
            "loading": {
                "mode": loading_mode,
            },
            "scoring": {
                "mode": scoring_mode,
                "era_chunk_size": scoring_era_chunk_size,
                "effective_backend": scoring_backend,
            },
            "resources": resource_policy,
            "cache": cache_policy,
            "cv": {
                "enabled": cv_enabled,
                "n_splits": cv_meta.get("n_splits"),
                "embargo": cv_meta.get("embargo"),
                "mode": cv_meta.get("mode"),
                "min_train_size": cv_meta.get("min_train_size"),
                "chunk_size": cv_meta.get("chunk_size"),
            },
        },
    }


def run_training(
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
    """Run full training pipeline and return predictions/results artifact paths."""
    resolved_config_path = Path(config_path).expanduser().resolve()
    output_dir_override = Path(output_dir).expanduser() if output_dir is not None else None

    config = load_config(resolved_config_path)
    data_config = _as_dict(config.get("data"))
    preprocessing_config = _as_dict(config.get("preprocessing"))
    training_config = _as_dict(config.get("training"))
    model_config = _as_dict(config.get("model"))

    data_version = str(data_config.get("data_version", "v5.2"))
    dataset_variant = _resolve_dataset_variant(data_config.get("dataset_variant"))
    feature_set = str(data_config.get("feature_set", "small"))
    target_col = str(data_config.get("target_col", "target"))
    era_col = str(data_config.get("era_col", "era"))
    id_col = str(data_config.get("id_col", "id"))
    full_data_path = _optional_path(data_config.get("full_data_path"))
    benchmark_data_path = _optional_path(data_config.get("benchmark_data_path"))
    meta_model_data_path = _optional_path(data_config.get("meta_model_data_path"))
    meta_model_col = str(data_config.get("meta_model_col", "numerai_meta_model"))
    data_embargo_eras = _coerce_int(data_config.get("embargo_eras"), default=13)
    benchmark_model = str(data_config.get("benchmark_model", DEFAULT_BENCHMARK_MODEL))
    dataset_scope_config = _resolve_dataset_scope(data_config.get("dataset_scope"))
    loading_config = _as_dict(data_config.get("loading"))
    loading_mode = _resolve_loading_mode(loading_config.get("mode"))
    scoring_mode = _resolve_scoring_mode(loading_config.get("scoring_mode"))
    scoring_era_chunk_size = _coerce_int(loading_config.get("era_chunk_size"), default=64)
    if scoring_era_chunk_size < 1:
        raise TrainingConfigError("training_data_loading_era_chunk_size_invalid")

    nan_missing_all_twos = bool(preprocessing_config.get("nan_missing_all_twos", False))
    missing_value = _coerce_float(preprocessing_config.get("missing_value"), default=2.0)

    resource_policy = _resolve_resource_policy(_as_dict(training_config.get("resources")))
    cache_policy = _resolve_cache_policy(_as_dict(training_config.get("cache")))
    _apply_resource_policy(resource_policy)

    engine_plan = resolve_training_engine(
        training_config=training_config,
        data_config=data_config,
        profile=profile,
        engine_mode=engine_mode,
        window_size_eras=window_size_eras,
        embargo_eras=embargo_eras,
    )
    dataset_scope = _resolve_dataset_scope_for_profile(
        profile=engine_plan.mode,
        configured_scope=dataset_scope_config,
        dataset_variant=dataset_variant,
        full_data_path=full_data_path,
    )
    model_type, model_params = resolve_model_config(model_config)
    run_hash = compute_run_hash(
        config=config,
        data_version=data_version,
        feature_set=feature_set,
        target_col=target_col,
        model_type=model_type,
        engine_mode=engine_plan.mode,
        engine_settings=engine_plan.resolved_config,
    )
    run_id = run_id_from_hash(run_hash)
    config_hash = compute_config_hash(config)

    output_root, baselines_dir, results_dir, predictions_dir = resolve_output_locations(
        config,
        output_dir_override,
        run_id,
    )
    run_manifest_path = resolve_run_manifest_path(output_root)
    resolved_snapshot_path = resolve_resolved_config_path(output_root)
    metrics_path = resolve_metrics_path(output_root)
    score_provenance_path = resolve_score_provenance_path(output_root)
    training_runtime_metadata: dict[str, object] = {
        "data": {
            "dataset_scope": dataset_scope,
            "dataset_variant": dataset_variant,
        },
        "loading": {"mode": loading_mode},
        "scoring": {
            "mode": scoring_mode,
            "era_chunk_size": scoring_era_chunk_size,
            "effective_backend": scoring_mode,
        },
        "resources": resource_policy,
        "cache": cache_policy,
    }
    created_at: str | None = None
    run_log_path = resolve_run_log_path(output_root)
    artifacts: dict[str, str] = {
        "resolved_config": str(resolved_snapshot_path.relative_to(output_root)),
        "log": str(run_log_path.relative_to(output_root)),
    }
    predictions_path: Path | None = None
    results_path: Path | None = None
    score_provenance_relative: Path | None = None
    run_store_root: Path | None = None
    launch_metadata = get_launch_metadata()
    telemetry_session: LocalRunTelemetrySession | None = None
    telemetry_sampler: LocalResourceSampler | None = None
    completed_stages: list[str] = []

    try:
        if launch_metadata is not None:
            try:
                run_store_root = _resolve_store_root_for_run(output_root)
                telemetry_session = begin_local_training_session(
                    store_root=run_store_root,
                    config_path=resolved_config_path,
                    source=launch_metadata.source,
                    experiment_id=experiment_id,
                    operation_type=launch_metadata.operation_type,
                    job_type=launch_metadata.job_type,
                    request_payload={
                        "run_hash": run_hash,
                        "engine_mode": engine_plan.mode,
                        "data_version": data_version,
                        "feature_set": feature_set,
                        "target_col": target_col,
                    },
                )
                if telemetry_session is not None:
                    mark_job_starting(telemetry_session, pid=os.getpid(), worker_id="local")
                    mark_job_running(telemetry_session)
                    telemetry_sampler = start_local_resource_sampler(telemetry_session, interval_seconds=5.0)
                    _record_telemetry_stage(
                        telemetry_session,
                        completed_stages=completed_stages,
                        stage_name="initializing",
                        message="Training session initialized.",
                        run_log_path=None,
                    )
            except Exception:
                logger.exception("training telemetry bootstrap failed for run_id=%s", run_id)

        running_manifest = build_run_manifest(
            run_id=run_id,
            run_hash=run_hash,
            status="RUNNING",
            config_path=resolved_config_path,
            config_hash=config_hash,
            data_version=data_version,
            feature_set=feature_set,
            target_col=target_col,
            model_type=model_type,
            engine_mode=engine_plan.mode,
            experiment_id=experiment_id,
            artifacts=artifacts,
            training_metadata=training_runtime_metadata,
        )
        save_run_manifest(running_manifest, run_manifest_path)
        try:
            initialize_run_log(output_root)
        except Exception:
            logger.exception("failed to initialize run-local log for run_id=%s", run_id)
        log_info(
            run_log_path,
            event="run_started",
            message=(
                f"run_id={run_id} engine_mode={engine_plan.mode} "
                f"loading_mode={loading_mode} scoring_mode={scoring_mode}"
            ),
        )
        save_resolved_config(config, resolved_snapshot_path)
        created_at = str(running_manifest["created_at"])
        training_client = create_training_data_client() if client is None else client
        _record_telemetry_stage(
            telemetry_session,
            completed_stages=completed_stages,
            stage_name="load_data",
            message="Loading training datasets and feature metadata.",
            run_log_path=run_log_path,
        )

        features = load_features(training_client, data_version, feature_set, dataset_variant=dataset_variant)
        full: pd.DataFrame | None = None
        source_paths: tuple[Path, ...] | None = None
        profile_frame: pd.DataFrame | None = None
        lazy_include_validation_only = False
        baseline: pd.DataFrame | None = None

        if loading_mode == _MATERIALIZED_LOADING_MODE:
            full = load_full_data(
                training_client,
                data_version,
                dataset_variant,
                features,
                era_col,
                target_col,
                id_col,
                full_data_path=full_data_path,
                dataset_scope=dataset_scope,
            )
            if nan_missing_all_twos:
                full = apply_missing_all_twos_as_nan(full, features, era_col, missing_value)
            all_eras = sorted(set(full[era_col].tolist()), key=_era_sort_key)
            full_rows = int(full.shape[0])
            full_eras = int(full[era_col].nunique())
        else:
            if nan_missing_all_twos:
                raise TrainingConfigError("training_data_loading_fold_lazy_disallows_nan_missing_all_twos")
            # In split-source lazy mode, filter `data_type=validation` rows on validation sources only.
            lazy_include_validation_only = full_data_path is None and dataset_scope == "train_plus_validation"
            source_paths = resolve_fold_lazy_source_paths(
                training_client,
                data_version,
                dataset_variant=dataset_variant,
                full_data_path=full_data_path,
                dataset_scope=dataset_scope,
                data_root=DEFAULT_DATASETS_DIR,
            )
            all_eras = list_lazy_source_eras(
                source_paths,
                era_col=era_col,
                include_validation_only=lazy_include_validation_only,
            )
            profile_frame = load_fold_data_lazy(
                source_paths,
                eras=all_eras,
                columns=[era_col],
                era_col=era_col,
                id_col=None,
                include_validation_only=lazy_include_validation_only,
            )
            all_eras = sorted(set(profile_frame[era_col].tolist()), key=_era_sort_key)
            full_rows = int(profile_frame.shape[0])
            full_eras = int(profile_frame[era_col].nunique()) if era_col in profile_frame.columns else len(all_eras)

        raw_x_groups = model_config.get("x_groups") or model_config.get("data_needed")
        if raw_x_groups is None:
            x_groups_input: list[str] | None = None
        elif isinstance(raw_x_groups, (list, tuple)):
            x_groups_input = [str(item) for item in raw_x_groups]
        else:
            raise TrainingConfigError("training_model_x_groups_invalid")
        try:
            x_groups = normalize_x_groups(x_groups_input)
        except ValueError as exc:
            raise TrainingConfigError(str(exc)) from exc

        baseline_col: str | None = None
        join_columns: list[FoldJoinColumn] = []

        if "baseline" in x_groups:
            baseline_spec = _as_dict(model_config.get("baseline"))
            baseline_name = baseline_spec.get("name")
            baseline_path = baseline_spec.get("predictions_path")
            pred_col = str(baseline_spec.get("pred_col", "prediction"))

            if not baseline_name or not baseline_path:
                raise TrainingConfigError("training_baseline_config_missing_name_or_predictions_path")
            if not id_col:
                raise TrainingConfigError("training_id_col_required_for_baseline")

            resolved_baseline_path = _resolve_baseline_path(str(baseline_path), baselines_dir)
            baseline, baseline_col = load_custom_benchmark_predictions(
                resolved_baseline_path,
                str(baseline_name),
                pred_col=pred_col,
                era_col=era_col,
                id_col=id_col,
            )
            if loading_mode == _MATERIALIZED_LOADING_MODE:
                if full is None:
                    raise TrainingConfigError("training_data_loading_materialized_missing_frame")
                full = attach_benchmark_predictions(full, baseline, baseline_col, era_col=era_col, id_col=id_col)
            else:
                join_columns.append(FoldJoinColumn(frame=baseline, column=baseline_col))

        x_cols = build_x_cols(
            x_groups=x_groups,
            features=features,
            era_col=era_col,
            id_col=id_col,
            baseline_col=baseline_col,
        )

        data_loader: ModelDataLoaderProtocol | None
        if loading_mode == _MATERIALIZED_LOADING_MODE:
            if full is None:
                raise TrainingConfigError("training_data_loading_materialized_missing_frame")
            data_loader = build_model_data_loader(
                full=full,
                x_cols=x_cols,
                era_col=era_col,
                target_col=target_col,
                id_col=id_col,
            )
        else:
            if source_paths is None:
                raise TrainingConfigError("training_data_loading_fold_lazy_missing_sources")
            data_loader = build_lazy_parquet_data_loader(
                source_paths=source_paths,
                x_cols=x_cols,
                era_col=era_col,
                target_col=target_col,
                id_col=id_col,
                include_validation_only=lazy_include_validation_only,
                join_columns=join_columns,
            )

        simple_train_eras: list[object] | None = None
        simple_val_eras: list[object] | None = None
        if engine_plan.mode == _SIMPLE_PROFILE:
            if loading_mode == _FOLD_LAZY_LOADING_MODE and source_paths is not None and len(source_paths) >= 2:
                train_path, validation_path = source_paths[0], source_paths[1]
            elif full_data_path is not None:
                train_path, validation_path = _resolve_simple_split_paths_from_full_data_path(full_data_path)
            else:
                train_path, validation_path = ensure_split_dataset_paths(
                    training_client,
                    data_version,
                    dataset_variant=dataset_variant,
                    data_root=DEFAULT_DATASETS_DIR,
                )
            simple_train_eras = list_lazy_source_eras(
                (train_path,),
                era_col=era_col,
                include_validation_only=False,
            )
            simple_val_eras = list_lazy_source_eras(
                (validation_path,),
                era_col=era_col,
                include_validation_only=True,
            )
            if not simple_train_eras or not simple_val_eras:
                raise TrainingConfigError("training_profile_simple_requires_nonempty_split_eras")

        cv_config = dict(engine_plan.cv_config)
        if engine_plan.mode == _SIMPLE_PROFILE:
            cv_config["embargo"] = 0
            cv_config["train_eras"] = simple_train_eras
            cv_config["val_eras"] = simple_val_eras
        else:
            cv_config.setdefault("embargo", data_embargo_eras)
        cv_enabled = bool(cv_config.get("enabled", True))
        _record_telemetry_stage(
            telemetry_session,
            completed_stages=completed_stages,
            stage_name="train_model",
            message="Building model predictions.",
            run_log_path=run_log_path,
        )

        if _is_submission_profile(engine_plan.mode):
            predictions, cv_meta = build_full_history_predictions(
                all_eras,
                data_loader,
                model_type,
                model_params,
                model_config,
                id_col,
                era_col,
                target_col,
                feature_cols=features,
            )
        else:
            if not cv_enabled:
                raise TrainingConfigError("training_cv_required")
            predictions, cv_meta = build_oof_predictions(
                all_eras,
                data_loader,
                model_type,
                model_params,
                model_config,
                cv_config,
                id_col,
                era_col,
                target_col,
                feature_cols=features,
                parallel_folds=cast(int, resource_policy["parallel_folds"]),
                parallel_backend=str(resource_policy["parallel_backend"]),
                memmap_enabled=bool(resource_policy["memmap_enabled"]),
            )

        effective_embargo_eras = _coerce_int(cv_meta.get("embargo"), default=data_embargo_eras)

        predictions = select_prediction_columns(predictions, id_col, era_col, target_col)
        predictions_path, predictions_relative = save_predictions(
            predictions,
            config,
            resolved_config_path,
            predictions_dir,
            output_root,
        )
        artifacts["predictions"] = str(predictions_relative)

        oof_rows = int(predictions.shape[0])
        oof_eras = int(predictions[era_col].nunique())
        scoring_feature_source_paths = source_paths
        summaries: dict[str, pd.DataFrame] | None = None
        metrics_status: dict[str, object] | None
        effective_scoring_backend = scoring_mode

        _record_telemetry_stage(
            telemetry_session,
            completed_stages=completed_stages,
            stage_name="score_predictions",
            message="Deferring scoring to post-run metrics phase.",
            run_log_path=run_log_path,
        )
        if _is_submission_profile(engine_plan.mode):
            metrics_status = {
                "status": "not_applicable",
                "reason": "training_profile_submission_no_validation_metrics",
            }
            effective_scoring_backend = "not_applicable"
        else:
            metrics_status = {
                "status": "pending",
                "reason": "post_run_metrics_pending",
            }
            effective_scoring_backend = "pending"

        training_runtime_metadata["scoring"] = {
            "mode": scoring_mode,
            "era_chunk_size": scoring_era_chunk_size,
            "effective_backend": effective_scoring_backend,
        }
        _record_telemetry_stage(
            telemetry_session,
            completed_stages=completed_stages,
            stage_name="persist_artifacts",
            message="Persisting run artifacts and metrics.",
            run_log_path=run_log_path,
        )

        results_path = resolve_results_path(config, resolved_config_path, results_dir)
        results = build_results_payload(
            model_type=model_type,
            model_params=model_params,
            model_config=model_config,
            nan_missing_all_twos=nan_missing_all_twos,
            missing_value=missing_value,
            data_version=data_version,
            dataset_variant=dataset_variant,
            feature_set=feature_set,
            target_col=target_col,
            full_data_path=full_data_path,
            dataset_scope=dataset_scope,
            full_rows=full_rows,
            full_eras=full_eras,
            oof_rows=oof_rows,
            oof_eras=oof_eras,
            configured_embargo_eras=data_embargo_eras,
            effective_embargo_eras=effective_embargo_eras,
            benchmark_model=benchmark_model,
            benchmark_data_path=benchmark_data_path,
            meta_model_col=meta_model_col,
            meta_model_data_path=meta_model_data_path,
            output_dir=output_root,
            predictions_relative=predictions_relative,
            score_provenance_relative=score_provenance_relative,
            summaries=summaries,
            cv_meta=cv_meta,
            engine_plan=engine_plan,
            cv_enabled=cv_enabled,
            loading_mode=loading_mode,
            scoring_mode=scoring_mode,
            scoring_era_chunk_size=scoring_era_chunk_size,
            resource_policy=resource_policy,
            cache_policy=cache_policy,
            scoring_backend=effective_scoring_backend,
            metrics_status=metrics_status,
        )
        save_results(results, results_path)
        artifacts["results"] = str(results_path.relative_to(output_root))

        metrics_payload = _extract_metrics_payload(results)
        save_metrics(metrics_payload, metrics_path)
        artifacts["metrics"] = str(metrics_path.relative_to(output_root))
        _record_telemetry_stage(
            telemetry_session,
            completed_stages=completed_stages,
            stage_name="index_run",
            message="Indexing run artifacts in store.",
            run_log_path=run_log_path,
        )

        run_store_root = _resolve_store_root_for_run(output_root)
        try:
            index_run(store_root=run_store_root, run_id=run_id)
        except StoreError as exc:
            raise TrainingError(f"training_store_index_failed:{run_id}") from exc

        if not _is_submission_profile(engine_plan.mode):
            _record_telemetry_stage(
                telemetry_session,
                completed_stages=completed_stages,
                stage_name="score_predictions_post_run",
                message="Computing metrics in post-run phase.",
                run_log_path=run_log_path,
            )
            log_info(
                run_log_path,
                event="post_run_scoring_started",
                message=f"mode={scoring_mode} era_chunk_size={scoring_era_chunk_size}",
            )
            # Release large in-memory training frames before post-run scoring to
            # lower peak RAM usage during metric computation.
            full = None
            source_paths = None
            profile_frame = None
            join_columns = []
            baseline = None
            all_eras = []
            data_loader = None
            predictions = pd.DataFrame()
            gc.collect()

            try:
                scoring_result = run_post_training_scoring(
                    request=PostTrainingScoringRequest(
                        predictions_path=predictions_path,
                        pred_cols=("prediction",),
                        target_col=target_col,
                        data_version=data_version,
                        dataset_variant=dataset_variant,
                        feature_set=feature_set,
                        feature_cols=tuple(features),
                        feature_source_paths=scoring_feature_source_paths,
                        full_data_path=full_data_path,
                        dataset_scope=dataset_scope,
                        benchmark_model=benchmark_model,
                        benchmark_data_path=benchmark_data_path,
                        meta_model_col=meta_model_col,
                        meta_model_data_path=meta_model_data_path,
                        era_col=era_col,
                        id_col=id_col,
                        data_root=DEFAULT_DATASETS_DIR,
                        scoring_mode=scoring_mode,
                        era_chunk_size=scoring_era_chunk_size,
                    ),
                    client=training_client,
                )
                summaries = scoring_result.summaries
                score_provenance = scoring_result.score_provenance
                effective_scoring_backend = scoring_result.effective_scoring_backend
                training_runtime_metadata["scoring"] = {
                    "mode": scoring_mode,
                    "era_chunk_size": scoring_era_chunk_size,
                    "effective_backend": effective_scoring_backend,
                }
                save_score_provenance(score_provenance, score_provenance_path)
                score_provenance_relative = score_provenance_path.relative_to(output_root)
                artifacts["score_provenance"] = str(score_provenance_relative)
                metrics_status = None
                log_info(
                    run_log_path,
                    event="post_run_scoring_succeeded",
                    message=f"effective_backend={effective_scoring_backend}",
                )
            except Exception as exc:
                logger.exception("post-run scoring failed for run_id=%s", run_id)
                summaries = None
                metrics_status = {
                    "status": "failed",
                    "reason": str(exc),
                }
                effective_scoring_backend = "failed"
                training_runtime_metadata["scoring"] = {
                    "mode": scoring_mode,
                    "era_chunk_size": scoring_era_chunk_size,
                    "effective_backend": effective_scoring_backend,
                }
                log_error(
                    run_log_path,
                    event="post_run_scoring_failed",
                    message=str(exc),
                )

            results = build_results_payload(
                model_type=model_type,
                model_params=model_params,
                model_config=model_config,
                nan_missing_all_twos=nan_missing_all_twos,
                missing_value=missing_value,
                data_version=data_version,
                dataset_variant=dataset_variant,
                feature_set=feature_set,
                target_col=target_col,
                full_data_path=full_data_path,
                dataset_scope=dataset_scope,
                full_rows=full_rows,
                full_eras=full_eras,
                oof_rows=oof_rows,
                oof_eras=oof_eras,
                configured_embargo_eras=data_embargo_eras,
                effective_embargo_eras=effective_embargo_eras,
                benchmark_model=benchmark_model,
                benchmark_data_path=benchmark_data_path,
                meta_model_col=meta_model_col,
                meta_model_data_path=meta_model_data_path,
                output_dir=output_root,
                predictions_relative=predictions_relative,
                score_provenance_relative=score_provenance_relative,
                summaries=summaries,
                cv_meta=cv_meta,
                engine_plan=engine_plan,
                cv_enabled=cv_enabled,
                loading_mode=loading_mode,
                scoring_mode=scoring_mode,
                scoring_era_chunk_size=scoring_era_chunk_size,
                resource_policy=resource_policy,
                cache_policy=cache_policy,
                scoring_backend=effective_scoring_backend,
                metrics_status=metrics_status,
            )
            save_results(results, results_path)
            metrics_payload = _extract_metrics_payload(results)
            save_metrics(metrics_payload, metrics_path)
            _record_telemetry_metrics(telemetry_session, metrics_payload)

        # Release large training state before final manifest/index updates.
        full = None
        source_paths = None
        profile_frame = None
        join_columns = []
        baseline = None
        all_eras = []
        data_loader = None
        predictions = pd.DataFrame()
        gc.collect()

        _record_telemetry_stage(
            telemetry_session,
            completed_stages=completed_stages,
            stage_name="finalize_manifest",
            message="Writing final run manifest.",
            run_log_path=run_log_path,
        )
        finished_manifest = build_run_manifest(
            run_id=run_id,
            run_hash=run_hash,
            status="FINISHED",
            config_path=resolved_config_path,
            config_hash=config_hash,
            data_version=data_version,
            feature_set=feature_set,
            target_col=target_col,
            model_type=model_type,
            engine_mode=engine_plan.mode,
            experiment_id=experiment_id,
            created_at=created_at,
            artifacts=artifacts,
            metrics_summary=metrics_payload,
            training_metadata=training_runtime_metadata,
        )
        save_run_manifest(finished_manifest, run_manifest_path)
        try:
            index_run(store_root=run_store_root, run_id=run_id)
        except StoreError:
            # Keep successful training as FINISHED; this second pass only refreshes
            # index status/details after final manifest write.
            pass

        metrics_payload = _extract_metrics_payload(results)
        # Optional observer hook; never fail core training on MLflow logging issues.
        _ = maybe_log_training_run(
            run_id=run_id,
            config=config,
            metrics_payload=metrics_payload,
            artifacts=artifacts,
            output_root=output_root,
        )
        _mark_telemetry_completed(
            telemetry_session,
            run_id=run_id,
            run_dir=output_root,
            run_log_path=run_log_path,
        )

        return TrainingRunResult(
            run_id=run_id,
            predictions_path=predictions_path,
            results_path=results_path,
        )
    except Exception as exc:
        failed_manifest = build_run_manifest(
            run_id=run_id,
            run_hash=run_hash,
            status="FAILED",
            config_path=resolved_config_path,
            config_hash=config_hash,
            data_version=data_version,
            feature_set=feature_set,
            target_col=target_col,
            model_type=model_type,
            engine_mode=engine_plan.mode,
            experiment_id=experiment_id,
            created_at=created_at,
            artifacts=artifacts,
            error=error_payload(exc),
            training_metadata=training_runtime_metadata,
        )
        try:
            save_run_manifest(failed_manifest, run_manifest_path)
        except Exception:
            pass
        if run_store_root is not None:
            try:
                index_run(store_root=run_store_root, run_id=run_id)
            except Exception:
                pass
        _mark_telemetry_failed(
            telemetry_session,
            run_id=run_id,
            error=error_payload(exc),
            message=str(exc),
            run_log_path=run_log_path,
        )
        raise
    finally:
        stop_local_resource_sampler(telemetry_sampler)


def _record_telemetry_stage(
    session: LocalRunTelemetrySession | None,
    *,
    completed_stages: list[str],
    stage_name: str,
    message: str,
    run_log_path: Path | None,
) -> None:
    log_stage(run_log_path, stage_name=stage_name, message=message)

    if session is not None:
        try:
            emit_stage_event(
                session,
                current_stage=stage_name,
                completed_stages=list(completed_stages),
            )
            append_log_line(session, stream="stdout", line=f"[stage] {stage_name} :: {message}")
            append_resource_sample(session, sample=capture_local_resource_sample())
        except Exception:
            logger.exception("failed to record telemetry stage: %s", stage_name)
    if stage_name not in completed_stages:
        completed_stages.append(stage_name)


def _record_telemetry_metrics(
    session: LocalRunTelemetrySession | None,
    metrics_payload: dict[str, object],
) -> None:
    if session is None:
        return
    try:
        emit_metric_event(session, metrics=_telemetry_metric_payload(metrics_payload))
    except Exception:
        logger.exception("failed to emit telemetry metrics")


def _extract_metrics_payload(results: dict[str, object]) -> dict[str, object]:
    metrics_payload_obj = results.get("metrics")
    if isinstance(metrics_payload_obj, dict):
        return cast(dict[str, object], metrics_payload_obj)
    return {}


def _mark_telemetry_completed(
    session: LocalRunTelemetrySession | None,
    *,
    run_id: str,
    run_dir: Path,
    run_log_path: Path | None,
) -> None:
    log_info(run_log_path, event="run_completed", message=f"run_id={run_id} run_dir={run_dir}")
    if session is not None:
        try:
            mark_job_completed(
                session,
                canonical_run_id=run_id,
                run_dir=str(run_dir),
                exit_code=0,
            )
            append_log_line(session, stream="stdout", line=f"[telemetry] completed run {run_id}")
        except Exception:
            logger.exception("failed to mark telemetry completion for run_id=%s", run_id)


def _mark_telemetry_failed(
    session: LocalRunTelemetrySession | None,
    *,
    run_id: str,
    error: dict[str, str],
    message: str,
    run_log_path: Path | None,
) -> None:
    log_error(run_log_path, event="run_failed", message=f"run_id={run_id} error={message}")
    if session is not None:
        try:
            mark_job_failed(
                session,
                error=error,
            )
            append_log_line(session, stream="stderr", line=f"[telemetry] failed run {run_id}: {message}")
        except Exception:
            logger.exception("failed to mark telemetry failure for run_id=%s", run_id)


def _telemetry_metric_payload(metrics_payload: dict[str, object]) -> dict[str, object]:
    payload: dict[str, object] = {}
    corr_obj = metrics_payload.get("corr")
    if isinstance(corr_obj, dict):
        corr_mean = _coerce_finite_float(corr_obj.get("mean"))
        corr_sharpe = _coerce_finite_float(corr_obj.get("sharpe"))
        if corr_mean is not None:
            payload["corr20v2_mean"] = corr_mean
        if corr_sharpe is not None:
            payload["corr20v2_sharpe"] = corr_sharpe
    fnc_obj = metrics_payload.get("fnc")
    if isinstance(fnc_obj, dict):
        fnc_mean = _coerce_finite_float(fnc_obj.get("mean"))
        if fnc_mean is not None:
            payload["fnc_mean"] = fnc_mean
    mmc_obj = metrics_payload.get("mmc")
    if isinstance(mmc_obj, dict):
        mmc_mean = _coerce_finite_float(mmc_obj.get("mean"))
        if mmc_mean is not None:
            payload["mmc_mean"] = mmc_mean
    bmc_obj = metrics_payload.get("bmc")
    if isinstance(bmc_obj, dict):
        bmc_mean = _coerce_finite_float(bmc_obj.get("mean"))
        if bmc_mean is not None:
            payload["bmc_mean"] = bmc_mean
    payout_obj = metrics_payload.get("payout_estimate")
    if isinstance(payout_obj, dict):
        payout_mean = _coerce_finite_float(payout_obj.get("mean"))
        if payout_mean is not None:
            payload["payout_estimate_mean"] = payout_mean
    return payload


def _as_dict(value: object) -> dict[str, object]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    raise TrainingConfigError("training_config_section_not_mapping")


def _resolve_loading_mode(value: object) -> str:
    if value is None:
        return _MATERIALIZED_LOADING_MODE
    resolved = str(value)
    if resolved not in {_MATERIALIZED_LOADING_MODE, _FOLD_LAZY_LOADING_MODE}:
        raise TrainingConfigError("training_data_loading_mode_invalid")
    return resolved


def _is_submission_profile(value: object) -> bool:
    return str(value) in {"submission", "full_history"}


def _resolve_dataset_scope(value: object) -> str:
    if value is None:
        return "train_only"
    resolved = str(value)
    if resolved not in {"train_only", "train_plus_validation"}:
        raise TrainingConfigError("training_data_dataset_scope_invalid")
    return resolved


def _resolve_dataset_scope_for_profile(
    *,
    profile: TrainingProfile,
    configured_scope: str,
    dataset_variant: str,
    full_data_path: str | Path | None,
) -> str:
    if profile in {_PURGED_WALK_FORWARD_PROFILE, _SUBMISSION_PROFILE}:
        return "train_plus_validation"
    if profile == _SIMPLE_PROFILE:
        if dataset_variant == "downsampled":
            raise TrainingConfigError("training_profile_simple_disallows_downsampled_dataset_variant")
        return "train_plus_validation"
    return configured_scope


def _resolve_simple_split_paths_from_full_data_path(full_data_path: str | Path) -> tuple[Path, Path]:
    full_path = Path(full_data_path).expanduser().resolve()
    train_path = full_path.parent / "train.parquet"
    validation_path = full_path.parent / "validation.parquet"
    if not train_path.is_file() or not validation_path.is_file():
        raise TrainingConfigError("training_profile_simple_requires_split_sources_near_full_data_path")
    return train_path, validation_path


def _resolve_dataset_variant(value: object) -> str:
    if value is None:
        raise TrainingConfigError("training_data_dataset_variant_required")
    resolved = str(value)
    if resolved not in {"non_downsampled", "downsampled"}:
        raise TrainingConfigError("training_data_dataset_variant_invalid")
    return resolved


def _default_variant_data_path(data_version: str, dataset_variant: str, filename: str) -> str:
    resolved_filename = resolve_variant_dataset_filename(dataset_variant=dataset_variant, filename=filename)
    return f"{data_version}/{resolved_filename}"


def _resolve_scoring_mode(value: object) -> str:
    if value is None:
        return _MATERIALIZED_SCORING_MODE
    resolved = str(value)
    if resolved not in {_MATERIALIZED_SCORING_MODE, _ERA_STREAM_SCORING_MODE}:
        raise TrainingConfigError("training_data_loading_scoring_mode_invalid")
    return resolved


def _resolve_resource_policy(config: dict[str, object]) -> dict[str, object]:
    parallel_folds = _coerce_int(config.get("parallel_folds"), default=1)
    parallel_backend = str(config.get("parallel_backend", "joblib"))
    if parallel_backend != "joblib":
        raise TrainingConfigError("training_resources_parallel_backend_invalid")
    if parallel_folds < 1:
        raise TrainingConfigError("training_resources_integer_value_invalid")
    max_threads_per_worker_obj = config.get("max_threads_per_worker")
    if max_threads_per_worker_obj is None or max_threads_per_worker_obj == "default":
        max_threads_per_worker = max(1, _available_cpu_count() // parallel_folds)
    else:
        max_threads_per_worker = _coerce_int(max_threads_per_worker_obj, default=0)
        if max_threads_per_worker < 1:
            raise TrainingConfigError("training_resources_integer_value_invalid")
    sklearn_working_memory_obj = config.get("sklearn_working_memory_mib")
    sklearn_working_memory_mib: int | None
    if sklearn_working_memory_obj is None:
        sklearn_working_memory_mib = None
    else:
        sklearn_working_memory_mib = _coerce_int(sklearn_working_memory_obj, default=0)
        if sklearn_working_memory_mib < 1:
            raise TrainingConfigError("training_resources_sklearn_working_memory_invalid")
    return {
        "parallel_folds": parallel_folds,
        "parallel_backend": parallel_backend,
        "memmap_enabled": bool(config.get("memmap_enabled", True)),
        "max_threads_per_worker": max_threads_per_worker,
        "sklearn_working_memory_mib": sklearn_working_memory_mib,
    }


def _resolve_cache_policy(config: dict[str, object]) -> dict[str, object]:
    mode = str(config.get("mode", "deterministic"))
    if mode != "deterministic":
        raise TrainingConfigError("training_cache_mode_invalid")
    return {
        "mode": mode,
        "cache_fold_specs": bool(config.get("cache_fold_specs", True)),
        "cache_features": bool(config.get("cache_features", True)),
        "cache_labels": bool(config.get("cache_labels", True)),
        "cache_fold_matrices": bool(config.get("cache_fold_matrices", False)),
    }


def _apply_resource_policy(resource_policy: dict[str, object]) -> None:
    max_threads = str(resource_policy.get("max_threads_per_worker", 1))
    os.environ["OMP_NUM_THREADS"] = max_threads
    os.environ["OPENBLAS_NUM_THREADS"] = max_threads
    os.environ["MKL_NUM_THREADS"] = max_threads
    os.environ["NUMEXPR_NUM_THREADS"] = max_threads

    sklearn_working_memory_mib = resource_policy.get("sklearn_working_memory_mib")
    if sklearn_working_memory_mib is None:
        return
    if not isinstance(sklearn_working_memory_mib, int):
        raise TrainingConfigError("training_resources_sklearn_working_memory_invalid")
    try:
        sklearn_mod = import_module("sklearn")
        sklearn_set_config = sklearn_mod.set_config
    except ImportError as exc:
        raise TrainingConfigError("training_resources_sklearn_dependency_missing") from exc
    sklearn_set_config(working_memory=sklearn_working_memory_mib)


def _available_cpu_count() -> int:
    try:
        affinity = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        affinity = None
    if affinity:
        return len(affinity)
    cpu_count = os.cpu_count()
    if isinstance(cpu_count, int) and cpu_count > 0:
        return cpu_count
    return 1


def _optional_path(value: object) -> str | Path | None:
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        return value
    raise TrainingConfigError("training_config_path_value_invalid")


def _resolve_baseline_path(path: str, baselines_dir: Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (baselines_dir / candidate).resolve()


def _coerce_int(value: object, *, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise TrainingConfigError("training_config_integer_value_invalid")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise TrainingConfigError("training_config_integer_value_invalid") from exc
    raise TrainingConfigError("training_config_integer_value_invalid")


def _coerce_float(value: object, *, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        raise TrainingConfigError("training_config_float_value_invalid")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:
            raise TrainingConfigError("training_config_float_value_invalid") from exc
    raise TrainingConfigError("training_config_float_value_invalid")


def _coerce_finite_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number):
            return number
    return None


def _resolve_store_root_for_run(run_dir: Path) -> Path:
    """Resolve store root path from canonical run dir `<store_root>/runs/<run_id>`."""
    if run_dir.parent.name != "runs":
        raise TrainingConfigError("training_output_run_dir_invalid")
    return run_dir.parent.parent


def _era_sort_key(era: object) -> int | str:
    if isinstance(era, int):
        return era
    if isinstance(era, str) and era.isdigit():
        return int(era)
    return str(era)
