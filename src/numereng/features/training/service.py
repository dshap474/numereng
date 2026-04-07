"""Business logic for core training pipeline workflows."""

from __future__ import annotations

import logging
import math
import os
from importlib import import_module
from pathlib import Path
from typing import cast

import pandas as pd

from numereng.config.training.contracts import PostTrainingScoringPolicy
from numereng.features.scoring.metrics import (
    attach_benchmark_predictions,  # noqa: F401
    load_custom_benchmark_predictions,  # noqa: F401
)
from numereng.features.scoring.models import BenchmarkSource, CanonicalScoringStage, ResolvedScoringPolicy
from numereng.features.scoring.service import run_post_training_scoring, run_scoring  # noqa: F401
from numereng.features.store import index_run  # noqa: F401
from numereng.features.telemetry import (
    LocalRunTelemetrySession,
    append_log_line,
    append_resource_sample,
    capture_local_resource_sample,
    emit_metric_event,
    emit_stage_event,
    is_cancel_requested,
    mark_job_canceled,
    mark_job_completed,
    mark_job_failed,
)
from numereng.features.training.client import TrainingDataClient, create_training_data_client  # noqa: F401
from numereng.features.training.cv import (
    build_full_history_predictions,  # noqa: F401
    build_oof_predictions,  # noqa: F401
)
from numereng.features.training.errors import TrainingCanceledError, TrainingConfigError, TrainingError
from numereng.features.training.mlflow_tracking import maybe_log_training_run  # noqa: F401
from numereng.features.training.models import (
    TrainingRunResult,
    build_lazy_parquet_data_loader,  # noqa: F401
    build_model_data_loader,  # noqa: F401
    build_x_cols,  # noqa: F401
    normalize_x_groups,  # noqa: F401
)
from numereng.features.training.repo import (
    apply_missing_all_twos_as_nan,
    ensure_split_dataset_paths,  # noqa: F401
    list_lazy_source_eras,  # noqa: F401
    load_config,  # noqa: F401
    load_features,
    load_fold_data_lazy,  # noqa: F401
    load_full_data,
    resolve_active_benchmark_metadata_path,
    resolve_active_benchmark_predictions_path,
    resolve_fold_lazy_source_paths,  # noqa: F401
    resolve_metrics_path,  # noqa: F401
    resolve_output_locations,  # noqa: F401
    resolve_resolved_config_path,  # noqa: F401
    resolve_results_path,  # noqa: F401
    resolve_run_manifest_path,  # noqa: F401
    resolve_score_provenance_path,  # noqa: F401
    resolve_variant_dataset_filename,
    save_metrics,  # noqa: F401
    save_predictions,  # noqa: F401
    save_resolved_config,  # noqa: F401
    save_results,  # noqa: F401
    save_run_manifest,  # noqa: F401
    save_score_provenance,  # noqa: F401
    select_prediction_columns,  # noqa: F401
)
from numereng.features.training.run_lock import (
    RUN_LOCK_FILENAME,
)
from numereng.features.training.run_log import (
    initialize_run_log,  # noqa: F401
    log_error,
    log_info,
    log_stage,
    resolve_run_log_path,  # noqa: F401
)
from numereng.features.training.run_store import (
    compute_config_hash,  # noqa: F401
    compute_run_hash,  # noqa: F401
)
from numereng.features.training.strategies import (
    TrainingEnginePlan,
    TrainingProfile,
    resolve_training_engine,  # noqa: F401
)

_SIMPLE_PROFILE: TrainingProfile = "simple"
_PURGED_WALK_FORWARD_PROFILE: TrainingProfile = "purged_walk_forward"
_FULL_HISTORY_REFIT_PROFILE: TrainingProfile = "full_history_refit"

logger = logging.getLogger(__name__)
_POST_TRAINING_SCORING_STAGE_BY_POLICY: dict[PostTrainingScoringPolicy, CanonicalScoringStage | None] = {
    "none": None,
    "core": "post_training_core",
    "full": "post_training_full",
    "round_core": "post_training_core",
    "round_full": "post_training_full",
}


def resolve_model_config(model_config: dict[str, object]) -> tuple[str, dict[str, object]]:
    """Resolve required model type/params from model config block."""
    model_type = str(model_config.get("type", "LGBMRegressor"))
    model_params = model_config.get("params")
    if model_params is None:
        raise TrainingConfigError("training_model_params_missing")
    if not isinstance(model_params, dict):
        raise TrainingConfigError("training_model_params_not_mapping")
    resolved_params = dict(model_params)

    model_device_raw = model_config.get("device")
    if model_device_raw is not None:
        model_device = str(model_device_raw).strip().lower()
        if model_device not in {"cpu", "cuda"}:
            raise TrainingConfigError("training_model_device_invalid")
        if model_type != "LGBMRegressor":
            raise TrainingConfigError("training_model_device_requires_lgbm")
        raw_device_type = resolved_params.get("device_type")
        if raw_device_type is not None:
            device_type = str(raw_device_type).strip().lower()
            if device_type != model_device:
                raise TrainingConfigError("training_model_device_conflict")
        resolved_params["device_type"] = model_device

    raw_device_type = resolved_params.get("device_type")
    if raw_device_type is not None:
        device_type = str(raw_device_type).strip().lower()
        if device_type not in {"cpu", "gpu", "cuda"}:
            raise TrainingConfigError("training_model_params_device_type_invalid")
        if model_type != "LGBMRegressor":
            raise TrainingConfigError("training_model_device_requires_lgbm")
        resolved_params["device_type"] = device_type

    return model_type, resolved_params


def load_and_prepare_data(
    client: TrainingDataClient,
    data_version: str,
    dataset_variant: str,
    feature_set: str,
    target_col: str,
    era_col: str,
    id_col: str,
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
    dataset_scope: str,
    full_rows: int,
    full_eras: int,
    oof_rows: int,
    oof_eras: int,
    configured_embargo_eras: int | None,
    effective_embargo_eras: int,
    benchmark_source: BenchmarkSource,
    meta_model_col: str,
    meta_model_data_path: str | Path | None,
    output_dir: Path,
    predictions_relative: Path,
    score_provenance_relative: Path | None,
    summaries: dict[str, pd.DataFrame] | None,
    cv_meta: dict[str, object],
    engine_plan: TrainingEnginePlan,
    cv_enabled: bool,
    resource_policy: dict[str, object],
    cache_policy: dict[str, object],
    scoring_metadata: dict[str, object],
    scoring_policy: ResolvedScoringPolicy | None = None,
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
        "benchmark",
        "baseline",
    ):
        if key in model_config:
            model_meta[key] = model_config[key]

    metrics_payload: dict[str, object]
    if summaries is None:
        metrics_payload = metrics_status or {"status": "not_applicable"}
    else:
        metrics_payload = _metrics_payload_from_summaries(summaries)

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
            "dataset_scope": dataset_scope,
            "full_rows": full_rows,
            "full_eras": full_eras,
            "oof_rows": oof_rows,
            "oof_eras": oof_eras,
            "embargo_eras": effective_embargo_eras,
            "configured_embargo_eras": configured_embargo_eras,
            "effective_embargo_eras": effective_embargo_eras,
        },
        "benchmark": build_results_benchmark_payload(benchmark_source),
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
            "scoring": dict(scoring_metadata),
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


def build_results_benchmark_payload(benchmark_source: BenchmarkSource) -> dict[str, object]:
    """Build the canonical results payload benchmark block."""
    return {
        "mode": benchmark_source.mode,
        "name": benchmark_source.name,
        "file": str(benchmark_source.predictions_path),
        "pred_col": benchmark_source.pred_col,
        "metadata_file": str(benchmark_source.metadata_path) if benchmark_source.metadata_path else None,
    }


def resolve_post_training_scoring_policy(
    *,
    training_config: dict[str, object],
    override: PostTrainingScoringPolicy | None = None,
) -> PostTrainingScoringPolicy:
    """Resolve one post-training scoring policy from override or config."""
    if override is not None:
        return override

    raw_value = training_config.get("post_training_scoring")
    if raw_value is None:
        return "none"
    return cast(PostTrainingScoringPolicy, str(raw_value))


def resolve_post_training_scoring_policy_from_config(
    *,
    config_path: str | Path,
    override: PostTrainingScoringPolicy | None = None,
) -> PostTrainingScoringPolicy:
    """Resolve one post-training scoring policy directly from a config file."""
    if override is not None:
        return override
    config = load_config(Path(config_path).expanduser().resolve())
    training_config = _as_dict(config.get("training"))
    return resolve_post_training_scoring_policy(training_config=training_config)


def post_training_scoring_requested_stage(
    policy: PostTrainingScoringPolicy,
) -> CanonicalScoringStage | None:
    """Map one post-training scoring policy to the requested scoring stage."""
    return _POST_TRAINING_SCORING_STAGE_BY_POLICY[policy]


def is_round_post_training_scoring_policy(policy: PostTrainingScoringPolicy) -> bool:
    """Return whether one policy defers to experiment-round batch scoring."""
    return policy in {"round_core", "round_full"}


def run_training(
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
    """Run full training pipeline and return predictions/results artifact paths."""
    from numereng.features.training import _pipeline as pipeline_module

    # Preserve the long-standing service-module monkeypatch seam used by tests
    # and higher-level orchestrators while moving execution into staged helpers.
    for name in (
        "apply_missing_all_twos_as_nan",
        "attach_benchmark_predictions",
        "build_full_history_predictions",
        "build_lazy_parquet_data_loader",
        "build_model_data_loader",
        "build_oof_predictions",
        "build_results_payload",
        "build_x_cols",
        "compute_config_hash",
        "compute_run_hash",
        "create_training_data_client",
        "ensure_split_dataset_paths",
        "index_run",
        "initialize_run_log",
        "list_lazy_source_eras",
        "load_config",
        "load_custom_benchmark_predictions",
        "load_features",
        "load_fold_data_lazy",
        "load_full_data",
        "maybe_log_training_run",
        "normalize_x_groups",
        "is_round_post_training_scoring_policy",
        "post_training_scoring_requested_stage",
        "resolve_fold_lazy_source_paths",
        "resolve_metrics_path",
        "resolve_model_config",
        "resolve_output_locations",
        "resolve_post_training_scoring_policy",
        "resolve_resolved_config_path",
        "resolve_results_path",
        "resolve_run_manifest_path",
        "resolve_run_log_path",
        "resolve_score_provenance_path",
        "resolve_training_engine",
        "run_post_training_scoring",
        "run_scoring",
        "save_metrics",
        "save_predictions",
        "save_resolved_config",
        "save_results",
        "save_run_manifest",
        "save_score_provenance",
        "select_prediction_columns",
    ):
        setattr(pipeline_module, name, globals()[name])

    return pipeline_module.run_training_pipeline(
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
    )


def _record_telemetry_stage(
    session: LocalRunTelemetrySession | None,
    *,
    completed_stages: list[str],
    stage_name: str,
    message: str,
    run_log_path: Path | None,
    attempt_id: str,
    extra_payload: dict[str, object] | None = None,
) -> None:
    log_stage(run_log_path, stage_name=stage_name, message=message, attempt_id=attempt_id)

    if session is not None:
        try:
            emit_stage_event(
                session,
                current_stage=stage_name,
                completed_stages=list(completed_stages),
                extra_payload=extra_payload,
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


def _metrics_payload_from_summaries(summaries: dict[str, pd.DataFrame]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for metric_name, summary in summaries.items():
        if "prediction" in summary.index:
            payload[metric_name] = summary.loc["prediction"].to_dict()
            continue
        if len(summary.index) != 1:
            raise TrainingError(f"training_metric_row_missing_prediction:{metric_name}")
        payload[metric_name] = summary.iloc[0].to_dict()

    for metric_name in ("bmc", "bmc_last_200_eras", "mmc"):
        if isinstance(payload.get(metric_name), dict):
            continue
        alias_keys = _aliased_metric_keys(payload, metric_name)
        if len(alias_keys) == 1:
            payload[metric_name] = payload[alias_keys[0]]
    return payload


def _aliased_metric_keys(payload: dict[str, object], metric_name: str) -> list[str]:
    if metric_name == "bmc":
        return [key for key in payload if key.startswith("bmc_") and not key.startswith("bmc_last_200_eras_")]
    return [key for key in payload if key.startswith(f"{metric_name}_")]


def _mark_telemetry_completed(
    session: LocalRunTelemetrySession | None,
    *,
    run_id: str,
    run_dir: Path,
    run_log_path: Path | None,
    attempt_id: str,
) -> None:
    log_info(
        run_log_path,
        event="run_completed",
        message=f"run_id={run_id} run_dir={run_dir}",
        attempt_id=attempt_id,
    )
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


def _mark_telemetry_canceled(
    session: LocalRunTelemetrySession | None,
    *,
    run_id: str,
    message: str,
    run_log_path: Path | None,
    attempt_id: str,
    cancel_requested_at: str | None,
    terminal_reason: str,
    terminal_detail: dict[str, object] | None = None,
) -> None:
    log_error(
        run_log_path,
        event="run_canceled",
        message=f"run_id={run_id} reason={message}",
        attempt_id=attempt_id,
    )
    if session is not None:
        try:
            mark_job_canceled(
                session,
                terminal_reason=terminal_reason,
                terminal_detail={
                    "message": message,
                    "cancel_requested_at": cancel_requested_at,
                    **(terminal_detail or {}),
                },
            )
            append_log_line(session, stream="stderr", line=f"[telemetry] canceled run {run_id}: {message}")
        except Exception:
            logger.exception("failed to mark telemetry cancellation for run_id=%s", run_id)


def _mark_telemetry_failed(
    session: LocalRunTelemetrySession | None,
    *,
    run_id: str,
    error: dict[str, str],
    message: str,
    run_log_path: Path | None,
    attempt_id: str,
    terminal_reason: str = "failed",
    terminal_detail: dict[str, object] | None = None,
) -> None:
    log_error(
        run_log_path,
        event="run_failed",
        message=f"run_id={run_id} error={message}",
        attempt_id=attempt_id,
    )
    if session is not None:
        try:
            mark_job_failed(
                session,
                error=error,
                terminal_reason=terminal_reason,
                terminal_detail=terminal_detail,
            )
            append_log_line(session, stream="stderr", line=f"[telemetry] failed run {run_id}: {message}")
        except Exception:
            logger.exception("failed to mark telemetry failure for run_id=%s", run_id)


def _lifecycle_manifest_payload(
    *,
    terminal_reason: str,
    cancel_requested_at: str | None = None,
    terminal_detail: dict[str, object] | None = None,
    reconciled: bool = False,
) -> dict[str, object]:
    return {
        "terminal_reason": terminal_reason,
        "terminal_detail": terminal_detail or {},
        "cancel_requested_at": cancel_requested_at,
        "reconciled": reconciled,
    }


def _raise_if_cancel_requested(
    session: LocalRunTelemetrySession | None,
    *,
    stage_name: str,
) -> None:
    if session is None:
        return
    if not is_cancel_requested(session):
        return
    raise TrainingCanceledError(f"training_run_canceled:{stage_name}")


def _telemetry_metric_payload(metrics_payload: dict[str, object]) -> dict[str, object]:
    payload: dict[str, object] = {}
    corr_obj = metrics_payload.get("corr")
    if isinstance(corr_obj, dict):
        corr_mean = _coerce_finite_float(corr_obj.get("mean"))
        corr_sharpe = _coerce_finite_float(corr_obj.get("sharpe"))
        if corr_mean is not None:
            payload["corr_mean"] = corr_mean
        if corr_sharpe is not None:
            payload["corr_sharpe"] = corr_sharpe
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
    bmc_last_200_obj = metrics_payload.get("bmc_last_200_eras")
    if isinstance(bmc_last_200_obj, dict):
        bmc_last_200_mean = _coerce_finite_float(bmc_last_200_obj.get("mean"))
        if bmc_last_200_mean is not None:
            payload["bmc_last_200_eras_mean"] = bmc_last_200_mean
    feature_exposure_obj = metrics_payload.get("feature_exposure")
    if isinstance(feature_exposure_obj, dict):
        feature_exposure_mean = _coerce_finite_float(feature_exposure_obj.get("mean"))
        if feature_exposure_mean is not None:
            payload["feature_exposure_mean"] = feature_exposure_mean
    max_feature_exposure_obj = metrics_payload.get("max_feature_exposure")
    if isinstance(max_feature_exposure_obj, dict):
        max_feature_exposure_mean = _coerce_finite_float(max_feature_exposure_obj.get("mean"))
        if max_feature_exposure_mean is not None:
            payload["max_feature_exposure"] = max_feature_exposure_mean
    return payload


def _as_dict(value: object) -> dict[str, object]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    raise TrainingConfigError("training_config_section_not_mapping")


def _resolve_scoring_target_cols(*, data_config: dict[str, object], target_col: str) -> tuple[str, ...]:
    raw = data_config.get("scoring_targets")
    if raw is None:
        return tuple(_dedupe_preserve_order([target_col, "target_ender_20"]))
    if not isinstance(raw, list) or not raw:
        raise TrainingConfigError("training_scoring_targets_invalid")
    resolved = [str(item).strip() for item in raw]
    if any(not item for item in resolved):
        raise TrainingConfigError("training_scoring_targets_invalid")
    return tuple(_dedupe_preserve_order(resolved))


def _scoring_targets_explicit(*, data_config: dict[str, object]) -> bool:
    return data_config.get("scoring_targets") is not None


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _is_full_history_refit_profile(value: object) -> bool:
    return str(value) == _FULL_HISTORY_REFIT_PROFILE


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
) -> str:
    _ = dataset_variant
    if profile in {_PURGED_WALK_FORWARD_PROFILE, _FULL_HISTORY_REFIT_PROFILE}:
        return "train_plus_validation"
    if profile == _SIMPLE_PROFILE:
        if dataset_variant == "downsampled":
            raise TrainingConfigError("training_profile_simple_disallows_downsampled_dataset_variant")
        return "train_plus_validation"
    return configured_scope


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


def _resolve_benchmark_source_config(data_config: dict[str, object]) -> dict[str, object]:
    raw = data_config.get("benchmark_source")
    if raw is None:
        return {
            "source": "active",
            "pred_col": "prediction",
        }
    if not isinstance(raw, dict):
        raise TrainingConfigError("training_benchmark_source_invalid")
    source = str(raw.get("source", "active")).strip()
    if source not in {"active", "path"}:
        raise TrainingConfigError("training_benchmark_source_invalid")
    pred_col = str(raw.get("pred_col", "prediction")).strip()
    if not pred_col:
        raise TrainingConfigError("training_benchmark_source_pred_col_invalid")
    predictions_path = raw.get("predictions_path")
    if source == "path":
        if not isinstance(predictions_path, str) or not predictions_path.strip():
            raise TrainingConfigError("training_benchmark_source_predictions_path_required")
    elif predictions_path is not None:
        raise TrainingConfigError("training_benchmark_source_predictions_path_disallowed_for_active")
    resolved: dict[str, object] = {
        "source": source,
        "pred_col": pred_col,
    }
    name = raw.get("name")
    if isinstance(name, str) and name.strip():
        resolved["name"] = name.strip()
    if isinstance(predictions_path, str) and predictions_path.strip():
        resolved["predictions_path"] = predictions_path.strip()
    return resolved


def resolve_benchmark_source(
    *,
    data_config: dict[str, object],
    data_root: Path,
) -> BenchmarkSource:
    """Resolve the configured benchmark prediction source for scoring."""

    config = _resolve_benchmark_source_config(data_config)
    source = str(config["source"])
    pred_col = str(config["pred_col"])
    if source == "active":
        predictions_path = resolve_active_benchmark_predictions_path(data_root=data_root)
        metadata_path = resolve_active_benchmark_metadata_path(data_root=data_root)
        name = str(config.get("name") or "active_benchmark")
    else:
        predictions_path = Path(str(config["predictions_path"])).expanduser()
        if not predictions_path.is_absolute():
            predictions_path = (data_root.expanduser().resolve() / predictions_path).resolve()
        else:
            predictions_path = predictions_path.resolve()
        metadata_path = None
        name = str(config.get("name") or predictions_path.stem)
    return BenchmarkSource(
        mode=source,
        name=name,
        predictions_path=predictions_path,
        pred_col=pred_col,
        metadata_path=metadata_path,
    )


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
    sched_getaffinity = getattr(os, "sched_getaffinity", None)
    try:
        affinity = sched_getaffinity(0) if callable(sched_getaffinity) else None
    except OSError:
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


def _scoring_policy_payload(policy: ResolvedScoringPolicy) -> dict[str, object]:
    return {
        "fnc_feature_set": policy.fnc_feature_set,
        "fnc_target_policy": policy.fnc_target_policy,
        "benchmark_min_overlap_ratio": policy.benchmark_min_overlap_ratio,
    }


def _coerce_optional_int(value: object) -> int | None:
    if value is None:
        return None
    return _coerce_int(value, default=0)


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


def _ensure_run_dir_is_fresh(run_dir: Path) -> None:
    """Require no pre-existing artifacts under deterministic run directory."""
    if not run_dir.exists():
        return
    if not run_dir.is_dir():
        raise TrainingError("training_run_dir_not_directory")
    preexisting_entries = [entry.name for entry in run_dir.iterdir() if entry.name != RUN_LOCK_FILENAME]
    if not preexisting_entries:
        return
    preexisting_entries.sort()
    raise TrainingError(
        f"training_run_dir_not_fresh:{run_dir.name}:preexisting={','.join(preexisting_entries)}:reset_required"
    )


def _era_sort_key(era: object) -> int | str:
    if isinstance(era, int):
        return era
    if isinstance(era, str) and era.isdigit():
        return int(era)
    return str(era)
