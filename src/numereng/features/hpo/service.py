"""HPO orchestration service built on top of training + store features."""

from __future__ import annotations

import json
import re
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from numereng.config.training import TrainingConfigLoaderError, ensure_json_config_path, load_training_config_json
from numereng.features.feature_neutralization import (
    NeutralizationDataError,
    NeutralizationValidationError,
    NeutralizePredictionsRequest,
    load_neutralizer_table,
    neutralize_predictions_file,
)
from numereng.features.hpo.artifacts import (
    build_study_id,
    resolve_study_storage_path,
    write_trial_config,
    write_trials_table,
)
from numereng.features.hpo.contracts import (
    HpoDirection,
    HpoStatus,
    HpoStudyCreateRequest,
    HpoStudyRecord,
    HpoStudyResult,
    HpoTrialRecord,
    HpoTrialResult,
)
from numereng.features.hpo.repo import get_study, list_studies, list_trials, save_study, save_trial
from numereng.features.hpo.runner_optuna import HpoOptunaError, run_optuna_study
from numereng.features.hpo.search_space import HpoSearchSpaceError, apply_param_overrides, resolve_search_space
from numereng.features.scoring.metrics import (
    DEFAULT_META_MODEL_COL,
)
from numereng.features.scoring.models import PostTrainingScoringRequest
from numereng.features.scoring.service import run_post_training_scoring
from numereng.features.store import StoreError, index_run, resolve_store_root
from numereng.features.telemetry import bind_launch_metadata, get_launch_metadata
from numereng.features.training import TrainingRunResult, run_training
from numereng.features.training.client import TrainingDataClient, create_training_data_client
from numereng.features.training.repo import DEFAULT_DATASETS_DIR
from numereng.features.training.service import resolve_benchmark_source

_SAFE_ID = re.compile(r"^[\w\-.]+$")
_DEFAULT_HPO_METRIC = "post_fold_champion_objective"
_POST_FOLD_CORR_WEIGHT = 0.25
_POST_FOLD_BMC_WEIGHT = 2.25


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
    """Create and execute one Optuna-backed HPO study."""

    if not request.study_name.strip():
        raise HpoValidationError("hpo_study_name_invalid")
    if request.n_trials < 1:
        raise HpoValidationError("hpo_n_trials_invalid")
    if request.neutralization_proportion < 0.0 or request.neutralization_proportion > 1.0:
        raise HpoValidationError("hpo_neutralization_proportion_invalid")
    if request.neutralization_mode not in {"era", "global"}:
        raise HpoValidationError("hpo_neutralization_mode_invalid")
    if request.neutralize and request.neutralizer_path is None:
        raise HpoValidationError("hpo_neutralizer_path_required")
    if request.experiment_id is not None and not _SAFE_ID.match(request.experiment_id):
        raise HpoValidationError(f"hpo_experiment_id_invalid:{request.experiment_id}")

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
        param_specs = resolve_search_space(base_config=base_config, raw_search_space=request.search_space)
    except HpoSearchSpaceError as exc:
        raise HpoValidationError(str(exc)) from exc

    resolved_neutralizer_cols: tuple[str, ...] | None = None
    if request.neutralize:
        if request.neutralizer_path is None:  # pragma: no cover - guarded by validation
            raise HpoValidationError("hpo_neutralizer_path_required")
        try:
            _, resolved_neutralizer_cols = load_neutralizer_table(
                neutralizer_path=request.neutralizer_path,
                neutralizer_cols=request.neutralizer_cols,
            )
        except (NeutralizationValidationError, NeutralizationDataError) as exc:
            raise HpoValidationError(str(exc)) from exc

    resolved_root = resolve_store_root(store_root)
    study_id = build_study_id(study_name=request.study_name, experiment_id=request.experiment_id)
    storage_path = resolve_study_storage_path(
        store_root=resolved_root,
        experiment_id=request.experiment_id,
        study_id=study_id,
    )

    initial_time = _utc_now_iso()
    result_snapshot = HpoStudyResult(
        study_id=study_id,
        study_name=request.study_name,
        experiment_id=request.experiment_id,
        status="running",
        metric=request.metric,
        direction=request.direction,
        n_trials=request.n_trials,
        sampler=request.sampler,
        seed=request.seed,
        best_trial_number=None,
        best_value=None,
        best_run_id=None,
        storage_path=storage_path,
        config={
            "config_path": str(config_path),
            "search_space": [
                {
                    "path": spec.path,
                    "kind": spec.kind,
                    "low": spec.low,
                    "high": spec.high,
                    "step": spec.step,
                    "log": spec.log,
                    "choices": list(spec.choices),
                }
                for spec in param_specs
            ],
            "neutralization": {
                "enabled": request.neutralize,
                "neutralizer_path": str(request.neutralizer_path) if request.neutralizer_path else None,
                "proportion": request.neutralization_proportion,
                "mode": request.neutralization_mode,
                "neutralizer_cols": (
                    list(resolved_neutralizer_cols)
                    if resolved_neutralizer_cols is not None
                    else (list(request.neutralizer_cols) if request.neutralizer_cols is not None else None)
                ),
                "rank_output": request.neutralization_rank_output,
            },
        },
        trials=(),
        created_at=initial_time,
        updated_at=initial_time,
    )
    save_study(store_root=resolved_root, payload=result_snapshot)

    trial_rows: list[dict[str, Any]] = []
    trial_results: list[HpoTrialResult] = []
    neutralized_metric_client = create_training_data_client() if request.neutralize else None

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
            if request.neutralize:
                if request.neutralizer_path is None:
                    raise HpoValidationError("hpo_neutralizer_path_required")
                neutralized_result = neutralize_predictions_file(
                    request=NeutralizePredictionsRequest(
                        predictions_path=training_result.predictions_path,
                        neutralizer_path=request.neutralizer_path,
                        proportion=request.neutralization_proportion,
                        mode=request.neutralization_mode,
                        neutralizer_cols=request.neutralizer_cols,
                        rank_output=request.neutralization_rank_output,
                    ),
                    run_id=training_result.run_id,
                )
                if neutralized_metric_client is None:  # pragma: no cover - defensive gate
                    raise HpoExecutionError("hpo_neutralization_client_missing")
                metric_value = _extract_metric_value_from_predictions(
                    predictions_path=neutralized_result.output_path,
                    trial_config_path=trial_config_path,
                    metric=request.metric,
                    scoring_client=neutralized_metric_client,
                )
            elif request.metric == _DEFAULT_HPO_METRIC:
                metric_value = _extract_post_fold_objective_value(
                    store_root=resolved_root,
                    run_id=training_result.run_id,
                    results_path=training_result.results_path,
                )
            else:
                metric_value = _extract_metric_value(results_path=training_result.results_path, metric=request.metric)
            finished_at = _utc_now_iso()
            trial_payload = HpoTrialResult(
                study_id=study_id,
                trial_number=trial_number,
                status="completed",
                params=params,
                value=metric_value,
                run_id=training_result.run_id,
                config_path=trial_config_path,
                started_at=started_at,
                finished_at=finished_at,
                error_message=None,
            )
            save_trial(store_root=resolved_root, payload=trial_payload)
            trial_results.append(trial_payload)
            trial_rows.append(
                {
                    "trial_number": trial_number,
                    "status": "completed",
                    "value": metric_value,
                    "run_id": training_result.run_id,
                    "params": json.dumps(params, sort_keys=True, ensure_ascii=True),
                    "started_at": started_at,
                    "finished_at": finished_at,
                }
            )
            return metric_value
        except StoreError as exc:
            finished_at = _utc_now_iso()
            trial_payload = HpoTrialResult(
                study_id=study_id,
                trial_number=trial_number,
                status="failed",
                params=params,
                value=None,
                run_id=training_result.run_id if training_result is not None else None,
                config_path=trial_config_path,
                started_at=started_at,
                finished_at=finished_at,
                error_message=f"hpo_trial_index_failed:{trial_number}",
            )
            save_trial(store_root=resolved_root, payload=trial_payload)
            trial_results.append(trial_payload)
            trial_rows.append(
                {
                    "trial_number": trial_number,
                    "status": "failed",
                    "value": None,
                    "run_id": training_result.run_id if training_result is not None else None,
                    "params": json.dumps(params, sort_keys=True, ensure_ascii=True),
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "error_message": f"hpo_trial_index_failed:{trial_number}",
                }
            )
            raise HpoExecutionError(f"hpo_trial_index_failed:{trial_number}") from exc
        except Exception as exc:
            finished_at = _utc_now_iso()
            trial_payload = HpoTrialResult(
                study_id=study_id,
                trial_number=trial_number,
                status="failed",
                params=params,
                value=None,
                run_id=training_result.run_id if training_result is not None else None,
                config_path=trial_config_path,
                started_at=started_at,
                finished_at=finished_at,
                error_message=str(exc),
            )
            save_trial(store_root=resolved_root, payload=trial_payload)
            trial_results.append(trial_payload)
            trial_rows.append(
                {
                    "trial_number": trial_number,
                    "status": "failed",
                    "value": None,
                    "run_id": training_result.run_id if training_result is not None else None,
                    "params": json.dumps(params, sort_keys=True, ensure_ascii=True),
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "error_message": str(exc),
                }
            )
            raise

    try:
        best_trial_number, best_value = run_optuna_study(
            direction=request.direction,
            n_trials=request.n_trials,
            sampler=request.sampler,
            seed=request.seed,
            specs=param_specs,
            objective_callback=objective_callback,
        )
    except HpoOptunaError as exc:
        failed_time = _utc_now_iso()
        failed_result = HpoStudyResult(
            study_id=study_id,
            study_name=request.study_name,
            experiment_id=request.experiment_id,
            status="failed",
            metric=request.metric,
            direction=request.direction,
            n_trials=request.n_trials,
            sampler=request.sampler,
            seed=request.seed,
            best_trial_number=None,
            best_value=None,
            best_run_id=None,
            storage_path=storage_path,
            config=result_snapshot.config,
            trials=tuple(trial_results),
            created_at=initial_time,
            updated_at=failed_time,
        )
        save_study(store_root=resolved_root, payload=failed_result, error_message=str(exc))
        write_trials_table(storage_path=storage_path, trials=trial_rows)
        raise HpoDependencyError(str(exc)) from exc

    completed_trials = [trial for trial in trial_results if trial.status == "completed" and trial.value is not None]
    status: HpoStatus
    if best_trial_number is None or best_value is None:
        if not completed_trials:
            status = "failed"
            best_run_id = None
            error_message = "hpo_trials_all_failed"
        else:
            winner = _best_trial_from_trials(completed_trials, direction=request.direction)
            best_trial_number = winner.trial_number
            best_value = winner.value
            best_run_id = winner.run_id
            status = "completed"
            error_message = None
    else:
        matched_winner = next((trial for trial in completed_trials if trial.trial_number == best_trial_number), None)
        best_run_id = matched_winner.run_id if matched_winner else None
        status = "completed"
        error_message = None

    updated_at = _utc_now_iso()
    final_result = HpoStudyResult(
        study_id=study_id,
        study_name=request.study_name,
        experiment_id=request.experiment_id,
        status=status,
        metric=request.metric,
        direction=request.direction,
        n_trials=request.n_trials,
        sampler=request.sampler,
        seed=request.seed,
        best_trial_number=best_trial_number,
        best_value=best_value,
        best_run_id=best_run_id,
        storage_path=storage_path,
        config=result_snapshot.config,
        trials=tuple(trial_results),
        created_at=initial_time,
        updated_at=updated_at,
    )
    save_study(store_root=resolved_root, payload=final_result, error_message=error_message)
    write_trials_table(storage_path=storage_path, trials=trial_rows)

    return final_result


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


def _best_trial_from_trials(trials: list[HpoTrialResult], *, direction: HpoDirection) -> HpoTrialResult:
    if direction == "minimize":
        return min(trials, key=lambda item: float(item.value if item.value is not None else float("inf")))
    return max(trials, key=lambda item: float(item.value if item.value is not None else float("-inf")))


def _extract_metric_value(*, results_path: Path, metric: str) -> float:
    if not results_path.exists():
        raise HpoExecutionError(f"hpo_results_not_found:{results_path}")

    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HpoExecutionError("hpo_results_invalid") from exc

    value = _metric_lookup(payload, metric)
    if value is None and not metric.startswith("metrics."):
        value = _metric_lookup(payload, f"metrics.{metric}")
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
    if not snapshot_path.exists():
        if results_path is None:
            raise HpoExecutionError(f"hpo_post_fold_snapshots_not_found:{snapshot_path}")
        return _extract_default_hpo_metric_from_results(results_path=results_path)
    try:
        snapshot = pd.read_parquet(snapshot_path)
    except Exception as exc:
        raise HpoExecutionError(f"hpo_post_fold_snapshots_invalid:{snapshot_path}") from exc
    if snapshot.empty:
        raise HpoExecutionError(f"hpo_post_fold_snapshots_empty:{snapshot_path}")
    if "corr_ender20_fold_mean" not in snapshot.columns:
        raise HpoExecutionError("hpo_post_fold_metric_missing:corr_ender20_fold_mean")
    bmc_column = "bmc_fold_mean" if "bmc_fold_mean" in snapshot.columns else "bmc_ender20_fold_mean"
    if bmc_column not in snapshot.columns:
        raise HpoExecutionError("hpo_post_fold_metric_missing:bmc_fold_mean,bmc_ender20_fold_mean")

    corr_mean = float(pd.Series(snapshot["corr_ender20_fold_mean"], dtype="float64").dropna().mean())
    bmc_mean = float(pd.Series(snapshot[bmc_column], dtype="float64").dropna().mean())
    if pd.isna(corr_mean) or pd.isna(bmc_mean):
        raise HpoExecutionError("hpo_post_fold_metric_nan")
    return (_POST_FOLD_CORR_WEIGHT * corr_mean) + (_POST_FOLD_BMC_WEIGHT * bmc_mean)


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
    loading_config = _as_mapping(data_config.get("loading"))
    scoring_mode = str(loading_config.get("scoring_mode", "materialized"))
    era_chunk_size_obj = loading_config.get("era_chunk_size", 64)
    if not isinstance(era_chunk_size_obj, int):
        raise HpoExecutionError("hpo_trial_scoring_chunk_size_invalid")
    era_chunk_size = era_chunk_size_obj

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
                scoring_mode=scoring_mode,
                era_chunk_size=era_chunk_size,
            ),
            client=scoring_client,
        )
    except Exception as exc:
        raise HpoExecutionError(f"hpo_neutralized_metric_compute_failed:{exc}") from exc

    metrics_payload = _metrics_payload_from_summaries(summaries=scoring_result.summaries)
    if metric == _DEFAULT_HPO_METRIC:
        value = _default_hpo_metric_from_payload(metrics_payload)
    else:
        value = _metric_lookup({"metrics": metrics_payload}, metric)
        if value is None and not metric.startswith("metrics."):
            value = _metric_lookup({"metrics": metrics_payload}, f"metrics.{metric}")
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


def _extract_default_hpo_metric_from_results(*, results_path: Path) -> float:
    if not results_path.exists():
        raise HpoExecutionError(f"hpo_results_not_found:{results_path}")

    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HpoExecutionError("hpo_results_invalid") from exc

    metrics_payload = payload.get("metrics")
    if not isinstance(metrics_payload, dict):
        raise HpoExecutionError(f"hpo_metric_not_found:{_DEFAULT_HPO_METRIC}")

    value = _default_hpo_metric_from_payload(metrics_payload)
    if value is None:
        raise HpoExecutionError(f"hpo_metric_not_found:{_DEFAULT_HPO_METRIC}")
    return value


def _default_hpo_metric_from_payload(metrics_payload: dict[str, Any]) -> float | None:
    payload = {"metrics": metrics_payload}
    corr_value = _metric_lookup(payload, "metrics.corr.mean")
    bmc_value = _metric_lookup(payload, "metrics.bmc_last_200_eras.mean")
    if bmc_value is None:
        bmc_value = _metric_lookup(payload, "metrics.bmc.mean")
    if bmc_value is None:
        bmc_value = _metric_lookup(payload, _aliased_metric_path(metrics_payload, "bmc_last_200_eras"))
    if bmc_value is None:
        bmc_value = _metric_lookup(payload, _aliased_metric_path(metrics_payload, "bmc"))

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


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()
