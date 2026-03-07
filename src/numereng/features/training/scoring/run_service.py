"""Score persisted run predictions using canonical post-training scoring."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import cast

import pandas as pd

from numereng.features.store import StoreError, index_run, resolve_store_root
from numereng.features.training.client import create_training_data_client
from numereng.features.training.errors import TrainingConfigError, TrainingError
from numereng.features.training.models import ScoreRunResult
from numereng.features.training.repo import (
    DEFAULT_DATASETS_DIR,
    resolve_metrics_path,
    resolve_run_manifest_path,
    resolve_score_provenance_path,
    save_metrics,
    save_results,
    save_run_manifest,
    save_score_provenance,
)
from numereng.features.training.run_log import log_error, log_info, resolve_run_log_path
from numereng.features.training.scoring.models import (
    PostTrainingScoringRequest,
    ResolvedScoringPolicy,
)
from numereng.features.training.scoring.service import run_post_training_scoring

_SAFE_ID = re.compile(r"^[\w\-.]+$")
_DEFAULT_DATASET_VARIANT = "non_downsampled"
_DEFAULT_DATASET_SCOPE = "train_plus_validation"
_DEFAULT_SCORING_MODE = "materialized"
_CLASSIC_PAYOUT_TARGET = "target_ender_20"
_PAYOUT_CORR_WEIGHT = 0.75
_PAYOUT_MMC_WEIGHT = 2.25
_PAYOUT_CLIP = 0.05


def score_run(
    *,
    run_id: str,
    store_root: str | Path = ".numereng",
) -> ScoreRunResult:
    """Recompute metrics/provenance for one persisted training run."""
    safe_run_id = _ensure_safe_run_id(run_id)
    root = resolve_store_root(store_root)
    run_dir = root / "runs" / safe_run_id
    run_log_path = resolve_run_log_path(run_dir)

    if not run_dir.is_dir():
        raise TrainingError(f"training_score_run_not_found:{safe_run_id}")

    run_manifest_path = resolve_run_manifest_path(run_dir)
    run_manifest = _load_required_json_mapping(
        run_manifest_path,
        missing_code="training_score_run_manifest_not_found",
        invalid_code="training_score_run_manifest_invalid",
    )
    resolved_path = run_dir / "resolved.json"
    resolved = _load_required_json_mapping(
        resolved_path,
        missing_code="training_score_run_resolved_not_found",
        invalid_code="training_score_run_resolved_invalid",
    )
    results_path = run_dir / "results.json"
    results = _load_required_json_mapping(
        results_path,
        missing_code="training_score_run_results_not_found",
        invalid_code="training_score_run_results_invalid",
    )

    predictions_path = _resolve_predictions_path(run_dir, run_manifest, resolved)
    if not predictions_path.is_file():
        raise TrainingError(f"training_score_predictions_not_found:{predictions_path}")

    data_config = _as_mapping(resolved.get("data"))
    loading_config = _as_mapping(data_config.get("loading"))

    scoring_mode = _resolve_scoring_mode(loading_config.get("scoring_mode"))
    era_chunk_size = _resolve_era_chunk_size(loading_config.get("era_chunk_size"), default=64)
    if era_chunk_size < 1:
        raise TrainingConfigError("training_data_loading_era_chunk_size_invalid")

    target_col = _coerce_required_str(
        value=data_config.get("target_col") or _as_mapping(run_manifest.get("data")).get("target_col"),
        error_code="training_score_target_col_missing",
    )
    data_version = _coerce_required_str(
        value=data_config.get("data_version") or _as_mapping(run_manifest.get("data")).get("version"),
        error_code="training_score_data_version_missing",
    )
    feature_set = _coerce_required_str(
        value=data_config.get("feature_set") or _as_mapping(run_manifest.get("data")).get("feature_set"),
        error_code="training_score_feature_set_missing",
    )

    request = PostTrainingScoringRequest(
        predictions_path=predictions_path,
        pred_cols=("prediction",),
        target_col=target_col,
        data_version=data_version,
        dataset_variant=str(data_config.get("dataset_variant", _DEFAULT_DATASET_VARIANT)),
        feature_set=feature_set,
        feature_source_paths=None,
        full_data_path=_optional_path(data_config.get("full_data_path")),
        dataset_scope=str(data_config.get("dataset_scope", _DEFAULT_DATASET_SCOPE)),
        benchmark_model=str(data_config.get("benchmark_model", "v52_lgbm_ender20")),
        benchmark_data_path=_optional_path(data_config.get("benchmark_data_path")),
        meta_model_col=str(data_config.get("meta_model_col", "numerai_meta_model")),
        meta_model_data_path=_optional_path(data_config.get("meta_model_data_path")),
        era_col=str(data_config.get("era_col", "era")),
        id_col=str(data_config.get("id_col", "id")),
        data_root=DEFAULT_DATASETS_DIR,
        scoring_mode=scoring_mode,
        era_chunk_size=era_chunk_size,
    )

    log_info(
        run_log_path,
        event="run_scoring_started",
        message=f"mode={scoring_mode} era_chunk_size={era_chunk_size}",
    )

    try:
        scoring_result = run_post_training_scoring(
            request=request,
            client=create_training_data_client(),
        )
    except Exception as exc:
        log_error(run_log_path, event="run_scoring_failed", message=str(exc))
        raise

    metrics_payload = _metrics_payload_from_summaries(
        scoring_result.summaries,
        target_col=target_col,
    )
    metrics_path = resolve_metrics_path(run_dir)
    score_provenance_path = resolve_score_provenance_path(run_dir)
    score_provenance_relative = score_provenance_path.relative_to(run_dir)
    predictions_relative = predictions_path.relative_to(run_dir)

    save_score_provenance(scoring_result.score_provenance, score_provenance_path)
    save_metrics(metrics_payload, metrics_path)

    results_output = _as_mapping(results.get("output"))
    results_output["output_dir"] = str(run_dir)
    results_output["predictions_file"] = str(predictions_relative)
    results_output["score_provenance_file"] = str(score_provenance_relative)
    results["output"] = results_output
    results["metrics"] = metrics_payload
    results_training = _as_mapping(results.get("training"))
    results_training["scoring"] = _updated_scoring_payload(
        existing=_as_mapping(results_training.get("scoring")),
        requested_mode=scoring_mode,
        requested_chunk_size=era_chunk_size,
        effective_backend=scoring_result.effective_scoring_backend,
        policy=scoring_result.policy,
    )
    results["training"] = results_training
    save_results(results, results_path)

    artifacts = _as_mapping(run_manifest.get("artifacts"))
    artifacts["predictions"] = str(predictions_relative)
    artifacts["results"] = "results.json"
    artifacts["metrics"] = "metrics.json"
    artifacts["score_provenance"] = str(score_provenance_relative)
    run_manifest["artifacts"] = artifacts
    run_manifest["metrics_summary"] = metrics_payload
    manifest_training = _as_mapping(run_manifest.get("training"))
    manifest_training["scoring"] = _updated_scoring_payload(
        existing=_as_mapping(manifest_training.get("scoring")),
        requested_mode=scoring_mode,
        requested_chunk_size=era_chunk_size,
        effective_backend=scoring_result.effective_scoring_backend,
        policy=scoring_result.policy,
    )
    run_manifest["training"] = manifest_training
    save_run_manifest(run_manifest, run_manifest_path)

    try:
        index_run(store_root=root, run_id=safe_run_id)
    except StoreError as exc:
        raise TrainingError(f"training_score_store_index_failed:{safe_run_id}") from exc

    log_info(
        run_log_path,
        event="run_scoring_succeeded",
        message=f"effective_backend={scoring_result.effective_scoring_backend}",
    )

    return ScoreRunResult(
        run_id=safe_run_id,
        predictions_path=predictions_path,
        results_path=results_path,
        metrics_path=metrics_path,
        score_provenance_path=score_provenance_path,
        effective_scoring_backend=scoring_result.effective_scoring_backend,
    )


def _ensure_safe_run_id(run_id: str) -> str:
    candidate = run_id.strip()
    if not candidate or not _SAFE_ID.match(candidate):
        raise TrainingError(f"training_score_run_id_invalid:{run_id}")
    return candidate


def _load_required_json_mapping(path: Path, *, missing_code: str, invalid_code: str) -> dict[str, object]:
    if not path.is_file():
        raise TrainingError(f"{missing_code}:{path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingError(f"{invalid_code}:{path}") from exc
    if not isinstance(payload, dict):
        raise TrainingError(f"{invalid_code}:{path}")
    return cast(dict[str, object], payload)


def _resolve_predictions_path(
    run_dir: Path,
    run_manifest: dict[str, object],
    resolved: dict[str, object],
) -> Path:
    artifacts = _as_mapping(run_manifest.get("artifacts"))
    predictions_rel = artifacts.get("predictions")
    if isinstance(predictions_rel, str) and predictions_rel.strip():
        return (run_dir / predictions_rel).resolve()

    predictions_dir = run_dir / "artifacts" / "predictions"
    parquet_files = sorted(predictions_dir.glob("*.parquet"))
    if len(parquet_files) == 1:
        return parquet_files[0].resolve()

    output = _as_mapping(resolved.get("output"))
    predictions_name = output.get("predictions_name")
    if isinstance(predictions_name, str) and predictions_name.strip():
        candidate = predictions_dir / f"{predictions_name}.parquet"
        if candidate.exists():
            return candidate.resolve()

    raise TrainingError(f"training_score_predictions_path_missing:{run_dir}")


def _metrics_payload_from_summaries(
    summaries: dict[str, pd.DataFrame],
    *,
    target_col: str,
) -> dict[str, object]:
    metrics_corr = summaries["corr"].loc["prediction"].to_dict()
    metrics_fnc = summaries["fnc"].loc["prediction"].to_dict()
    metrics_mmc = summaries["mmc"].loc["prediction"].to_dict()
    metrics_cwmm = summaries["cwmm"].loc["prediction"].to_dict()
    metrics_bmc = summaries["bmc"].loc["prediction"].to_dict()
    metrics_bmc_200 = summaries["bmc_last_200_eras"].loc["prediction"].to_dict()
    payout_estimate_mean = (
        _clip_payout_estimate_mean(
            corr_mean=_coerce_finite_float(metrics_corr.get("mean")),
            mmc_mean=_coerce_finite_float(metrics_mmc.get("mean")),
        )
        if _is_classic_payout_target(target_col)
        else None
    )
    return {
        "corr": metrics_corr,
        "fnc": metrics_fnc,
        "mmc": metrics_mmc,
        "cwmm": metrics_cwmm,
        "bmc": metrics_bmc,
        "bmc_last_200_eras": metrics_bmc_200,
        "payout_estimate": {"mean": payout_estimate_mean},
        "payout_estimate_mean": payout_estimate_mean,
    }


def _updated_scoring_payload(
    *,
    existing: dict[str, object],
    requested_mode: str,
    requested_chunk_size: int,
    effective_backend: str,
    policy: ResolvedScoringPolicy,
) -> dict[str, object]:
    payload = dict(existing)
    payload["mode"] = requested_mode
    payload["era_chunk_size"] = requested_chunk_size
    payload["effective_backend"] = effective_backend
    payload["policy"] = {
        "fnc_feature_set": policy.fnc_feature_set,
        "benchmark_overlap_policy": policy.benchmark_overlap_policy,
        "meta_overlap_policy": policy.meta_overlap_policy,
    }
    return payload


def _clip_payout_estimate_mean(*, corr_mean: float | None, mmc_mean: float | None) -> float | None:
    if corr_mean is None or mmc_mean is None:
        return None
    payout_mean = (_PAYOUT_CORR_WEIGHT * corr_mean) + (_PAYOUT_MMC_WEIGHT * mmc_mean)
    return max(min(payout_mean, _PAYOUT_CLIP), -_PAYOUT_CLIP)


def _is_classic_payout_target(target_col: str) -> bool:
    return target_col == _CLASSIC_PAYOUT_TARGET


def _coerce_finite_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        casted = float(value)
        return casted if pd.notna(casted) else None
    return None


def _resolve_scoring_mode(value: object) -> str:
    if value is None:
        return _DEFAULT_SCORING_MODE
    if isinstance(value, str) and value in {"materialized", "era_stream"}:
        return value
    raise TrainingConfigError("training_data_loading_scoring_mode_invalid")


def _resolve_era_chunk_size(value: object, *, default: int) -> int:
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


def _as_mapping(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return cast(dict[str, object], value)
    return {}


def _optional_path(value: object) -> str | Path | None:
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        return value
    raise TrainingConfigError("training_config_path_value_invalid")


def _coerce_required_str(*, value: object, error_code: str) -> str:
    if isinstance(value, str):
        candidate = value.strip()
        if candidate:
            return candidate
    raise TrainingError(error_code)
