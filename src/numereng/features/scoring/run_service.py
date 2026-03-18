"""Score persisted run predictions using canonical run scoring."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import cast

import pandas as pd

from numereng.features.scoring.models import (
    CanonicalScoringStage,
    PostTrainingScoringRequest,
    ResolvedScoringPolicy,
)
from numereng.features.scoring.service import run_scoring
from numereng.features.store import StoreError, index_run, resolve_store_root
from numereng.features.training.client import TrainingDataClient, create_training_data_client
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
    save_scoring_artifacts,
)
from numereng.features.training.run_log import log_error, log_info, resolve_run_log_path
from numereng.features.training.service import resolve_benchmark_source

_SAFE_ID = re.compile(r"^[\w\-.]+$")
_DEFAULT_DATASET_VARIANT = "non_downsampled"
_DEFAULT_DATASET_SCOPE = "train_plus_validation"
_DEFAULT_SCORING_MODE = "materialized"


def score_run(
    *,
    run_id: str,
    store_root: str | Path = ".numereng",
    stage: CanonicalScoringStage = "all",
    client: TrainingDataClient | None = None,
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
        run_id=safe_run_id,
        config_hash=str(_as_mapping(run_manifest.get("config")).get("hash", "")),
        seed=None,
        predictions_path=predictions_path,
        pred_cols=("prediction",),
        target_col=target_col,
        scoring_target_cols=_resolve_scoring_target_cols(data_config=data_config, target_col=target_col),
        data_version=data_version,
        dataset_variant=str(data_config.get("dataset_variant", _DEFAULT_DATASET_VARIANT)),
        feature_set=feature_set,
        feature_source_paths=None,
        dataset_scope=str(data_config.get("dataset_scope", _DEFAULT_DATASET_SCOPE)),
        benchmark_source=resolve_benchmark_source(data_config=data_config, data_root=DEFAULT_DATASETS_DIR),
        meta_model_col=str(data_config.get("meta_model_col", "numerai_meta_model")),
        meta_model_data_path=_optional_path(data_config.get("meta_model_data_path")),
        era_col=str(data_config.get("era_col", "era")),
        id_col=str(data_config.get("id_col", "id")),
        data_root=DEFAULT_DATASETS_DIR,
        scoring_mode=scoring_mode,
        era_chunk_size=era_chunk_size,
        stage=stage,
        include_feature_neutral_metrics=_include_feature_neutral_metrics(stage),
    )

    log_info(
        run_log_path,
        event="run_scoring_started",
        message=f"mode={scoring_mode} era_chunk_size={era_chunk_size} stage={stage}",
    )

    try:
        scoring_result = run_scoring(
            request=request,
            client=client or create_training_data_client(),
        )
    except Exception as exc:
        log_error(run_log_path, event="run_scoring_failed", message=str(exc))
        raise

    metrics_payload = _metrics_payload_from_summaries(
        scoring_result.summaries,
    )
    metrics_path = resolve_metrics_path(run_dir)
    score_provenance_path = resolve_score_provenance_path(run_dir)
    score_provenance_relative = score_provenance_path.relative_to(run_dir)
    predictions_relative = predictions_path.relative_to(run_dir)
    scoring_dir = run_dir / "artifacts" / "scoring"

    persisted_scoring = save_scoring_artifacts(
        scoring_result.artifacts,
        scoring_dir=scoring_dir,
        output_dir=run_dir,
        selected_stage=scoring_result.requested_stage,
    )
    _refresh_persisted_artifacts_provenance(
        score_provenance=scoring_result.score_provenance,
        persisted_scoring=persisted_scoring,
    )
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
        requested_stage=scoring_result.requested_stage,
        refreshed_stages=scoring_result.refreshed_stages,
    )
    results["training"] = results_training
    save_results(results, results_path)

    artifacts = _as_mapping(run_manifest.get("artifacts"))
    artifacts["predictions"] = str(predictions_relative)
    artifacts["results"] = "results.json"
    artifacts["metrics"] = "metrics.json"
    artifacts["score_provenance"] = str(score_provenance_relative)
    artifacts["scoring_manifest"] = str(persisted_scoring.manifest_relative)
    run_manifest["artifacts"] = artifacts
    run_manifest["metrics_summary"] = metrics_payload
    manifest_training = _as_mapping(run_manifest.get("training"))
    manifest_training["scoring"] = _updated_scoring_payload(
        existing=_as_mapping(manifest_training.get("scoring")),
        requested_mode=scoring_mode,
        requested_chunk_size=era_chunk_size,
        effective_backend=scoring_result.effective_scoring_backend,
        policy=scoring_result.policy,
        requested_stage=scoring_result.requested_stage,
        refreshed_stages=scoring_result.refreshed_stages,
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
        message=(
            f"effective_backend={scoring_result.effective_scoring_backend} "
            f"requested_stage={scoring_result.requested_stage}"
        ),
    )

    return ScoreRunResult(
        run_id=safe_run_id,
        predictions_path=predictions_path,
        results_path=results_path,
        metrics_path=metrics_path,
        score_provenance_path=score_provenance_path,
        effective_scoring_backend=scoring_result.effective_scoring_backend,
        requested_stage=scoring_result.requested_stage,
        refreshed_stages=scoring_result.refreshed_stages,
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
) -> dict[str, object]:
    payload: dict[str, object] = {}
    for metric_name, summary in summaries.items():
        if "prediction" in summary.index:
            payload[metric_name] = summary.loc["prediction"].to_dict()
            continue
        if len(summary.index) != 1:
            raise TrainingError(f"training_score_metric_row_missing_prediction:{metric_name}")
        payload[metric_name] = summary.iloc[0].to_dict()
    return payload


def _updated_scoring_payload(
    *,
    existing: dict[str, object],
    requested_mode: str,
    requested_chunk_size: int,
    effective_backend: str,
    policy: ResolvedScoringPolicy,
    requested_stage: CanonicalScoringStage,
    refreshed_stages: tuple[str, ...],
) -> dict[str, object]:
    payload = dict(existing)
    payload["mode"] = requested_mode
    payload["era_chunk_size"] = requested_chunk_size
    payload["effective_backend"] = effective_backend
    payload["requested_stage"] = requested_stage
    payload["refreshed_stages"] = list(refreshed_stages)
    payload["policy"] = {
        "fnc_feature_set": policy.fnc_feature_set,
        "fnc_target_policy": policy.fnc_target_policy,
        "benchmark_min_overlap_ratio": policy.benchmark_min_overlap_ratio,
        "include_feature_neutral_metrics": policy.include_feature_neutral_metrics,
    }
    return payload


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


def _refresh_persisted_artifacts_provenance(
    *,
    score_provenance: dict[str, object],
    persisted_scoring: object,
) -> None:
    if not hasattr(persisted_scoring, "manifest_path") or not hasattr(persisted_scoring, "manifest_relative"):
        return
    manifest_path = cast(Path, getattr(persisted_scoring, "manifest_path"))
    manifest_relative = cast(Path, getattr(persisted_scoring, "manifest_relative"))
    try:
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(manifest_payload, dict):
        return
    chart_files = _as_mapping(manifest_payload.get("chart_files"))
    stage_files = _as_mapping(manifest_payload.get("stage_files"))
    refreshed = manifest_payload.get("refreshed_canonical_stages")
    refreshed_stages = [str(value) for value in refreshed] if isinstance(refreshed, list) else []
    score_provenance["artifacts"] = {
        "scoring_manifest": str(manifest_relative),
        "charts": sorted(chart_files.keys()),
        "stage_files": sorted(stage_files.keys()),
        "requested_stage": manifest_payload.get("requested_stage"),
        "refreshed_canonical_stages": refreshed_stages,
    }


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


def _include_feature_neutral_metrics(stage: CanonicalScoringStage) -> bool:
    return stage in {"all", "run_metric_series", "post_training_full"}


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered
