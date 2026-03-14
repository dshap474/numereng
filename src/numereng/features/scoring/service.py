"""Service orchestration for modular post-training scoring."""

from __future__ import annotations

from pathlib import Path

from numereng.features.scoring.metrics import (
    build_scoring_artifact_bundle,
    summarize_prediction_file_with_scores,
)
from numereng.features.scoring.models import (
    PostTrainingScoringRequest,
    PostTrainingScoringResult,
    ResolvedScoringPolicy,
    default_scoring_policy,
)
from numereng.features.training.client import TrainingDataClient


def run_post_training_scoring(
    *,
    request: PostTrainingScoringRequest,
    client: TrainingDataClient,
) -> PostTrainingScoringResult:
    """Compute post-training metrics/provenance from persisted predictions."""

    effective_scoring_backend = _resolve_effective_scoring_mode(request)
    base_policy = default_scoring_policy()
    policy = ResolvedScoringPolicy(
        fnc_feature_set=base_policy.fnc_feature_set,
        fnc_target_policy=base_policy.fnc_target_policy,
        benchmark_min_overlap_ratio=base_policy.benchmark_min_overlap_ratio,
        include_feature_neutral_metrics=request.include_feature_neutral_metrics,
    )

    summaries, score_provenance = summarize_prediction_file_with_scores(
        predictions_path=request.predictions_path,
        pred_cols=list(request.pred_cols),
        target_col=request.target_col,
        scoring_target_cols=list(request.scoring_target_cols),
        data_version=request.data_version,
        dataset_variant=request.dataset_variant,
        feature_set=request.feature_set,
        feature_source_paths=request.feature_source_paths,
        dataset_scope=request.dataset_scope,
        client=client,
        benchmark_model=request.benchmark_source.pred_col,
        benchmark_data_path=request.benchmark_source.predictions_path,
        meta_model_data_path=request.meta_model_data_path,
        meta_model_col=request.meta_model_col,
        era_col=request.era_col,
        id_col=request.id_col,
        data_root=request.data_root,
        scoring_mode=effective_scoring_backend,
        era_chunk_size=request.era_chunk_size,
        scoring_policy=policy,
    )
    artifacts, benchmark_joins = build_scoring_artifact_bundle(
        predictions_path=request.predictions_path,
        pred_cols=list(request.pred_cols),
        target_col=request.target_col,
        scoring_target_cols=list(request.scoring_target_cols),
        benchmark_source=request.benchmark_source,
        client=client,
        data_version=request.data_version,
        dataset_variant=request.dataset_variant,
        feature_set=request.feature_set,
        feature_source_paths=request.feature_source_paths,
        dataset_scope=request.dataset_scope,
        meta_model_col=request.meta_model_col,
        meta_model_data_path=request.meta_model_data_path,
        era_col=request.era_col,
        id_col=request.id_col,
        data_root=request.data_root,
        scoring_policy=policy,
    )

    execution_payload: dict[str, object] = {
        "requested_scoring_mode": request.scoring_mode,
        "effective_scoring_mode": effective_scoring_backend,
        "era_chunk_size": request.era_chunk_size,
    }
    if request.scoring_mode == "era_stream" and effective_scoring_backend != request.scoring_mode:
        execution_payload["fallback_reason"] = "external_source_not_parquet"
    score_provenance["execution"] = execution_payload
    score_provenance["benchmark_source"] = {
        "mode": request.benchmark_source.mode,
        "name": request.benchmark_source.name,
        "predictions_path": str(request.benchmark_source.predictions_path),
        "pred_col": request.benchmark_source.pred_col,
        "metadata_path": (
            str(request.benchmark_source.metadata_path) if request.benchmark_source.metadata_path else None
        ),
    }
    columns_payload = score_provenance.setdefault("columns", {})
    if isinstance(columns_payload, dict):
        columns_payload["benchmark_prediction_col"] = request.benchmark_source.pred_col
        columns_payload["target_families_emitted"] = sorted(
            {
                "native",
                *(["ender20"] if "bmc_ender20_per_era" in artifacts.series_frames else []),
            }
        )
    joins_payload = score_provenance.setdefault("joins", {})
    if isinstance(joins_payload, dict):
        joins_payload["benchmark_source"] = benchmark_joins
    score_provenance["artifacts"] = {
        "scoring_manifest": "artifacts/scoring/manifest.json",
        "series": sorted(artifacts.series_frames.keys()),
    }

    return PostTrainingScoringResult(
        summaries=summaries,
        score_provenance=score_provenance,
        effective_scoring_backend=effective_scoring_backend,
        policy=policy,
        artifacts=artifacts,
    )


def _resolve_effective_scoring_mode(request: PostTrainingScoringRequest) -> str:
    if request.scoring_mode != "era_stream":
        return request.scoring_mode

    if not _looks_parquet(request.predictions_path):
        return "materialized"
    if not _looks_parquet(request.benchmark_source.predictions_path):
        return "materialized"
    if not _looks_parquet(request.meta_model_data_path):
        return "materialized"
    return request.scoring_mode


def _looks_parquet(value: str | Path | None) -> bool:
    if value is None:
        return True
    return Path(str(value)).suffix.lower() == ".parquet"
