"""Service orchestration for modular post-training scoring."""

from __future__ import annotations

from pathlib import Path

from numereng.features.training.client import TrainingDataClient
from numereng.features.training.scoring.metrics import summarize_prediction_file_with_scores
from numereng.features.training.scoring.models import (
    PostTrainingScoringRequest,
    PostTrainingScoringResult,
)


def run_post_training_scoring(
    *,
    request: PostTrainingScoringRequest,
    client: TrainingDataClient,
) -> PostTrainingScoringResult:
    """Compute post-training metrics/provenance from persisted predictions."""
    effective_scoring_backend = _resolve_effective_scoring_mode(request)

    summaries, score_provenance = summarize_prediction_file_with_scores(
        predictions_path=request.predictions_path,
        pred_cols=list(request.pred_cols),
        target_col=request.target_col,
        data_version=request.data_version,
        dataset_variant=request.dataset_variant,
        feature_set=request.feature_set,
        feature_cols=request.feature_cols,
        feature_source_paths=request.feature_source_paths,
        full_data_path=request.full_data_path,
        dataset_scope=request.dataset_scope,
        client=client,
        benchmark_model=request.benchmark_model,
        benchmark_data_path=request.benchmark_data_path,
        meta_model_data_path=request.meta_model_data_path,
        meta_model_col=request.meta_model_col,
        era_col=request.era_col,
        id_col=request.id_col,
        data_root=request.data_root,
        scoring_mode=effective_scoring_backend,
        era_chunk_size=request.era_chunk_size,
    )

    execution_payload: dict[str, object] = {
        "requested_scoring_mode": request.scoring_mode,
        "effective_scoring_mode": effective_scoring_backend,
        "era_chunk_size": request.era_chunk_size,
    }
    if request.scoring_mode == "era_stream" and effective_scoring_backend != request.scoring_mode:
        execution_payload["fallback_reason"] = "external_source_not_parquet"
    score_provenance["execution"] = execution_payload

    return PostTrainingScoringResult(
        summaries=summaries,
        score_provenance=score_provenance,
        effective_scoring_backend=effective_scoring_backend,
    )


def _resolve_effective_scoring_mode(request: PostTrainingScoringRequest) -> str:
    if request.scoring_mode != "era_stream":
        return request.scoring_mode

    if not _looks_parquet(request.predictions_path):
        return "materialized"
    if not _looks_parquet(request.benchmark_data_path):
        return "materialized"
    if not _looks_parquet(request.meta_model_data_path):
        return "materialized"
    return request.scoring_mode


def _looks_parquet(value: str | Path | None) -> bool:
    if value is None:
        return True
    return Path(str(value)).suffix.lower() == ".parquet"
