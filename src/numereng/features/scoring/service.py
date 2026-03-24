"""Service orchestration for canonical run-scoring workflows."""

from __future__ import annotations

from collections.abc import Mapping

from numereng.features.scoring.metrics import (
    build_scoring_artifact_bundle,
    score_prediction_file_with_details,
)
from numereng.features.scoring.models import (
    PostTrainingScoringRequest,
    PostTrainingScoringResult,
    RunScoringRequest,
    RunScoringResult,
    default_scoring_policy,
)
from numereng.features.training.client import TrainingDataClient


def run_scoring(
    *,
    request: RunScoringRequest,
    client: TrainingDataClient,
) -> RunScoringResult:
    """Compute canonical run-scoring outputs from persisted predictions."""

    policy = default_scoring_policy()

    summaries, score_provenance, metric_frames = score_prediction_file_with_details(
        predictions_path=request.predictions_path,
        pred_cols=list(request.pred_cols),
        target_col=request.target_col,
        scoring_target_cols=list(request.scoring_target_cols),
        scoring_targets_explicit=request.scoring_targets_explicit,
        data_version=request.data_version,
        dataset_variant=request.dataset_variant,
        feature_set=request.feature_set,
        feature_source_paths=request.feature_source_paths,
        dataset_scope=request.dataset_scope,
        client=client,
        benchmark_model=request.benchmark_source.pred_col,
        benchmark_name=request.benchmark_source.name,
        benchmark_data_path=request.benchmark_source.predictions_path,
        benchmark_metadata_path=request.benchmark_source.metadata_path,
        meta_model_data_path=request.meta_model_data_path,
        meta_model_col=request.meta_model_col,
        era_col=request.era_col,
        id_col=request.id_col,
        data_root=request.data_root,
        include_feature_neutral_metrics=request.stage in {"all", "post_training_full"},
        scoring_policy=policy,
    )
    artifacts, benchmark_joins = build_scoring_artifact_bundle(
        run_id=request.run_id,
        config_hash=request.config_hash,
        seed=request.seed,
        predictions_path=request.predictions_path,
        pred_cols=list(request.pred_cols),
        target_col=request.target_col,
        scoring_target_cols=list(request.scoring_target_cols),
        scoring_targets_explicit=request.scoring_targets_explicit,
        summaries=summaries,
        metric_frames=metric_frames,
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
        requested_stage=request.stage,
        scoring_policy=policy,
    )
    refreshed_stages = _refreshed_canonical_stages(request.stage, artifacts.stage_frames)

    score_provenance["execution"] = {"requested_stage": request.stage}
    score_provenance["benchmark_source"] = {
        "mode": request.benchmark_source.mode,
        "name": request.benchmark_source.name,
        "predictions_path": str(request.benchmark_source.predictions_path),
        "source_pred_col": request.benchmark_source.pred_col,
        "scoring_col": request.benchmark_source.name,
        "metadata_path": (
            str(request.benchmark_source.metadata_path) if request.benchmark_source.metadata_path else None
        ),
    }
    columns_payload = score_provenance.setdefault("columns", {})
    if isinstance(columns_payload, dict):
        columns_payload["benchmark_prediction_col"] = request.benchmark_source.pred_col
        columns_payload["benchmark_scoring_col"] = request.benchmark_source.name
        columns_payload["target_families_emitted"] = {
            "corr_fnc": sorted(
                {
                    "native",
                    *(["ender20"] if "corr_ender20" in metric_frames or "fnc_ender20" in metric_frames else []),
                }
            ),
            "contribution": sorted(
                {
                    *(["ender20"] if "mmc" in summaries or "bmc" in summaries else []),
                    *(
                        ["aliased"]
                        if any(key.startswith(("mmc_", "bmc_", "bmc_last_200_eras_")) for key in summaries)
                        else []
                    ),
                }
            ),
        }
    joins_payload = score_provenance.setdefault("joins", {})
    if isinstance(joins_payload, dict):
        joins_payload["benchmark_source"] = benchmark_joins
    score_provenance["artifacts"] = {
        "scoring_manifest": "artifacts/scoring/manifest.json",
        "charts": sorted(key for key in artifacts.stage_frames.keys() if key == "run_metric_series"),
        "stage_files": sorted(key for key in artifacts.stage_frames.keys()),
        "requested_stage": request.stage,
        "refreshed_canonical_stages": refreshed_stages,
    }
    score_provenance["stages"] = artifacts.manifest.get("stages", {})

    return RunScoringResult(
        summaries=summaries,
        score_provenance=score_provenance,
        policy=policy,
        artifacts=artifacts,
        requested_stage=request.stage,
        refreshed_stages=tuple(refreshed_stages),
    )


def run_post_training_scoring(
    *,
    request: PostTrainingScoringRequest,
    client: TrainingDataClient,
) -> PostTrainingScoringResult:
    """Backward-compatible alias for the canonical run scorer."""

    return run_scoring(request=request, client=client)


def _refreshed_canonical_stages(stage: str, stage_frames: Mapping[str, object]) -> list[str]:
    if stage == "all":
        refreshed: list[str] = []
        if "run_metric_series" in stage_frames:
            refreshed.append("run_metric_series")
        if "post_fold_per_era" in stage_frames or "post_fold_snapshots" in stage_frames:
            refreshed.append("post_fold")
        if "post_training_core_summary" in stage_frames:
            refreshed.append("post_training_core")
        if "post_training_full_summary" in stage_frames:
            refreshed.append("post_training_full")
        return refreshed
    if stage == "post_fold":
        return ["post_fold"] if "post_fold_per_era" in stage_frames or "post_fold_snapshots" in stage_frames else []
    if stage == "post_training_core":
        return ["post_training_core"] if "post_training_core_summary" in stage_frames else []
    if stage == "post_training_full":
        refreshed: list[str] = []
        if "post_training_core_summary" in stage_frames:
            refreshed.append("post_training_core")
        if "post_training_full_summary" in stage_frames:
            refreshed.append("post_training_full")
        return refreshed
    if stage == "run_metric_series":
        return ["run_metric_series"] if "run_metric_series" in stage_frames else []
    return []
