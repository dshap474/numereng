from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import numereng.features.scoring.service as scoring_service_module
from numereng.features.scoring.models import BenchmarkSource, PostTrainingScoringRequest, ScoringArtifactBundle


class _FakeClient:
    def download_dataset(
        self,
        filename: str,
        *,
        dest_path: str | None = None,
        round_num: int | None = None,
    ) -> str:
        _ = (filename, round_num)
        return dest_path or filename


def _request(*, stage: str) -> PostTrainingScoringRequest:
    return PostTrainingScoringRequest(
        run_id="run-123",
        config_hash="config-hash",
        seed=None,
        predictions_path=Path("predictions.parquet"),
        pred_cols=("prediction",),
        target_col="target",
        scoring_target_cols=("target", "target_ender_20"),
        data_version="v5.2",
        dataset_variant="non_downsampled",
        feature_set="small",
        feature_source_paths=None,
        dataset_scope="train_plus_validation",
        benchmark_source=BenchmarkSource(
            mode="path",
            name="benchmark",
            predictions_path=Path("benchmark.parquet"),
            pred_col="prediction",
        ),
        meta_model_col="numerai_meta_model",
        meta_model_data_path=Path("meta_model.parquet"),
        era_col="era",
        id_col="id",
        data_root=Path(".numereng") / "datasets",
        stage=stage,
    )


def test_run_post_training_scoring_uses_stage_driven_full_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    call_args: dict[str, object] = {}

    def _fake_score(
        *args: object, **kwargs: object
    ) -> tuple[dict[str, pd.DataFrame], dict[str, object], dict[str, pd.DataFrame]]:
        _ = args
        call_args.update(kwargs)
        return (
            {"corr": pd.DataFrame(index=["prediction"])},
            {},
            {"corr_native": pd.DataFrame([{"era": "era1", "value": 0.1}])},
        )

    monkeypatch.setattr(scoring_service_module, "score_prediction_file_with_details", _fake_score)
    monkeypatch.setattr(
        scoring_service_module,
        "build_scoring_artifact_bundle",
        lambda **kwargs: (
            ScoringArtifactBundle(
                series_frames={},
                stage_frames={"run_metric_series": pd.DataFrame([{"run_id": "run-123"}])},
                manifest={},
            ),
            {"benchmark_overlap_rows": 1},
        ),
    )

    result = scoring_service_module.run_post_training_scoring(
        request=_request(stage="all"),
        client=_FakeClient(),
    )

    assert call_args["include_feature_neutral_metrics"] is True
    assert call_args["benchmark_name"] == "benchmark"
    assert result.score_provenance["execution"] == {"requested_stage": "all"}
    assert result.policy.fnc_feature_set == "fncv3_features"
    assert "run_metric_series" in result.artifacts.stage_frames


def test_run_post_training_scoring_uses_stage_driven_core_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    call_args: dict[str, object] = {}

    def _fake_score(
        *args: object, **kwargs: object
    ) -> tuple[dict[str, pd.DataFrame], dict[str, object], dict[str, pd.DataFrame]]:
        _ = args
        call_args.update(kwargs)
        return (
            {"corr": pd.DataFrame(index=["prediction"])},
            {},
            {"corr_native": pd.DataFrame([{"era": "era1", "value": 0.2}])},
        )

    monkeypatch.setattr(scoring_service_module, "score_prediction_file_with_details", _fake_score)
    monkeypatch.setattr(
        scoring_service_module,
        "build_scoring_artifact_bundle",
        lambda **kwargs: (
            ScoringArtifactBundle(
                series_frames={},
                stage_frames={"post_training_core_summary": pd.DataFrame([{"run_id": "run-123", "value": 0.2}])},
                manifest={"stages": {"omissions": {"post_training_full": "not_requested"}}},
            ),
            {"benchmark_overlap_rows": 1},
        ),
    )

    result = scoring_service_module.run_post_training_scoring(
        request=_request(stage="post_training_core"),
        client=_FakeClient(),
    )

    assert call_args["include_feature_neutral_metrics"] is False
    assert result.score_provenance["execution"] == {"requested_stage": "post_training_core"}
    assert result.score_provenance["stages"]["omissions"]["post_training_full"] == "not_requested"
    assert result.policy.benchmark_min_overlap_ratio == pytest.approx(0.0)
    assert result.policy.fnc_target_policy == "scoring_target"
    assert result.artifacts.stage_frames["post_training_core_summary"].iloc[0]["value"] == pytest.approx(0.2)
