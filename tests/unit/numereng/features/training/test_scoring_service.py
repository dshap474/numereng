from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import numereng.features.scoring.service as scoring_service_module
from numereng.features.scoring.models import PostTrainingScoringRequest


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


def _request(
    *,
    scoring_mode: str,
    benchmark_path: str | Path | None,
    meta_path: str | Path | None,
) -> PostTrainingScoringRequest:
    return PostTrainingScoringRequest(
        predictions_path=Path("predictions.parquet"),
        pred_cols=("prediction",),
        target_col="target",
        scoring_target_cols=("target", "target_ender_20"),
        data_version="v5.2",
        dataset_variant="non_downsampled",
        feature_set="small",
        feature_source_paths=None,
        dataset_scope="train_plus_validation",
        benchmark_model="v52_lgbm_ender20",
        benchmark_data_path=benchmark_path,
        meta_model_col="numerai_meta_model",
        meta_model_data_path=meta_path,
        era_col="era",
        id_col="id",
        data_root=Path(".numereng") / "datasets",
        scoring_mode=scoring_mode,
        era_chunk_size=64,
    )


def test_run_post_training_scoring_falls_back_to_materialized_for_non_parquet_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_args: dict[str, object] = {}

    def _fake_summarize(*args: object, **kwargs: object) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
        _ = args
        call_args.update(kwargs)
        return {"corr": pd.DataFrame(index=["prediction"])}, {}

    monkeypatch.setattr(scoring_service_module, "summarize_prediction_file_with_scores", _fake_summarize)
    monkeypatch.setattr(
        scoring_service_module,
        "build_primary_per_era_corr_frame",
        lambda **kwargs: pd.DataFrame([{"era": "era1", "corr": 0.1}]),
    )

    result = scoring_service_module.run_post_training_scoring(
        request=_request(
            scoring_mode="era_stream",
            benchmark_path=Path("benchmark.csv"),
            meta_path=Path("meta_model.parquet"),
        ),
        client=_FakeClient(),
    )

    assert call_args["scoring_mode"] == "materialized"
    assert result.effective_scoring_backend == "materialized"
    execution = result.score_provenance["execution"]
    assert isinstance(execution, dict)
    assert execution["requested_scoring_mode"] == "era_stream"
    assert execution["effective_scoring_mode"] == "materialized"
    assert execution["fallback_reason"] == "external_source_not_parquet"
    assert result.policy.fnc_feature_set == "fncv3_features"
    assert result.per_era_corr is not None
    assert list(result.per_era_corr.columns) == ["era", "corr"]


def test_run_post_training_scoring_preserves_era_stream_for_parquet_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_args: dict[str, object] = {}

    def _fake_summarize(*args: object, **kwargs: object) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
        _ = args
        call_args.update(kwargs)
        return {"corr": pd.DataFrame(index=["prediction"])}, {}

    monkeypatch.setattr(scoring_service_module, "summarize_prediction_file_with_scores", _fake_summarize)
    monkeypatch.setattr(
        scoring_service_module,
        "build_primary_per_era_corr_frame",
        lambda **kwargs: pd.DataFrame([{"era": "era1", "corr": 0.2}]),
    )

    result = scoring_service_module.run_post_training_scoring(
        request=_request(
            scoring_mode="era_stream",
            benchmark_path=Path("benchmark.parquet"),
            meta_path=Path("meta_model.parquet"),
        ),
        client=_FakeClient(),
    )

    assert call_args["scoring_mode"] == "era_stream"
    assert result.effective_scoring_backend == "era_stream"
    execution = result.score_provenance["execution"]
    assert isinstance(execution, dict)
    assert execution["requested_scoring_mode"] == "era_stream"
    assert execution["effective_scoring_mode"] == "era_stream"
    assert "fallback_reason" not in execution
    assert result.policy.benchmark_min_overlap_ratio == pytest.approx(0.0)
    assert result.policy.fnc_target_policy == "scoring_target"
    assert result.per_era_corr is not None
    assert result.per_era_corr.iloc[0]["corr"] == pytest.approx(0.2)
