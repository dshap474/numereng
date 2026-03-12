from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numerai_tools.scoring import (
    correlation_contribution,
    feature_neutral_corr,
    numerai_corr,
    pearson_correlation,
    tie_kept_rank__gaussianize__pow_1_5,
)
from pandas.testing import assert_frame_equal

import numereng.features.scoring.metrics as metrics_module
from numereng.features.scoring.metrics import (
    attach_benchmark_predictions,
    ensure_full_benchmark_models,
    per_era_corr,
    per_era_reference_corr,
    summarize_prediction_file_with_scores,
    validate_join_source_coverage,
)
from numereng.features.training.errors import TrainingDataError, TrainingMetricsError


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


def _write_feature_sources(
    tmp_path: Path,
    *,
    ids: list[str],
    eras: list[str],
    feature_cols: list[str] | None = None,
    include_fncv3_features: bool = True,
    extra_cols: dict[str, list[object]] | None = None,
) -> list[str]:
    resolved_feature_cols = feature_cols or ["feature_1", "feature_2"]
    version_dir = tmp_path / "v5.2"
    version_dir.mkdir(parents=True, exist_ok=True)

    feature_frame = pd.DataFrame({"id": ids, "era": eras})
    for idx, feature_col in enumerate(resolved_feature_cols, start=1):
        feature_frame[feature_col] = [float((row_idx + idx) % 11) / 10.0 for row_idx in range(len(feature_frame))]
    for col, values in (extra_cols or {}).items():
        feature_frame[col] = values

    train_path = version_dir / "train.parquet"
    validation_path = version_dir / "validation.parquet"
    feature_frame.to_parquet(train_path, index=False)
    validation_frame = pd.DataFrame(
        {
            "id": ["__unused_validation_row__"],
            "era": ["__unused_validation_era__"],
            "data_type": ["validation"],
        }
    )
    for feature_col in resolved_feature_cols:
        validation_frame[feature_col] = [0.0]
    for col in (extra_cols or {}):
        validation_frame[col] = [0.0]
    validation_frame.to_parquet(validation_path, index=False)

    features_path = version_dir / "features.json"
    feature_sets: dict[str, list[str]] = {
        "small": resolved_feature_cols,
        "all": resolved_feature_cols,
    }
    if include_fncv3_features:
        feature_sets["fncv3_features"] = resolved_feature_cols
    features_path.write_text(
        json.dumps({"feature_sets": feature_sets}, sort_keys=True),
        encoding="utf-8",
    )
    return resolved_feature_cols


def test_per_era_corr_requires_numerai_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame(
        {
            "era": ["era1", "era1"],
            "target": [0.1, 0.2],
            "prediction": [0.3, 0.4],
        }
    )

    def fake_load_scoring_functions() -> tuple[object, object]:
        raise TrainingMetricsError("training_metrics_dependency_missing_numerai_tools")

    monkeypatch.setattr(metrics_module, "_load_scoring_functions", fake_load_scoring_functions)

    with pytest.raises(TrainingMetricsError, match="training_metrics_dependency_missing_numerai_tools"):
        per_era_corr(df, ["prediction"], "target")


def test_per_era_corr_multi_column_filters_nan_per_column(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame(
        {
            "era": ["era1", "era1", "era2", "era2"],
            "target": [0.1, 0.2, 0.3, 0.4],
            "pred_a": [1.0, None, 3.0, 4.0],
            "pred_b": [10.0, 20.0, 30.0, None],
        }
    )

    def fake_load_scoring_functions() -> tuple[object, object]:
        def _numerai_corr(preds: pd.DataFrame, target: pd.Series) -> pd.Series:
            _ = target
            return pd.Series({str(col): float(preds[str(col)].mean()) for col in preds.columns})

        return object(), _numerai_corr

    monkeypatch.setattr(metrics_module, "_load_scoring_functions", fake_load_scoring_functions)

    scores = per_era_corr(df, ["pred_a", "pred_b"], "target", era_col="era")
    assert scores.loc["era1", "pred_a"] == pytest.approx(1.0)
    assert scores.loc["era1", "pred_b"] == pytest.approx(15.0)
    assert scores.loc["era2", "pred_a"] == pytest.approx(3.5)
    assert scores.loc["era2", "pred_b"] == pytest.approx(30.0)


def test_per_era_fnc_multi_column_filters_nan_per_column(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame(
        {
            "era": ["era1", "era1", "era2", "era2"],
            "target": [0.1, 0.2, 0.3, 0.4],
            "feature_1": [0.5, 0.6, 0.7, 0.8],
            "feature_2": [0.9, 1.0, 1.1, 1.2],
            "pred_a": [1.0, None, 3.0, 4.0],
            "pred_b": [10.0, 20.0, 30.0, None],
        }
    )

    def _fake_feature_neutral_corr(
        predictions: pd.DataFrame,
        features: pd.DataFrame,
        targets: pd.Series,
    ) -> pd.Series:
        _ = (features, targets)
        return pd.Series({str(col): float(predictions[str(col)].mean()) for col in predictions.columns})

    monkeypatch.setattr(metrics_module, "_load_feature_neutral_corr", lambda: _fake_feature_neutral_corr)

    scores = metrics_module.per_era_fnc(
        df,
        ["pred_a", "pred_b"],
        ["feature_1", "feature_2"],
        "target",
        era_col="era",
    )
    assert scores.loc["era1", "pred_a"] == pytest.approx(1.0)
    assert scores.loc["era1", "pred_b"] == pytest.approx(15.0)
    assert scores.loc["era2", "pred_a"] == pytest.approx(3.5)
    assert scores.loc["era2", "pred_b"] == pytest.approx(30.0)


def test_per_era_feature_exposure_returns_rms_and_max() -> None:
    df = pd.DataFrame(
        {
            "era": ["era1", "era1", "era1", "era2", "era2", "era2"],
            "feature_1": [0.1, 0.2, 0.3, 0.3, 0.2, 0.1],
            "feature_2": [0.3, 0.2, 0.1, 0.1, 0.2, 0.3],
            "prediction": [0.1, 0.2, 0.3, 0.3, 0.2, 0.1],
        }
    )

    rms, max_exposure = metrics_module.per_era_feature_exposure(
        df,
        ["prediction"],
        ["feature_1", "feature_2"],
        era_col="era",
    )

    assert rms.loc["era1", "prediction"] == pytest.approx(1.0)
    assert rms.loc["era2", "prediction"] == pytest.approx(1.0)
    assert max_exposure.loc["era1", "prediction"] == pytest.approx(1.0)
    assert max_exposure.loc["era2", "prediction"] == pytest.approx(1.0)


def test_per_era_feature_exposure_skips_constant_features() -> None:
    df = pd.DataFrame(
        {
            "era": ["era1", "era1", "era1", "era2", "era2", "era2"],
            "feature_1": [0.1, 0.2, 0.3, 0.3, 0.2, 0.1],
            "feature_constant": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            "prediction": [0.1, 0.2, 0.3, 0.3, 0.2, 0.1],
        }
    )

    rms, max_exposure = metrics_module.per_era_feature_exposure(
        df,
        ["prediction"],
        ["feature_1", "feature_constant"],
        era_col="era",
    )

    assert rms.loc["era1", "prediction"] == pytest.approx(1.0)
    assert rms.loc["era2", "prediction"] == pytest.approx(1.0)
    assert max_exposure.loc["era1", "prediction"] == pytest.approx(1.0)
    assert max_exposure.loc["era2", "prediction"] == pytest.approx(1.0)


def test_per_era_feature_exposure_returns_nan_when_all_features_constant() -> None:
    df = pd.DataFrame(
        {
            "era": ["era1", "era1", "era1"],
            "feature_1": [1.0, 1.0, 1.0],
            "feature_2": [2.0, 2.0, 2.0],
            "prediction": [0.1, 0.2, 0.3],
        }
    )

    rms, max_exposure = metrics_module.per_era_feature_exposure(
        df,
        ["prediction"],
        ["feature_1", "feature_2"],
        era_col="era",
    )

    assert pd.isna(rms.loc["era1", "prediction"])
    assert pd.isna(max_exposure.loc["era1", "prediction"])


def test_per_era_core_metrics_match_numerai_tools_reference() -> None:
    rng = np.random.default_rng(7)
    rows: list[dict[str, object]] = []
    for era in ("era1", "era2"):
        for idx in range(50):
            rows.append(
                {
                    "id": f"{era}_{idx:03d}",
                    "era": era,
                    "target": float(rng.normal()),
                    "prediction": float(rng.normal()),
                    "feature_1": float(rng.normal()),
                    "feature_2": float(rng.normal()),
                    "meta_model": float(rng.normal()),
                }
            )
    frame = pd.DataFrame(rows).sort_values(["era", "id"]).reset_index(drop=True)

    corr_scores = metrics_module.per_era_corr(frame, ["prediction"], "target", era_col="era")
    fnc_scores = metrics_module.per_era_fnc(
        frame,
        ["prediction"],
        ["feature_1", "feature_2"],
        "target",
        era_col="era",
    )
    mmc_scores = metrics_module.per_era_mmc(
        frame,
        ["prediction"],
        "meta_model",
        "target",
        era_col="era",
    )
    cwmm_scores = metrics_module.per_era_cwmm(
        frame,
        ["prediction"],
        "meta_model",
        era_col="era",
    )

    for era, era_frame in frame.groupby("era", sort=True):
        indexed = era_frame.sort_values("id").set_index("id")
        expected_corr = float(numerai_corr(indexed[["prediction"]], indexed["target"]).iloc[0])
        expected_fnc = float(
            feature_neutral_corr(
                indexed[["prediction"]],
                indexed[["feature_1", "feature_2"]],
                indexed["target"],
            ).iloc[0]
        )
        expected_mmc = float(
            correlation_contribution(
                indexed[["prediction"]],
                indexed["meta_model"],
                indexed["target"],
            ).iloc[0]
        )
        transformed_prediction = tie_kept_rank__gaussianize__pow_1_5(indexed[["prediction"]])["prediction"]
        expected_cwmm = float(pearson_correlation(indexed["meta_model"], transformed_prediction))

        assert corr_scores.loc[era, "prediction"] == pytest.approx(expected_corr)
        assert fnc_scores.loc[era, "prediction"] == pytest.approx(expected_fnc)
        assert mmc_scores.loc[era, "prediction"] == pytest.approx(expected_mmc)
        assert cwmm_scores.loc[era, "prediction"] == pytest.approx(expected_cwmm)


def test_per_era_cwmm_is_not_numerai_corr_against_reference() -> None:
    frame = pd.DataFrame(
        {
            "id": [f"id_{idx:02d}" for idx in range(12)],
            "era": ["era1"] * 12,
            "prediction": [0.10, 0.30, 0.20, 0.50, 0.90, 0.80, 0.70, 0.60, 0.40, 0.11, 0.31, 0.21],
            "meta_model": [0.77, 0.12, 0.63, 0.19, 0.85, 0.41, 0.33, 0.57, 0.24, 0.66, 0.05, 0.48],
        }
    )

    per_era = metrics_module.per_era_cwmm(frame, ["prediction"], "meta_model", era_col="era")
    indexed = frame.sort_values("id").set_index("id")
    transformed_prediction = tie_kept_rank__gaussianize__pow_1_5(indexed[["prediction"]])["prediction"]
    expected = float(pearson_correlation(indexed["meta_model"], transformed_prediction))
    numerai_corr_against_reference = float(numerai_corr(indexed[["prediction"]], indexed["meta_model"]).iloc[0])

    assert per_era.loc["era1", "prediction"] == pytest.approx(expected)
    assert per_era.loc["era1", "prediction"] != pytest.approx(numerai_corr_against_reference)


def test_per_era_reference_corr_matches_numerai_corr_against_reference() -> None:
    frame = pd.DataFrame(
        {
            "id": [f"id_{idx:02d}" for idx in range(12)],
            "era": ["era1"] * 12,
            "prediction": [0.10, 0.30, 0.20, 0.50, 0.90, 0.80, 0.70, 0.60, 0.40, 0.11, 0.31, 0.21],
            "benchmark": [0.77, 0.12, 0.63, 0.19, 0.85, 0.41, 0.33, 0.57, 0.24, 0.66, 0.05, 0.48],
        }
    )

    per_era = per_era_reference_corr(frame, ["prediction"], "benchmark", era_col="era")
    indexed = frame.sort_values("id").set_index("id")
    expected = float(numerai_corr(indexed[["prediction"]], indexed["benchmark"]).iloc[0])

    assert per_era.loc["era1", "prediction"] == pytest.approx(expected)


def test_attach_benchmark_predictions_allows_partial_id_overlap_by_default() -> None:
    predictions = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "era": ["era1", "era1", "era2"],
            "target": [0.1, 0.2, 0.3],
        }
    )
    benchmark = pd.DataFrame(
        {
            "id": ["a", "b"],
            "era": ["era1", "era1"],
            "v52_lgbm_ender20": [0.5, 0.6],
        }
    ).set_index("id")

    attached = attach_benchmark_predictions(
        predictions=predictions,
        benchmark=benchmark,
        benchmark_col="v52_lgbm_ender20",
        era_col="era",
        id_col="id",
    )

    assert list(attached["id"]) == ["a", "b"]
    assert list(attached["era"]) == ["era1", "era1"]
    assert list(attached["v52_lgbm_ender20"]) == [0.5, 0.6]


def test_attach_benchmark_predictions_respects_explicit_min_overlap_ratio() -> None:
    predictions = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "era": ["era1", "era1", "era2"],
            "target": [0.1, 0.2, 0.3],
        }
    )
    benchmark = pd.DataFrame(
        {
            "id": ["a", "b"],
            "era": ["era1", "era1"],
            "v52_lgbm_ender20": [0.5, 0.6],
        }
    ).set_index("id")

    with pytest.raises(TrainingDataError, match="training_benchmark_partial_id_overlap"):
        attach_benchmark_predictions(
            predictions=predictions,
            benchmark=benchmark,
            benchmark_col="v52_lgbm_ender20",
            era_col="era",
            id_col="id",
            min_overlap_ratio=0.90,
        )


def test_validate_join_source_coverage_enforces_min_overlap_ratio() -> None:
    predictions = pd.DataFrame(
        {
            "id": [f"id_{idx}" for idx in range(10)],
            "era": ["era1"] * 10,
        }
    )
    source = pd.DataFrame(
        {
            "id": [f"id_{idx}" for idx in range(8)],
            "era": ["era1"] * 8,
        }
    )

    stats = validate_join_source_coverage(
        predictions,
        source,
        source_name="benchmark",
        era_col="era",
        id_col="id",
        min_overlap_ratio=0.80,
        include_missing_counts=True,
    )

    assert stats["benchmark_overlap_rows"] == 8
    assert stats["benchmark_missing_rows"] == 2
    assert stats["benchmark_overlap_ratio"] == pytest.approx(0.80)

    with pytest.raises(TrainingDataError, match="training_benchmark_partial_id_overlap"):
        validate_join_source_coverage(
            predictions,
            source,
            source_name="benchmark",
            era_col="era",
            id_col="id",
            min_overlap_ratio=0.81,
            include_missing_counts=True,
        )


def test_ensure_full_benchmark_models_writes_to_derived_cache(tmp_path: Path) -> None:
    data_root = (tmp_path / ".numereng" / "datasets").resolve()
    version_dir = data_root / "v5.2"
    version_dir.mkdir(parents=True, exist_ok=True)

    train = pd.DataFrame(
        {
            "id": ["id_train_1", "id_train_2"],
            "era": ["001", "002"],
            "v52_lgbm_ender20": [0.4, 0.5],
        }
    ).set_index("id")
    train.to_parquet(version_dir / "train_benchmark_models.parquet")

    validation = pd.DataFrame(
        {
            "id": ["id_val_keep", "id_val_drop"],
            "era": ["003", "004"],
            "v52_lgbm_ender20": [0.6, 0.7],
        }
    ).set_index("id")
    validation.to_parquet(version_dir / "validation_benchmark_models.parquet")

    validation_meta = pd.DataFrame(
        {
            "id": ["id_val_keep", "id_val_drop"],
            "data_type": ["validation", "live"],
        }
    ).set_index("id")
    validation_meta.to_parquet(version_dir / "validation.parquet")

    full_path = ensure_full_benchmark_models(_FakeClient(), "v5.2", data_root=data_root)

    assert full_path == (
        tmp_path / ".numereng" / "cache" / "derived_datasets" / "v5.2" / "full_benchmark_models.parquet"
    ).resolve()
    assert not (version_dir / "full_benchmark_models.parquet").exists()

    full = pd.read_parquet(full_path)
    assert list(full.index) == ["id_train_1", "id_train_2", "id_val_keep"]


def test_summarize_prediction_file_with_scores_includes_mmc_cwmm_and_provenance(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.parquet"
    predictions = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "era": ["era1", "era1", "era2", "era2"],
            "target": [0.2, 0.4, 0.1, 0.3],
            "prediction": [0.1, 0.3, 0.2, 0.4],
        }
    )
    predictions.to_parquet(predictions_path, index=False)
    _write_feature_sources(
        tmp_path,
        ids=predictions["id"].tolist(),
        eras=predictions["era"].tolist(),
    )

    benchmark_path = tmp_path / "benchmark.parquet"
    benchmark = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "era": ["era1", "era1", "era2", "era2"],
            "v52_lgbm_ender20": [0.6, 0.7, 0.2, 0.3],
        }
    ).set_index("id")
    benchmark.to_parquet(benchmark_path)

    meta_path = tmp_path / "meta.parquet"
    meta = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "era": ["era1", "era1", "era2", "era2"],
            "numerai_meta_model": [0.55, 0.65, 0.15, 0.25],
        }
    ).set_index("id")
    meta.to_parquet(meta_path)

    summaries, provenance = summarize_prediction_file_with_scores(
        predictions_path=predictions_path,
        pred_cols=["prediction"],
        target_col="target",
        scoring_target_cols=["target"],
        data_version="v5.2",
        client=_FakeClient(),
        benchmark_model="v52_lgbm_ender20",
        benchmark_data_path=benchmark_path,
        meta_model_data_path=meta_path,
        era_col="era",
        id_col="id",
        data_root=tmp_path,
    )

    assert set(summaries) >= {
        "corr",
        "fnc",
        "feature_exposure",
        "max_feature_exposure",
        "mmc",
        "cwmm",
        "bmc",
        "bmc_last_200_eras",
    }
    assert "prediction" in summaries["corr"].index
    assert "prediction" in summaries["fnc"].index
    assert "prediction" in summaries["feature_exposure"].index
    assert "prediction" in summaries["max_feature_exposure"].index
    assert "prediction" in summaries["mmc"].index
    assert "prediction" in summaries["cwmm"].index

    sources = provenance["sources"]
    assert isinstance(sources, dict)
    predictions_source = sources["predictions"]
    assert isinstance(predictions_source, dict)
    assert predictions_source["path"] == str(predictions_path.resolve())

    joins = provenance["joins"]
    assert isinstance(joins, dict)
    assert joins["predictions_rows"] == 4
    assert joins["fnc_overlap_rows"] == 4
    assert joins["benchmark_overlap_rows"] == 4
    assert joins["benchmark_missing_rows"] == 0
    assert joins["benchmark_missing_eras"] == 0
    assert joins["meta_overlap_rows"] == 4
    indexed_predictions = predictions.sort_values("id").set_index("id")
    indexed_benchmark = benchmark.sort_index()
    expected_benchmark_corr = []
    for era in sorted(indexed_predictions["era"].unique()):
        era_predictions = indexed_predictions[indexed_predictions["era"] == era]
        era_benchmark = indexed_benchmark[indexed_benchmark["era"] == era]
        expected_benchmark_corr.append(
            float(numerai_corr(era_predictions[["prediction"]], era_benchmark["v52_lgbm_ender20"]).iloc[0])
        )
    expected_avg_corr_with_benchmark = float(np.mean(expected_benchmark_corr))
    assert summaries["bmc"].loc["prediction", "avg_corr_with_benchmark"] == pytest.approx(
        expected_avg_corr_with_benchmark
    )
    policy = provenance["policy"]
    assert isinstance(policy, dict)
    assert policy["fnc_feature_set"] == "fncv3_features"
    assert policy["fnc_target_policy"] == "scoring_target"
    assert policy["benchmark_min_overlap_ratio"] == pytest.approx(0.0)


def test_summarize_prediction_file_with_scores_fails_when_fncv3_features_missing(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.parquet"
    predictions = pd.DataFrame(
        {
            "id": ["a", "b"],
            "era": ["era1", "era1"],
            "target": [0.2, 0.4],
            "prediction": [0.1, 0.3],
        }
    )
    predictions.to_parquet(predictions_path, index=False)
    _write_feature_sources(
        tmp_path,
        ids=predictions["id"].tolist(),
        eras=predictions["era"].tolist(),
        include_fncv3_features=False,
    )

    benchmark_path = tmp_path / "benchmark.parquet"
    benchmark = pd.DataFrame(
        {
            "id": ["a", "b"],
            "era": ["era1", "era1"],
            "v52_lgbm_ender20": [0.6, 0.7],
        }
    ).set_index("id")
    benchmark.to_parquet(benchmark_path)

    meta_path = tmp_path / "meta.parquet"
    meta = pd.DataFrame(
        {
            "id": ["a", "b"],
            "era": ["era1", "era1"],
            "numerai_meta_model": [0.55, 0.65],
        }
    ).set_index("id")
    meta.to_parquet(meta_path)

    with pytest.raises(TrainingDataError, match="training_feature_set_not_found:fncv3_features"):
        summarize_prediction_file_with_scores(
            predictions_path=predictions_path,
            pred_cols=["prediction"],
            target_col="target",
            scoring_target_cols=["target"],
            data_version="v5.2",
            client=_FakeClient(),
            benchmark_model="v52_lgbm_ender20",
            benchmark_data_path=benchmark_path,
            meta_model_data_path=meta_path,
            era_col="era",
            id_col="id",
            data_root=tmp_path,
        )


def test_summarize_prediction_file_with_scores_skips_fnc_dependencies_when_disabled(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.parquet"
    predictions = pd.DataFrame(
        {
            "id": ["a", "b"],
            "era": ["era1", "era1"],
            "target": [0.2, 0.4],
            "prediction": [0.1, 0.3],
        }
    )
    predictions.to_parquet(predictions_path, index=False)
    _write_feature_sources(
        tmp_path,
        ids=predictions["id"].tolist(),
        eras=predictions["era"].tolist(),
        include_fncv3_features=False,
    )

    benchmark_path = tmp_path / "benchmark.parquet"
    pd.DataFrame(
        {
            "id": ["a", "b"],
            "era": ["era1", "era1"],
            "v52_lgbm_ender20": [0.6, 0.7],
        }
    ).set_index("id").to_parquet(benchmark_path)

    meta_path = tmp_path / "meta.parquet"
    pd.DataFrame(
        {
            "id": ["a", "b"],
            "era": ["era1", "era1"],
            "numerai_meta_model": [0.55, 0.65],
        }
    ).set_index("id").to_parquet(meta_path)

    summaries, provenance = summarize_prediction_file_with_scores(
        predictions_path=predictions_path,
        pred_cols=["prediction"],
        target_col="target",
        scoring_target_cols=["target"],
        data_version="v5.2",
        client=_FakeClient(),
        benchmark_model="v52_lgbm_ender20",
        benchmark_data_path=benchmark_path,
        meta_model_data_path=meta_path,
        era_col="era",
        id_col="id",
        data_root=tmp_path,
        scoring_policy=metrics_module.default_scoring_policy(False),
    )

    assert "corr" in summaries
    assert "fnc" not in summaries
    assert "feature_exposure" not in summaries
    assert "max_feature_exposure" not in summaries
    policy = provenance["policy"]
    assert isinstance(policy, dict)
    assert policy["include_feature_neutral_metrics"] is False
    columns = provenance["columns"]
    assert isinstance(columns, dict)
    assert "fnc_feature_count" not in columns
    sources = provenance["sources"]
    assert isinstance(sources, dict)
    assert "fnc_feature_sources" not in sources
    joins = provenance["joins"]
    assert isinstance(joins, dict)
    assert "fnc_overlap_rows" not in joins


def test_summarize_prediction_file_with_scores_omits_meta_metrics_when_meta_missing(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.parquet"
    predictions = pd.DataFrame(
        {
            "id": ["a", "b"],
            "era": ["era1", "era1"],
            "target": [0.2, 0.4],
            "prediction": [0.1, 0.3],
        }
    )
    predictions.to_parquet(predictions_path, index=False)
    _write_feature_sources(
        tmp_path,
        ids=predictions["id"].tolist(),
        eras=predictions["era"].tolist(),
    )

    benchmark_path = tmp_path / "benchmark.parquet"
    benchmark = pd.DataFrame(
        {
            "id": ["a", "b"],
            "era": ["era1", "era1"],
            "v52_lgbm_ender20": [0.6, 0.7],
        }
    ).set_index("id")
    benchmark.to_parquet(benchmark_path)

    meta_path = tmp_path / "meta.parquet"
    meta = pd.DataFrame(
        {
            "id": ["x", "y"],
            "era": ["era2", "era2"],
            "numerai_meta_model": [0.55, 0.65],
        }
    ).set_index("id")
    meta.to_parquet(meta_path)

    summaries, provenance = summarize_prediction_file_with_scores(
        predictions_path=predictions_path,
        pred_cols=["prediction"],
        target_col="target",
        scoring_target_cols=["target"],
        data_version="v5.2",
        client=_FakeClient(),
        benchmark_model="v52_lgbm_ender20",
        benchmark_data_path=benchmark_path,
        meta_model_data_path=meta_path,
        era_col="era",
        id_col="id",
        data_root=tmp_path,
    )

    assert "mmc" not in summaries
    assert "cwmm" not in summaries
    meta_metrics = provenance["meta_metrics"]
    assert isinstance(meta_metrics, dict)
    assert meta_metrics["emitted"] is False
    assert meta_metrics["reason"] == "no_meta_overlap"


def test_summarize_prediction_file_with_scores_era_stream_matches_materialized(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.parquet"
    predictions = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e", "f"],
            "era": ["era1", "era1", "era2", "era2", "era3", "era3"],
            "target": [0.2, 0.4, 0.1, 0.3, 0.5, 0.7],
            "prediction": [0.1, 0.3, 0.2, 0.4, 0.6, 0.8],
        }
    )
    predictions.to_parquet(predictions_path, index=False)
    _write_feature_sources(
        tmp_path,
        ids=predictions["id"].tolist(),
        eras=predictions["era"].tolist(),
    )

    benchmark_path = tmp_path / "benchmark.parquet"
    benchmark = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e", "f"],
            "era": ["era1", "era1", "era2", "era2", "era3", "era3"],
            "v52_lgbm_ender20": [0.6, 0.7, 0.2, 0.3, 0.1, 0.2],
        }
    ).set_index("id")
    benchmark.to_parquet(benchmark_path)

    meta_path = tmp_path / "meta.parquet"
    meta = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e", "f"],
            "era": ["era1", "era1", "era2", "era2", "era3", "era3"],
            "numerai_meta_model": [0.55, 0.65, 0.15, 0.25, 0.35, 0.45],
        }
    ).set_index("id")
    meta.to_parquet(meta_path)

    materialized_summaries, materialized_provenance = summarize_prediction_file_with_scores(
        predictions_path=predictions_path,
        pred_cols=["prediction"],
        target_col="target",
        scoring_target_cols=["target"],
        data_version="v5.2",
        client=_FakeClient(),
        benchmark_model="v52_lgbm_ender20",
        benchmark_data_path=benchmark_path,
        meta_model_data_path=meta_path,
        era_col="era",
        id_col="id",
        data_root=tmp_path,
        scoring_mode="materialized",
    )
    stream_summaries, stream_provenance = summarize_prediction_file_with_scores(
        predictions_path=predictions_path,
        pred_cols=["prediction"],
        target_col="target",
        scoring_target_cols=["target"],
        data_version="v5.2",
        client=_FakeClient(),
        benchmark_model="v52_lgbm_ender20",
        benchmark_data_path=benchmark_path,
        meta_model_data_path=meta_path,
        era_col="era",
        id_col="id",
        data_root=tmp_path,
        scoring_mode="era_stream",
        era_chunk_size=1,
    )

    for metric_name in ("feature_exposure", "max_feature_exposure", "bmc", "bmc_last_200_eras", "cwmm"):
        assert_frame_equal(
            materialized_summaries[metric_name].sort_index(axis=0).sort_index(axis=1),
            stream_summaries[metric_name].sort_index(axis=0).sort_index(axis=1),
            check_exact=False,
            rtol=1e-9,
            atol=1e-9,
        )

    materialized_execution = materialized_provenance["execution"]
    assert isinstance(materialized_execution, dict)
    assert materialized_execution["effective_scoring_mode"] == "materialized"
    stream_execution = stream_provenance["execution"]
    assert isinstance(stream_execution, dict)
    assert stream_execution["effective_scoring_mode"] == "era_stream"
    assert stream_execution["era_chunk_size"] == 1


def test_summarize_prediction_file_with_scores_allows_partial_benchmark_overlap(
    tmp_path: Path,
) -> None:
    predictions_path = tmp_path / "predictions.parquet"
    predictions = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e", "f"],
            "era": ["era1", "era1", "era2", "era2", "era3", "era3"],
            "target": [0.2, 0.4, 0.1, 0.3, 0.5, 0.7],
            "prediction": [0.1, 0.3, 0.2, 0.4, 0.6, 0.8],
        }
    )
    predictions.to_parquet(predictions_path, index=False)
    _write_feature_sources(
        tmp_path,
        ids=predictions["id"].tolist(),
        eras=predictions["era"].tolist(),
    )

    benchmark_path = tmp_path / "benchmark.parquet"
    benchmark = pd.DataFrame(
        {
            "id": ["c", "d", "e", "f"],
            "era": ["era2", "era2", "era3", "era3"],
            "v52_lgbm_ender20": [0.2, 0.3, 0.1, 0.2],
        }
    ).set_index("id")
    benchmark.to_parquet(benchmark_path)

    meta_path = tmp_path / "meta.parquet"
    meta = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e", "f"],
            "era": ["era1", "era1", "era2", "era2", "era3", "era3"],
            "numerai_meta_model": [0.55, 0.65, 0.15, 0.25, 0.35, 0.45],
        }
    ).set_index("id")
    meta.to_parquet(meta_path)

    for scoring_mode in ("materialized", "era_stream"):
        summaries, provenance = summarize_prediction_file_with_scores(
            predictions_path=predictions_path,
            pred_cols=["prediction"],
            target_col="target",
            scoring_target_cols=["target"],
            data_version="v5.2",
            client=_FakeClient(),
            benchmark_model="v52_lgbm_ender20",
            benchmark_data_path=benchmark_path,
            meta_model_data_path=meta_path,
            era_col="era",
            id_col="id",
            data_root=tmp_path,
            scoring_mode=scoring_mode,
            era_chunk_size=1,
        )

        assert "bmc" in summaries
        assert "bmc_last_200_eras" in summaries
        joins = provenance["joins"]
        assert isinstance(joins, dict)
        assert joins["benchmark_overlap_rows"] == 4
        assert joins["benchmark_missing_rows"] == 2


def test_summarize_prediction_file_with_scores_emits_meta_metrics_on_overlap_window(
    tmp_path: Path,
) -> None:
    predictions_path = tmp_path / "predictions.parquet"
    predictions = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e", "f"],
            "era": ["era1", "era1", "era2", "era2", "era3", "era3"],
            "target": [0.2, 0.4, 0.1, 0.3, 0.5, 0.7],
            "prediction": [0.1, 0.3, 0.2, 0.4, 0.6, 0.8],
        }
    )
    predictions.to_parquet(predictions_path, index=False)
    _write_feature_sources(
        tmp_path,
        ids=predictions["id"].tolist(),
        eras=predictions["era"].tolist(),
    )

    benchmark_path = tmp_path / "benchmark.parquet"
    benchmark = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e", "f"],
            "era": ["era1", "era1", "era2", "era2", "era3", "era3"],
            "v52_lgbm_ender20": [0.6, 0.7, 0.2, 0.3, 0.1, 0.2],
        }
    ).set_index("id")
    benchmark.to_parquet(benchmark_path)

    meta_path = tmp_path / "meta.parquet"
    meta = pd.DataFrame(
        {
            "id": ["c", "d", "e", "f"],
            "era": ["era2", "era2", "era3", "era3"],
            "numerai_meta_model": [0.15, 0.25, 0.35, 0.45],
        }
    ).set_index("id")
    meta.to_parquet(meta_path)

    for scoring_mode in ("materialized", "era_stream"):
        summaries, provenance = summarize_prediction_file_with_scores(
            predictions_path=predictions_path,
            pred_cols=["prediction"],
            target_col="target",
            scoring_target_cols=["target"],
            data_version="v5.2",
            client=_FakeClient(),
            benchmark_model="v52_lgbm_ender20",
            benchmark_data_path=benchmark_path,
            meta_model_data_path=meta_path,
            era_col="era",
            id_col="id",
            data_root=tmp_path,
            scoring_mode=scoring_mode,
            era_chunk_size=1,
        )

        assert "mmc" in summaries
        assert "cwmm" in summaries
        meta_metrics = provenance["meta_metrics"]
        assert isinstance(meta_metrics, dict)
        assert meta_metrics["emitted"] is True
        assert meta_metrics["reason"] is None
        joins = provenance["joins"]
        assert isinstance(joins, dict)
        assert joins["meta_overlap_rows"] == 4
        assert joins["meta_overlap_eras"] == 2


def test_summarize_prediction_file_with_scores_omits_meta_metrics_without_overlap(
    tmp_path: Path,
) -> None:
    predictions_path = tmp_path / "predictions.parquet"
    predictions = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "era": ["era1", "era1", "era2", "era2"],
            "target": [0.2, 0.4, 0.1, 0.3],
            "prediction": [0.1, 0.3, 0.2, 0.4],
        }
    )
    predictions.to_parquet(predictions_path, index=False)
    _write_feature_sources(
        tmp_path,
        ids=predictions["id"].tolist(),
        eras=predictions["era"].tolist(),
    )

    benchmark_path = tmp_path / "benchmark.parquet"
    benchmark = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "era": ["era1", "era1", "era2", "era2"],
            "v52_lgbm_ender20": [0.6, 0.7, 0.2, 0.3],
        }
    ).set_index("id")
    benchmark.to_parquet(benchmark_path)

    meta_path = tmp_path / "meta.parquet"
    meta = pd.DataFrame(
        {
            "id": ["x", "y"],
            "era": ["era9", "era9"],
            "numerai_meta_model": [0.55, 0.65],
        }
    ).set_index("id")
    meta.to_parquet(meta_path)

    for scoring_mode in ("materialized", "era_stream"):
        summaries, provenance = summarize_prediction_file_with_scores(
            predictions_path=predictions_path,
            pred_cols=["prediction"],
            target_col="target",
            scoring_target_cols=["target"],
            data_version="v5.2",
            client=_FakeClient(),
            benchmark_model="v52_lgbm_ender20",
            benchmark_data_path=benchmark_path,
            meta_model_data_path=meta_path,
            era_col="era",
            id_col="id",
            data_root=tmp_path,
            scoring_mode=scoring_mode,
            era_chunk_size=1,
        )

        assert "mmc" not in summaries
        assert "cwmm" not in summaries
        assert "corr" in summaries
        assert "bmc" in summaries
        meta_metrics = provenance["meta_metrics"]
        assert isinstance(meta_metrics, dict)
        assert meta_metrics["emitted"] is False
        assert meta_metrics["reason"] == "no_meta_overlap"
        joins = provenance["joins"]
        assert isinstance(joins, dict)
        assert joins["meta_overlap_rows"] == 0


def test_summarize_prediction_file_with_scores_era_stream_avoids_full_table_reads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    predictions_path = tmp_path / "predictions.parquet"
    predictions = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "era": ["era1", "era1", "era2", "era2"],
            "target": [0.2, 0.4, 0.1, 0.3],
            "prediction": [0.1, 0.3, 0.2, 0.4],
        }
    )
    predictions.to_parquet(predictions_path, index=False)
    _write_feature_sources(
        tmp_path,
        ids=predictions["id"].tolist(),
        eras=predictions["era"].tolist(),
    )

    benchmark_path = tmp_path / "benchmark.parquet"
    benchmark = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "era": ["era1", "era1", "era2", "era2"],
            "v52_lgbm_ender20": [0.6, 0.7, 0.2, 0.3],
        }
    ).set_index("id")
    benchmark.to_parquet(benchmark_path)

    meta_path = tmp_path / "meta.parquet"
    meta = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "era": ["era1", "era1", "era2", "era2"],
            "numerai_meta_model": [0.55, 0.65, 0.15, 0.25],
        }
    ).set_index("id")
    meta.to_parquet(meta_path)

    original_read_table = metrics_module._read_table
    read_calls: list[tuple[Path, tuple[str, ...] | None]] = []

    def _track_read_table(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
        read_calls.append((path.expanduser().resolve(), tuple(columns) if columns is not None else None))
        return original_read_table(path, columns=columns)

    monkeypatch.setattr(metrics_module, "_read_table", _track_read_table)

    summaries, provenance = summarize_prediction_file_with_scores(
        predictions_path=predictions_path,
        pred_cols=["prediction"],
        target_col="target",
        scoring_target_cols=["target"],
        data_version="v5.2",
        client=_FakeClient(),
        benchmark_model="v52_lgbm_ender20",
        benchmark_data_path=benchmark_path,
        meta_model_data_path=meta_path,
        era_col="era",
        id_col="id",
        data_root=tmp_path,
        scoring_mode="era_stream",
        era_chunk_size=1,
    )
    assert set(summaries) == {
        "corr",
        "fnc",
        "feature_exposure",
        "max_feature_exposure",
        "bmc",
        "bmc_last_200_eras",
        "mmc",
        "cwmm",
    }
    joins = provenance["joins"]
    assert isinstance(joins, dict)
    assert joins["predictions_rows"] == 4
    predictions_calls = [call for call in read_calls if call[0] == predictions_path.resolve()]
    assert predictions_calls == [(predictions_path.resolve(), ("id", "era"))]
    benchmark_calls = [call for call in read_calls if call[0] == benchmark_path.resolve()]
    assert benchmark_calls == [(benchmark_path.resolve(), ("era", "v52_lgbm_ender20"))]
    meta_calls = [call for call in read_calls if call[0] == meta_path.resolve()]
    assert meta_calls == [(meta_path.resolve(), ("era", "numerai_meta_model"))]


def test_summarize_prediction_file_with_scores_materialized_reads_predictions_once_and_projects_aux_tables(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    predictions_path = tmp_path / "predictions.parquet"
    predictions = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "era": ["era1", "era1", "era2", "era2"],
            "target": [0.2, 0.4, 0.1, 0.3],
            "prediction": [0.1, 0.3, 0.2, 0.4],
            "unused_prediction_col": [1.0, 1.1, 1.2, 1.3],
        }
    )
    predictions.to_parquet(predictions_path, index=False)
    _write_feature_sources(
        tmp_path,
        ids=predictions["id"].tolist(),
        eras=predictions["era"].tolist(),
    )

    benchmark_path = tmp_path / "benchmark.parquet"
    benchmark = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "era": ["era1", "era1", "era2", "era2"],
            "v52_lgbm_ender20": [0.6, 0.7, 0.2, 0.3],
            "unused_benchmark_col": [9.0, 9.1, 9.2, 9.3],
        }
    ).set_index("id")
    benchmark.to_parquet(benchmark_path)

    meta_path = tmp_path / "meta.parquet"
    meta = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "era": ["era1", "era1", "era2", "era2"],
            "numerai_meta_model": [0.55, 0.65, 0.15, 0.25],
            "unused_meta_col": [7.0, 7.1, 7.2, 7.3],
        }
    ).set_index("id")
    meta.to_parquet(meta_path)

    original_read_table = metrics_module._read_table
    read_calls: list[tuple[Path, tuple[str, ...] | None]] = []

    def _track_read_table(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
        read_calls.append((path.expanduser().resolve(), tuple(columns) if columns is not None else None))
        return original_read_table(path, columns=columns)

    monkeypatch.setattr(metrics_module, "_read_table", _track_read_table)

    summarize_prediction_file_with_scores(
        predictions_path=predictions_path,
        pred_cols=["prediction"],
        target_col="target",
        scoring_target_cols=["target"],
        data_version="v5.2",
        client=_FakeClient(),
        benchmark_model="v52_lgbm_ender20",
        benchmark_data_path=benchmark_path,
        meta_model_data_path=meta_path,
        era_col="era",
        id_col="id",
        data_root=tmp_path,
        scoring_mode="materialized",
    )

    predictions_calls = [call for call in read_calls if call[0] == predictions_path.resolve()]
    assert len(predictions_calls) == 1
    assert predictions_calls[0][1] is None

    benchmark_calls = [call for call in read_calls if call[0] == benchmark_path.resolve()]
    assert len(benchmark_calls) == 1
    assert benchmark_calls[0][1] is not None
    benchmark_cols = set(benchmark_calls[0][1] or ())
    assert benchmark_cols in (
        {"era", "v52_lgbm_ender20"},
        {"id", "era", "v52_lgbm_ender20"},
    )

    meta_calls = [call for call in read_calls if call[0] == meta_path.resolve()]
    assert len(meta_calls) == 1
    assert meta_calls[0][1] is not None
    meta_cols = set(meta_calls[0][1] or ())
    assert meta_cols in (
        {"era", "numerai_meta_model"},
        {"id", "era", "numerai_meta_model"},
    )


def test_summarize_prediction_file_with_scores_rejects_unknown_mode(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.parquet"
    pd.DataFrame(
        {
            "id": ["a", "b"],
            "era": ["era1", "era1"],
            "target": [0.2, 0.4],
            "prediction": [0.1, 0.3],
        }
    ).to_parquet(predictions_path, index=False)

    with pytest.raises(TrainingMetricsError, match="training_metrics_scoring_mode_invalid"):
        summarize_prediction_file_with_scores(
            predictions_path=predictions_path,
            pred_cols=["prediction"],
            target_col="target",
            scoring_target_cols=["target"],
            data_version="v5.2",
            client=_FakeClient(),
            scoring_mode="unknown",
        )


def test_summarize_prediction_file_with_scores_emits_extra_target_metrics_from_dataset_sources(
    tmp_path: Path,
) -> None:
    predictions_path = tmp_path / "predictions.parquet"
    predictions = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "era": ["era1", "era1", "era2", "era2"],
            "target": [0.2, 0.4, 0.1, 0.3],
            "prediction": [0.1, 0.3, 0.2, 0.4],
        }
    )
    predictions.to_parquet(predictions_path, index=False)
    _write_feature_sources(
        tmp_path,
        ids=predictions["id"].tolist(),
        eras=predictions["era"].tolist(),
        extra_cols={"target_ender_20": [0.25, 0.35, 0.15, 0.45]},
    )

    benchmark_path = tmp_path / "benchmark.parquet"
    pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "era": ["era1", "era1", "era2", "era2"],
            "v52_lgbm_ender20": [0.6, 0.7, 0.2, 0.3],
        }
    ).set_index("id").to_parquet(benchmark_path)

    meta_path = tmp_path / "meta.parquet"
    pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "era": ["era1", "era1", "era2", "era2"],
            "numerai_meta_model": [0.55, 0.65, 0.15, 0.25],
        }
    ).set_index("id").to_parquet(meta_path)

    for scoring_mode in ("materialized", "era_stream"):
        summaries, provenance = summarize_prediction_file_with_scores(
            predictions_path=predictions_path,
            pred_cols=["prediction"],
            target_col="target",
            scoring_target_cols=["target", "target_ender_20"],
            data_version="v5.2",
            client=_FakeClient(),
            benchmark_model="v52_lgbm_ender20",
            benchmark_data_path=benchmark_path,
            meta_model_data_path=meta_path,
            era_col="era",
            id_col="id",
            data_root=tmp_path,
            scoring_mode=scoring_mode,
            era_chunk_size=1,
        )

        assert "corr" in summaries
        assert "fnc" in summaries
        assert "mmc" in summaries
        assert "corr_ender20" in summaries
        assert "fnc_ender20" in summaries
        assert "mmc_ender20" in summaries
        columns = provenance["columns"]
        assert isinstance(columns, dict)
        assert columns["scoring_target_cols"] == ["target", "target_ender_20"]
