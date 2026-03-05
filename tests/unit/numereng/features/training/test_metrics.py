from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import numereng.features.training.scoring.metrics as metrics_module
from numereng.features.training.errors import TrainingDataError, TrainingMetricsError
from numereng.features.training.scoring.metrics import (
    attach_benchmark_predictions,
    ensure_full_benchmark_models,
    per_era_corr,
    summarize_prediction_file_with_scores,
)


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
) -> list[str]:
    resolved_feature_cols = feature_cols or ["feature_1", "feature_2"]
    version_dir = tmp_path / "v5.2"
    version_dir.mkdir(parents=True, exist_ok=True)

    feature_frame = pd.DataFrame({"id": ids, "era": eras})
    for idx, feature_col in enumerate(resolved_feature_cols, start=1):
        feature_frame[feature_col] = [float((row_idx + idx) % 11) / 10.0 for row_idx in range(len(feature_frame))]

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
    validation_frame.to_parquet(validation_path, index=False)

    features_path = version_dir / "features.json"
    features_path.write_text(
        json.dumps({"feature_sets": {"small": resolved_feature_cols}}, sort_keys=True),
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


def test_attach_benchmark_predictions_fails_on_partial_id_overlap() -> None:
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
        data_version="v5.2",
        client=_FakeClient(),
        benchmark_model="v52_lgbm_ender20",
        benchmark_data_path=benchmark_path,
        meta_model_data_path=meta_path,
        era_col="era",
        id_col="id",
        data_root=tmp_path,
    )

    assert set(summaries) >= {"corr", "fnc", "mmc", "cwmm", "bmc", "bmc_last_200_eras"}
    assert "prediction" in summaries["corr"].index
    assert "prediction" in summaries["fnc"].index
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
    assert joins["benchmark_overlap_rows"] == 4
    assert joins["meta_overlap_rows"] == 4


def test_summarize_prediction_file_with_scores_requires_meta_overlap(tmp_path: Path) -> None:
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

    with pytest.raises(TrainingDataError, match="training_meta_model_no_overlapping_ids"):
        summarize_prediction_file_with_scores(
            predictions_path=predictions_path,
            pred_cols=["prediction"],
            target_col="target",
            data_version="v5.2",
            client=_FakeClient(),
            benchmark_model="v52_lgbm_ender20",
            benchmark_data_path=benchmark_path,
            meta_model_data_path=meta_path,
            era_col="era",
            id_col="id",
            data_root=tmp_path,
        )


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

    assert set(materialized_summaries) == set(stream_summaries)
    for metric_name in materialized_summaries:
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


def test_summarize_prediction_file_with_scores_era_stream_handles_sparse_meta_and_benchmark_overlap(
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
            "id": ["e", "f"],
            "era": ["era3", "era3"],
            "numerai_meta_model": [0.35, 0.45],
        }
    ).set_index("id")
    meta.to_parquet(meta_path)

    materialized_summaries, _ = summarize_prediction_file_with_scores(
        predictions_path=predictions_path,
        pred_cols=["prediction"],
        target_col="target",
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

    assert set(materialized_summaries) == set(stream_summaries)
    for metric_name in materialized_summaries:
        assert_frame_equal(
            materialized_summaries[metric_name].sort_index(axis=0).sort_index(axis=1),
            stream_summaries[metric_name].sort_index(axis=0).sort_index(axis=1),
            check_exact=False,
            rtol=1e-9,
            atol=1e-9,
        )

    joins = stream_provenance["joins"]
    assert isinstance(joins, dict)
    assert joins["predictions_rows"] == 6
    assert joins["benchmark_overlap_rows"] == 4
    assert joins["meta_overlap_rows"] == 2


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
    blocked_paths = {predictions_path.resolve(), benchmark_path.resolve(), meta_path.resolve()}

    def _guard_read_table(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
        resolved = path.expanduser().resolve()
        if resolved in blocked_paths:
            raise AssertionError(f"_read_table should not materialize {resolved}")
        return original_read_table(path, columns=columns)

    monkeypatch.setattr(metrics_module, "_read_table", _guard_read_table)

    summaries, provenance = summarize_prediction_file_with_scores(
        predictions_path=predictions_path,
        pred_cols=["prediction"],
        target_col="target",
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
    assert set(summaries) == {"corr", "fnc", "bmc", "bmc_last_200_eras", "mmc", "cwmm"}
    joins = provenance["joins"]
    assert isinstance(joins, dict)
    assert joins["predictions_rows"] == 4


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
            data_version="v5.2",
            client=_FakeClient(),
            scoring_mode="unknown",
        )
