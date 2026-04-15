from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from numerai_tools import scoring as numerai_tools_scoring

from numereng.features.feature_neutralization import (
    NeutralizationDataError,
    NeutralizationValidationError,
    NeutralizePredictionsRequest,
    neutralize_prediction_frame,
    neutralize_predictions_file,
    neutralize_run_predictions,
)
from numereng.features.feature_neutralization import service as service_module


def _write_predictions(path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "era": ["0001", "0001", "0002", "0002"],
            "id": ["a", "b", "c", "d"],
            "target_ender_20": [0.1, 0.9, 0.2, 0.8],
            "prediction": [0.2, 0.7, 0.25, 0.75],
        }
    )
    frame.to_parquet(path, index=False)
    return frame


def _write_neutralizers(path: Path) -> None:
    frame = pd.DataFrame(
        {
            "era": ["0001", "0001", "0002", "0002"],
            "id": ["a", "b", "c", "d"],
            "feature_1": [0.1, 0.2, 0.3, 0.4],
            "feature_2": [0.9, 0.8, 0.7, 0.6],
        }
    )
    frame.to_parquet(path, index=False)


def test_neutralize_predictions_file_writes_sidecar(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.parquet"
    source = _write_predictions(predictions_path)
    neutralizer_path = tmp_path / "neutralizers.parquet"
    _write_neutralizers(neutralizer_path)

    result = neutralize_predictions_file(
        request=NeutralizePredictionsRequest(
            predictions_path=predictions_path,
            neutralizer_path=neutralizer_path,
            proportion=0.5,
            mode="era",
            rank_output=True,
        )
    )

    assert result.output_path.name == "predictions.neutralized.parquet"
    assert result.source_rows == len(source)
    assert result.matched_rows == len(source)
    assert result.neutralizer_cols == ("feature_1", "feature_2")

    neutralized = pd.read_parquet(result.output_path)
    assert len(neutralized) == len(source)
    assert "prediction" in neutralized.columns
    assert pq.ParquetFile(result.output_path).metadata.row_group(0).column(0).compression == "ZSTD"


def test_neutralize_predictions_file_rejects_missing_neutralizer_cols(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.parquet"
    _write_predictions(predictions_path)
    neutralizer_path = tmp_path / "neutralizers.parquet"
    _write_neutralizers(neutralizer_path)

    with pytest.raises(NeutralizationValidationError, match="neutralization_neutralizer_columns_missing"):
        neutralize_predictions_file(
            request=NeutralizePredictionsRequest(
                predictions_path=predictions_path,
                neutralizer_path=neutralizer_path,
                neutralizer_cols=("missing_col",),
            )
        )


def test_neutralize_run_predictions_resolves_run_artifact(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    run_id = "run-1"
    run_dir = store_root / "runs" / run_id
    predictions_dir = run_dir / "artifacts" / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = predictions_dir / "predictions.parquet"
    source = _write_predictions(predictions_path)
    manifest = {"artifacts": {"predictions": "artifacts/predictions/predictions.parquet"}}
    (run_dir / "run.json").write_text(json.dumps(manifest), encoding="utf-8")

    neutralizer_path = tmp_path / "neutralizers.parquet"
    _write_neutralizers(neutralizer_path)

    result = neutralize_run_predictions(
        run_id=run_id,
        neutralizer_path=neutralizer_path,
        store_root=store_root,
    )

    assert result.run_id == run_id
    assert result.source_rows == len(source)
    assert result.output_path.is_file()


def test_neutralize_predictions_file_rejects_incomplete_join(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.parquet"
    _write_predictions(predictions_path)

    neutralizer_path = tmp_path / "neutralizers.parquet"
    pd.DataFrame(
        {
            "era": ["0001", "0001", "0002"],
            "id": ["a", "b", "c"],
            "feature_1": [0.1, 0.2, 0.3],
        }
    ).to_parquet(neutralizer_path, index=False)

    with pytest.raises(NeutralizationDataError, match="neutralization_missing_neutralizer_rows"):
        neutralize_predictions_file(
            request=NeutralizePredictionsRequest(
                predictions_path=predictions_path,
                neutralizer_path=neutralizer_path,
            )
        )


def test_neutralize_prediction_frame_preserves_original_join_key_formatting() -> None:
    predictions = pd.DataFrame(
        {
            "era": ["0001", "0001", "0002", "0002"],
            "id": ["a", "b", "c", "d"],
            "prediction": [0.2, 0.7, 0.25, 0.75],
        }
    )
    neutralizers = pd.DataFrame(
        {
            "era": [1, 1, 2, 2],
            "id": ["a", "b", "c", "d"],
            "feature_1": [0.1, 0.2, 0.3, 0.4],
        }
    )

    output = neutralize_prediction_frame(
        predictions=predictions,
        neutralizers=neutralizers,
        neutralizer_cols=("feature_1",),
        proportion=0.5,
        mode="era",
        rank_output=False,
    )

    assert output["era"].tolist() == ["0001", "0001", "0002", "0002"]
    assert output["id"].tolist() == ["a", "b", "c", "d"]


def test_neutralize_prediction_frame_rejects_duplicate_neutralizer_keys() -> None:
    predictions = pd.DataFrame(
        {
            "era": ["0001", "0001"],
            "id": ["a", "b"],
            "prediction": [0.2, 0.7],
        }
    )
    neutralizers = pd.DataFrame(
        {
            "era": ["0001", "0001", "0001"],
            "id": ["a", "a", "b"],
            "feature_1": [0.1, 0.2, 0.3],
        }
    )

    with pytest.raises(NeutralizationValidationError, match="neutralization_neutralizer_keys_duplicated"):
        neutralize_prediction_frame(
            predictions=predictions,
            neutralizers=neutralizers,
            neutralizer_cols=("feature_1",),
            proportion=0.5,
            mode="era",
            rank_output=False,
        )


def test_neutralize_prediction_frame_matches_vendored_numerai_tools_global_mode() -> None:
    predictions = pd.DataFrame(
        {
            "era": ["0001", "0001", "0002", "0002"],
            "id": ["a", "b", "c", "d"],
            "prediction": [0.12, 0.81, 0.23, 0.74],
        }
    )
    neutralizers = pd.DataFrame(
        {
            "era": ["0001", "0001", "0002", "0002"],
            "id": ["a", "b", "c", "d"],
            "feature_1": [0.1, 0.2, 0.3, 0.4],
            "feature_2": [0.8, 0.7, 0.6, 0.5],
        }
    )

    actual = neutralize_prediction_frame(
        predictions=predictions,
        neutralizers=neutralizers,
        neutralizer_cols=("feature_1", "feature_2"),
        proportion=0.5,
        mode="global",
        rank_output=False,
    )

    expected = numerai_tools_scoring.neutralize(
        predictions[["prediction"]].copy(),
        neutralizers[["feature_1", "feature_2"]].copy(),
        proportion=0.5,
    )

    np.testing.assert_allclose(
        actual["prediction"].to_numpy(dtype=float),
        expected["prediction"].to_numpy(dtype=float),
        rtol=1e-10,
        atol=1e-10,
    )


def test_neutralize_prediction_frame_matches_vendored_numerai_tools_era_mode() -> None:
    predictions = pd.DataFrame(
        {
            "era": ["0001", "0001", "0001", "0002", "0002", "0002"],
            "id": ["a", "b", "c", "d", "e", "f"],
            "prediction": [0.11, 0.42, 0.93, 0.22, 0.61, 0.84],
        }
    )
    neutralizers = pd.DataFrame(
        {
            "era": ["0001", "0001", "0001", "0002", "0002", "0002"],
            "id": ["a", "b", "c", "d", "e", "f"],
            "feature_1": [0.1, 0.3, 0.6, 0.2, 0.5, 0.7],
            "feature_2": [0.8, 0.5, 0.2, 0.7, 0.4, 0.1],
        }
    )

    actual = neutralize_prediction_frame(
        predictions=predictions,
        neutralizers=neutralizers,
        neutralizer_cols=("feature_1", "feature_2"),
        proportion=0.3,
        mode="era",
        rank_output=False,
    )

    expected_parts: list[pd.DataFrame] = []
    for era in ("0001", "0002"):
        pred_slice = predictions.loc[predictions["era"] == era, ["prediction"]].copy()
        neutralizer_slice = neutralizers.loc[neutralizers["era"] == era, ["feature_1", "feature_2"]].copy()
        expected_parts.append(numerai_tools_scoring.neutralize(pred_slice, neutralizer_slice, proportion=0.3))
    expected = pd.concat(expected_parts, axis=0)

    np.testing.assert_allclose(
        actual["prediction"].to_numpy(dtype=float),
        expected["prediction"].to_numpy(dtype=float),
        rtol=1e-10,
        atol=1e-10,
    )


def test_internal_neutralize_matrix_matches_vendored_numerai_tools_for_multiple_columns() -> None:
    values = pd.DataFrame(
        {
            "prediction_a": [0.11, 0.42, 0.73, 0.21],
            "prediction_b": [0.17, 0.32, 0.68, 0.29],
        }
    )
    neutralizers = pd.DataFrame(
        {
            "feature_1": [0.1, 0.2, 0.3, 0.4],
            "feature_2": [0.4, 0.3, 0.2, 0.1],
        }
    )

    actual = service_module._neutralize_matrix(
        values=values.to_numpy(dtype=float),
        neutralizers=neutralizers.to_numpy(dtype=float),
        proportion=0.5,
    )
    expected = numerai_tools_scoring.neutralize(values.copy(), neutralizers.copy(), proportion=0.5)

    np.testing.assert_allclose(actual, expected.to_numpy(dtype=float), rtol=1e-10, atol=1e-10)


def test_era_mode_uses_one_lstsq_solve_per_era_block(monkeypatch: pytest.MonkeyPatch) -> None:
    original_lstsq = service_module.np.linalg.lstsq
    calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def _counting_lstsq(a: np.ndarray, b: np.ndarray, rcond: float | None = None):
        calls.append((a.shape, b.shape))
        return original_lstsq(a, b, rcond=rcond)

    monkeypatch.setattr(service_module.np.linalg, "lstsq", _counting_lstsq)

    values = np.asarray(
        [
            [0.1, 0.9],
            [0.2, 0.8],
            [0.3, 0.7],
            [0.4, 0.6],
        ],
        dtype=float,
    )
    neutralizers = np.asarray(
        [
            [0.1, 0.8],
            [0.2, 0.7],
            [0.3, 0.6],
            [0.4, 0.5],
        ],
        dtype=float,
    )
    eras = pd.Series(["0001", "0001", "0002", "0002"], dtype=str)

    resolved = service_module._neutralize_matrix_by_mode(
        values=values,
        neutralizers=neutralizers,
        eras=eras,
        proportion=0.5,
        mode="era",
    )

    assert resolved.shape == values.shape
    assert calls == [((2, 3), (2, 2)), ((2, 3), (2, 2))]


def test_rank_output_still_produces_percentile_ranked_predictions() -> None:
    predictions = pd.DataFrame(
        {
            "era": ["0001", "0001", "0001", "0002", "0002", "0002"],
            "id": ["a", "b", "c", "d", "e", "f"],
            "prediction": [0.11, 0.42, 0.93, 0.22, 0.61, 0.84],
        }
    )
    neutralizers = pd.DataFrame(
        {
            "era": ["0001", "0001", "0001", "0002", "0002", "0002"],
            "id": ["a", "b", "c", "d", "e", "f"],
            "feature_1": [0.1, 0.3, 0.6, 0.2, 0.5, 0.7],
            "feature_2": [0.8, 0.5, 0.2, 0.7, 0.4, 0.1],
        }
    )

    output = neutralize_prediction_frame(
        predictions=predictions,
        neutralizers=neutralizers,
        neutralizer_cols=("feature_1", "feature_2"),
        proportion=0.5,
        mode="era",
        rank_output=True,
    )

    by_era = output.groupby("era", sort=False)["prediction"]
    assert by_era.apply(lambda s: list(np.round(s.to_numpy(dtype=float), 6))).tolist() == [
        [0.333333, 0.666667, 1.0],
        [0.333333, 0.666667, 1.0],
    ]
