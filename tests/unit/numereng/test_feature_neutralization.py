from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from numereng.features.feature_neutralization import (
    NeutralizationDataError,
    NeutralizationValidationError,
    NeutralizePredictionsRequest,
    neutralize_prediction_frame,
    neutralize_predictions_file,
    neutralize_run_predictions,
)


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
