from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from numereng.features.training.errors import TrainingConfigError
from numereng.features.training.service import attach_baseline_predictions_frame


def _write_baseline(baselines_dir: Path, ids: list[str]) -> Path:
    baselines_dir.mkdir(parents=True, exist_ok=True)
    path = baselines_dir / "ender.parquet"
    pd.DataFrame(
        {
            "id": ids,
            "era": ["0001"] * len(ids),
            "prediction": [0.1 * (index + 1) for index in range(len(ids))],
        }
    ).to_parquet(path)
    return path


def test_attach_baseline_predictions_frame_joins_on_intersecting_ids(tmp_path: Path) -> None:
    baselines_dir = tmp_path / "baselines"
    _write_baseline(baselines_dir, ["a", "b"])
    full = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "era": ["0001", "0001", "0001"],
            "target": [0.0, 1.0, 2.0],
        }
    )

    attachment = attach_baseline_predictions_frame(
        full,
        baseline_spec={"name": "ender_baseline", "predictions_path": "ender.parquet"},
        era_col="era",
        id_col="id",
        baselines_dir=baselines_dir,
    )

    assert attachment.baseline_col == "ender_baseline"
    assert attachment.baseline_name == "ender_baseline"
    assert attachment.baseline_pred_col == "prediction"
    assert attachment.baseline_predictions_path == str((baselines_dir / "ender.parquet").resolve())
    # The join intersects ids, so the unmatched row is dropped from the returned frame.
    assert sorted(attachment.frame["id"].tolist()) == ["a", "b"]
    assert "ender_baseline" in attachment.frame.columns


def test_attach_baseline_predictions_frame_requires_name_and_path(tmp_path: Path) -> None:
    with pytest.raises(TrainingConfigError, match="training_baseline_config_missing_name_or_predictions_path"):
        attach_baseline_predictions_frame(
            pd.DataFrame({"id": ["a"], "era": ["0001"]}),
            baseline_spec={"name": "ender_baseline"},
            era_col="era",
            id_col="id",
            baselines_dir=tmp_path,
        )


def test_attach_baseline_predictions_frame_requires_id_col(tmp_path: Path) -> None:
    baselines_dir = tmp_path / "baselines"
    _write_baseline(baselines_dir, ["a"])

    with pytest.raises(TrainingConfigError, match="training_id_col_required_for_baseline"):
        attach_baseline_predictions_frame(
            pd.DataFrame({"id": ["a"], "era": ["0001"]}),
            baseline_spec={"name": "ender_baseline", "predictions_path": "ender.parquet"},
            era_col="era",
            id_col="",
            baselines_dir=baselines_dir,
        )
