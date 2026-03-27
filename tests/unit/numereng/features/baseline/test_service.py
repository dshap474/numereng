from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from numereng.features.baseline import BaselineBuildRequest, build_baseline


class _FakeClient:
    def download_dataset(
        self,
        filename: str,
        *,
        dest_path: str | None = None,
        round_num: int | None = None,
    ) -> str:
        raise AssertionError(f"unexpected dataset download: {filename}")


def test_build_baseline_writes_named_baseline_and_promotes_active(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    runs_root = store_root / "runs"
    data_root = store_root / "datasets" / "v5.2"
    runs_root.mkdir(parents=True)
    data_root.mkdir(parents=True)

    prediction_rows = [
        {"era": "0001", "id": "a"},
        {"era": "0001", "id": "b"},
        {"era": "0001", "id": "c"},
        {"era": "0002", "id": "d"},
        {"era": "0002", "id": "e"},
        {"era": "0002", "id": "f"},
    ]
    validation = pd.DataFrame(
        [
            {
                **row,
                "target_ender_20": value20,
                "target_ender_60": value60,
                "data_type": "validation",
            }
            for row, value20, value60 in zip(
                prediction_rows,
                [0.1, 0.3, 0.2, 0.4, 0.5, 0.6],
                [0.6, 0.4, 0.5, 0.3, 0.2, 0.1],
                strict=True,
            )
        ]
    )
    validation.to_parquet(data_root / "validation.parquet", index=False)
    pd.DataFrame(
        [
            {
                "era": "9999",
                "id": "train-only",
                "target_ender_20": 0.0,
                "target_ender_60": 0.0,
                "data_type": "train",
            }
        ]
    ).to_parquet(data_root / "train.parquet", index=False)

    run_payloads = {
        "run20a": ("target_ender_20", 42, [0.10, 0.40, 0.30, 0.20, 0.60, 0.50]),
        "run20b": ("target_ender_20", 43, [0.20, 0.60, 0.10, 0.30, 0.50, 0.40]),
        "run20c": ("target_ender_20", 44, [0.30, 0.50, 0.20, 0.60, 0.40, 0.10]),
        "run60a": ("target_ender_60", 42, [0.60, 0.30, 0.10, 0.50, 0.20, 0.40]),
        "run60b": ("target_ender_60", 43, [0.50, 0.20, 0.40, 0.60, 0.10, 0.30]),
        "run60c": ("target_ender_60", 44, [0.40, 0.10, 0.60, 0.30, 0.50, 0.20]),
    }

    for run_id, (target_col, seed, predictions) in run_payloads.items():
        run_dir = runs_root / run_id
        run_dir.mkdir(parents=True)
        prediction_frame = pd.DataFrame(
            [{**row, "prediction": pred} for row, pred in zip(prediction_rows, predictions, strict=True)]
        )
        prediction_frame.to_parquet(run_dir / "predictions.parquet", index=False)
        (run_dir / "run.json").write_text(
            json.dumps(
                {
                    "artifacts": {"predictions": "predictions.parquet"},
                    "data": {
                        "feature_set": "medium",
                        "target_col": target_col,
                        "version": "v5.2",
                    },
                    "experiment_id": "2026-03-15_medium-deep-lgbm-ender20-ender60-3seed",
                    "training": {
                        "data": {
                            "dataset_scope": "train_plus_validation",
                            "dataset_variant": "non_downsampled",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "resolved.json").write_text(
            json.dumps(
                {
                    "data": {
                        "data_version": "v5.2",
                        "era_col": "era",
                        "id_col": "id",
                    },
                    "model": {"params": {"random_state": seed}},
                }
            ),
            encoding="utf-8",
        )

    result = build_baseline(
        store_root=store_root,
        request=BaselineBuildRequest(
            run_ids=tuple(run_payloads.keys()),
            name="medium_ender20_ender60_6run_blend",
            default_target="target_ender_20",
            promote_active=True,
        ),
        client=_FakeClient(),
    )

    assert result.name == "medium_ender20_ender60_6run_blend"
    assert result.default_target == "target_ender_20"
    assert result.available_targets == ("target_ender_20", "target_ender_60")
    assert result.source_experiment_id == "2026-03-15_medium-deep-lgbm-ender20-ender60-3seed"
    assert result.active_predictions_path is not None
    assert result.active_metadata_path is not None

    baseline_predictions = pd.read_parquet(result.predictions_path)
    expected = (
        pd.DataFrame(
            {
                "era": [row["era"] for row in prediction_rows],
                "pred_run20a": run_payloads["run20a"][2],
                "pred_run20b": run_payloads["run20b"][2],
                "pred_run20c": run_payloads["run20c"][2],
                "pred_run60a": run_payloads["run60a"][2],
                "pred_run60b": run_payloads["run60b"][2],
                "pred_run60c": run_payloads["run60c"][2],
            }
        )
        .groupby("era", group_keys=False)[
            ["pred_run20a", "pred_run20b", "pred_run20c", "pred_run60a", "pred_run60b", "pred_run60c"]
        ]
        .rank(pct=True)
        .mean(axis=1)
    )
    assert baseline_predictions["prediction"].tolist() == pytest.approx(expected.tolist())

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["default_target"] == "target_ender_20"
    assert metadata["available_targets"] == ["target_ender_20", "target_ender_60"]
    assert metadata["source_run_ids"] == list(run_payloads.keys())
    assert sorted(metadata["source_seeds"]) == [42, 43, 44]
    assert metadata["source_families"] == {
        "ender20": ["run20a", "run20b", "run20c"],
        "ender60": ["run60a", "run60b", "run60c"],
    }
    assert Path(metadata["artifacts"]["per_era_corr_target_ender_20"]).is_file()
    assert Path(metadata["artifacts"]["per_era_corr_target_ender_60"]).is_file()

    active_metadata = json.loads(result.active_metadata_path.read_text(encoding="utf-8"))
    assert active_metadata["name"] == "medium_ender20_ender60_6run_blend"
    assert active_metadata["predictions_file"] == "predictions.parquet"
    assert Path(active_metadata["source_path"]) == result.baseline_dir
