from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from numereng.features.submission.calibration import (
    load_live_calibration_report,
    load_live_calibration_rows,
    materialize_live_calibration,
)


def test_materialize_live_calibration_joins_submission_rounds_to_package_metrics(tmp_path: Path) -> None:
    submissions_root = tmp_path / ".numereng" / "submissions"
    submission_dir = submissions_root / "lgbm_m_ender20"
    submission_dir.mkdir(parents=True)
    (submission_dir / "submission.json").write_text(
        json.dumps(
            {
                "model_name": "lgbm_m_ender20",
                "model_id": "model-1",
                "hosted_pickle": {"uploaded_at": "2026-05-09T00:00:00+00:00"},
                "source": {
                    "experiment_id": "exp-1",
                    "package_id": "pkg-1",
                    "recipe": "moderate_lgbm",
                },
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame.from_records(
        [
            {
                "round": 1264,
                "round_number": 1264,
                "state": "resolving",
                "close_date": "2026-05-10",
                "resolve_date": "2026-06-11",
                "mmc20": -0.0058,
                "mmc20_percentile": 0.68,
                "corr20": -0.0029,
                "corr20_percentile": 0.72,
                "pulled_at": "2026-05-16T00:00:00+00:00",
                "is_estimate": True,
            },
            {
                "round": 1234,
                "round_number": 1234,
                "state": "resolved",
                "mmc20": 0.001,
                "corr20": 0.002,
                "is_estimate": False,
            },
        ]
    ).to_parquet(submission_dir / "live_rounds.parquet", index=False)

    summaries_dir = (
        tmp_path
        / ".numereng"
        / "experiments"
        / "exp-1"
        / "submission_packages"
        / "pkg-1"
        / "artifacts"
        / "eval"
        / "validation"
        / "pickle"
    )
    summaries_dir.mkdir(parents=True)
    (summaries_dir / "summaries.json").write_text(
        json.dumps(
            {
                "bmc_last_200_eras_target_ender_20": {
                    "mean": 0.123,
                    "sharpe": 4.5,
                    "max_drawdown": -0.01,
                },
                "bmc_target_ender_20": {"mean": 0.111},
                "corr_target_ender_20": {"mean": 0.222},
                "mmc_target_ender_20": {"mean": 0.333},
                "fnc_target_ender_20": {"mean": 0.444},
            }
        ),
        encoding="utf-8",
    )

    result = materialize_live_calibration(workspace_root=tmp_path)

    assert result.row_count == 2
    assert result.model_count == 1
    assert result.scored_row_count == 2
    rows = load_live_calibration_rows(workspace_root=tmp_path)
    assert rows[0]["model_name"] == "lgbm_m_ender20"
    assert rows[0]["feature_scope"] == "medium"
    assert rows[0]["target"] == "ender20"
    assert rows[0]["local_bmc200_mean"] == 0.123
    assert rows[0]["local_bmc200_sharpe"] == 4.5
    assert rows[0]["live_mmc20"] == -0.0058
    assert rows[0]["live_mmc20_percentile"] == 0.68

    report = load_live_calibration_report(workspace_root=tmp_path)
    assert report.row_count == 2
    assert report.report["scopes"]["all_scored"]["row_count"] == 2
    assert report.report["scopes"]["resolved_only"]["row_count"] == 1
