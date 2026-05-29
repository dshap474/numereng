from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from numereng.features.submission.calibration import (
    build_live_calibration_observations,
    build_live_calibration_report,
    load_live_calibration_observations,
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
    assert result.observation_count == 2
    assert result.model_count == 1
    assert result.scored_row_count == 2
    assert result.scored_observation_count == 2
    rows = load_live_calibration_rows(workspace_root=tmp_path)
    assert rows[0]["model_name"] == "lgbm_m_ender20"
    assert rows[0]["upload_id"] == "pkg-1"
    assert rows[0]["feature_scope"] == "medium"
    assert rows[0]["target"] == "ender20"
    assert rows[0]["local_bmc200_mean"] == 0.123
    assert rows[0]["local_bmc200_sharpe"] == 4.5
    assert rows[0]["live_mmc20"] == -0.0058
    assert rows[0]["live_mmc20_percentile"] == 0.68
    observations = load_live_calibration_observations(workspace_root=tmp_path)
    all_scored = [row for row in observations if row["scope"] == "all_scored"][0]
    assert all_scored["scored_round_count"] == 2
    assert all_scored["live_mmc20"] == -0.0024

    report = load_live_calibration_report(workspace_root=tmp_path)
    assert report.row_count == 2
    assert report.report["scopes"]["all_scored"]["row_count"] == 2
    assert report.report["scopes"]["all_scored"]["observation_count"] == 1
    assert report.report["scopes"]["resolved_only"]["row_count"] == 1


def test_materialize_live_calibration_attributes_rounds_to_upload_windows(tmp_path: Path) -> None:
    submissions_root = tmp_path / ".numereng" / "submissions"
    submission_dir = submissions_root / "lgbm_m_ender20"
    submission_dir.mkdir(parents=True)
    (submission_dir / "submission.json").write_text(
        json.dumps(
            {
                "model_name": "lgbm_m_ender20",
                "model_id": "model-1",
                "uploads": [
                    {
                        "upload_id": "old-upload",
                        "live_started_at": "2026-05-01T00:00:00+00:00",
                        "live_ended_at": "2026-05-09T00:00:00+00:00",
                        "source": {"experiment_id": "exp-1", "package_id": "pkg-old"},
                    },
                    {
                        "upload_id": "new-upload",
                        "live_started_at": "2026-05-09T00:00:00+00:00",
                        "source": {"experiment_id": "exp-1", "package_id": "pkg-new"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame.from_records(
        [
            {"round": 1263, "round_number": 1263, "state": "resolving", "close_date": "2026-05-08", "mmc20": 0.01},
            {"round": 1264, "round_number": 1264, "state": "resolving", "close_date": "2026-05-10", "mmc20": 0.02},
        ]
    ).to_parquet(submission_dir / "live_rounds.parquet", index=False)
    _write_summary(tmp_path, package_id="pkg-old", mean=0.11)
    _write_summary(tmp_path, package_id="pkg-new", mean=0.22)

    result = materialize_live_calibration(workspace_root=tmp_path)

    assert result.row_count == 2
    rows = {row["upload_id"]: row for row in load_live_calibration_rows(workspace_root=tmp_path)}
    assert rows["old-upload"]["round_number"] == 1263
    assert rows["old-upload"]["local_bmc200_mean"] == 0.11
    assert rows["new-upload"]["round_number"] == 1264
    assert rows["new-upload"]["local_bmc200_mean"] == 0.22


def test_live_calibration_report_correlates_upload_observations_not_round_rows() -> None:
    rows = [
        {
            "model_name": "model_a",
            "upload_id": "upload-a",
            "round_number": 1,
            "state": "resolving",
            "has_live_score": True,
            "local_bmc200_mean": 1.0,
            "live_mmc20": 1.0,
        },
        {
            "model_name": "model_a",
            "upload_id": "upload-a",
            "round_number": 2,
            "state": "resolving",
            "has_live_score": True,
            "local_bmc200_mean": 1.0,
            "live_mmc20": 1.0,
        },
        {
            "model_name": "model_a",
            "upload_id": "upload-a",
            "round_number": 3,
            "state": "resolving",
            "has_live_score": True,
            "local_bmc200_mean": 1.0,
            "live_mmc20": 1.0,
        },
        {
            "model_name": "model_b",
            "upload_id": "upload-b",
            "round_number": 1,
            "state": "resolving",
            "has_live_score": True,
            "local_bmc200_mean": 2.0,
            "live_mmc20": 2.0,
        },
        {
            "model_name": "model_c",
            "upload_id": "upload-c",
            "round_number": 1,
            "state": "resolving",
            "has_live_score": True,
            "local_bmc200_mean": 3.0,
            "live_mmc20": 3.0,
        },
    ]

    observations = build_live_calibration_observations(rows)
    report = build_live_calibration_report(rows, observations=observations, generated_at="2026-05-29T00:00:00+00:00")

    assert report["row_count"] == 5
    assert report["scopes"]["all_scored"]["observation_count"] == 3
    stats = report["scopes"]["all_scored"]["correlations"]["local_bmc200_mean"]["live_mmc20"]
    assert stats["n"] == 3
    assert stats["pearson_r"] == 1.0


def test_materialize_live_calibration_dry_run_does_not_write_artifacts(tmp_path: Path) -> None:
    submission_dir = tmp_path / ".numereng" / "submissions" / "model_a"
    submission_dir.mkdir(parents=True)
    (submission_dir / "submission.json").write_text(
        json.dumps({"model_name": "model_a", "offline_snapshot": {"local_bmc_last_200_mean": 0.1}}),
        encoding="utf-8",
    )
    pd.DataFrame.from_records([{"round": 1, "round_number": 1, "state": "resolving", "mmc20": 0.01}]).to_parquet(
        submission_dir / "live_rounds.parquet",
        index=False,
    )

    result = materialize_live_calibration(workspace_root=tmp_path, dry_run=True)

    assert result.dry_run is True
    assert result.row_count == 1
    assert not result.rows_path.exists()
    assert not result.observations_path.exists()
    assert not result.report_path.exists()


def _write_summary(tmp_path: Path, *, package_id: str, mean: float) -> None:
    summaries_dir = (
        tmp_path
        / ".numereng"
        / "experiments"
        / "exp-1"
        / "submission_packages"
        / package_id
        / "artifacts"
        / "eval"
        / "validation"
        / "pickle"
    )
    summaries_dir.mkdir(parents=True)
    (summaries_dir / "summaries.json").write_text(
        json.dumps({"bmc_last_200_eras_target_ender_20": {"mean": mean}}),
        encoding="utf-8",
    )
