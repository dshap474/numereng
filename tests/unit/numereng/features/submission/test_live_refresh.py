from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from numereng.features.submission.live_refresh import refresh_submission_snapshots


class _FakeRefreshClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get_models(self) -> dict[str, str]:
        return {"model_a": "model-1"}

    def round_model_performances_v2(self, *, model_id: str) -> list[dict[str, Any]]:
        self.calls.append(model_id)
        return [
            {
                "roundNumber": 1264,
                "roundOpenTime": "2026-05-09T12:24:12+00:00",
                "roundCloseTime": "2026-05-10T12:24:12+00:00",
                "roundResolveTime": "2026-06-11T20:00:00+00:00",
                "roundResolved": False,
                "roundPayoutFactor": "0.092335072311476494",
                "atRisk": "0E-18",
                "corrMultiplier": 0.75,
                "mmcMultiplier": 2.25,
                "submissionScores": [
                    {"displayName": "bmc", "value": 0.0027, "percentile": 0.74},
                    {"displayName": "mmc", "value": -0.0058, "percentile": 0.68},
                    {"displayName": "v2_corr20", "value": -0.0029, "percentile": 0.72},
                    {"displayName": "fnc_v3", "value": 0.0068, "percentile": 0.78},
                ],
            },
            {
                "roundNumber": 1234,
                "roundOpenTime": "2026-03-30T12:24:12+00:00",
                "roundResolveTime": "2026-04-30T20:00:00+00:00",
                "roundResolved": True,
                "submissionScores": [{"displayName": "canon_bmc", "value": 0.0012}],
            },
        ]


def test_refresh_submission_snapshots_writes_live_rounds_and_metadata(tmp_path: Path) -> None:
    submission_dir = tmp_path / ".numereng" / "submissions" / "model_a"
    submission_dir.mkdir(parents=True)
    (submission_dir / "submission.json").write_text(
        json.dumps({"model_name": "model_a", "source": {"experiment_id": "exp-1"}}),
        encoding="utf-8",
    )
    client = _FakeRefreshClient()

    results = refresh_submission_snapshots(workspace_root=tmp_path, client=client)

    assert client.calls == ["model-1"]
    assert len(results) == 1
    assert results[0].round_count == 2
    assert results[0].scored_round_count == 2
    assert results[0].resolved_round_count == 1
    assert results[0].resolved_scored_round_count == 1

    frame = pd.read_parquet(submission_dir / "live_rounds.parquet")
    assert frame["round"].tolist() == [1264, 1234]
    assert frame.loc[0, "state"] == "resolving"
    assert frame.loc[0, "close_date"] == "2026-05-10"
    assert frame.loc[0, "bmc"] == 0.0027
    assert frame.loc[0, "bmc_percentile"] == 0.74
    assert frame.loc[0, "mmc20"] == -0.0058
    assert frame.loc[0, "mmc20_percentile"] == 0.68
    assert frame.loc[0, "corr20"] == -0.0029
    assert frame.loc[0, "corr20_percentile"] == 0.72
    assert frame.loc[0, "fnc"] == 0.0068
    assert frame.loc[0, "fnc_percentile"] == 0.78

    metadata = json.loads((submission_dir / "submission.json").read_text(encoding="utf-8"))
    assert metadata["source"] == {"experiment_id": "exp-1"}
    assert metadata["model_id"] == "model-1"
    assert metadata["status"] == "live_scores_available"
    assert metadata["refresh"]["source"] == "numerai.round_model_performances_v2"
    assert metadata["refresh"]["round_count"] == 2
    assert metadata["refresh"]["latest_scored_round"] == 1264
    assert metadata["refresh"]["latest_resolved_round"] == 1234


def test_refresh_submission_snapshots_skips_missing_account_model(tmp_path: Path) -> None:
    submission_dir = tmp_path / ".numereng" / "submissions" / "unknown_model"
    submission_dir.mkdir(parents=True)
    (submission_dir / "submission.json").write_text("{}", encoding="utf-8")
    client = _FakeRefreshClient()

    results = refresh_submission_snapshots(workspace_root=tmp_path, client=client)

    assert results[0].skipped is True
    assert results[0].warning == "model_not_found_in_numerai_account"
    assert client.calls == []
    assert not (submission_dir / "live_rounds.parquet").exists()
