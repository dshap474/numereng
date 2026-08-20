"""Phase-0 strict evidence tests: a corrupt record fails closeout hard with a stable token."""

from __future__ import annotations

import json

import pytest

from numereng.agentic_research.engine.closeout import evidence
from numereng.agentic_research.engine.closeout import types as ct

from .conftest import CloseoutFixture


def _build(fixture: CloseoutFixture) -> dict:
    experiment = fixture.experiment()
    state = json.loads(fixture.state_path().read_text(encoding="utf-8"))
    return evidence.build_evidence(experiment=experiment, state=state, runs_dir=fixture.store_root / "runs")


def test_build_evidence_happy_path(closeout_fixture: CloseoutFixture) -> None:
    summary = _build(closeout_fixture)
    assert summary["experiment_id"] == closeout_fixture.experiment_id
    assert summary["believed_best"]["config"] == "config_001.json"
    assert summary["totals"]["completed"] == 3
    assert summary["sweep_abandoned"]["count"] == 1
    # runs are not pulled in unit fixtures: enrichment is marked, never silently dropped.
    for run_metrics in summary["metrics_enrichment"].values():
        assert all(value == "unavailable: run not pulled" for value in run_metrics.values())


def test_malformed_journal_line_raises(closeout_fixture: CloseoutFixture) -> None:
    with closeout_fixture.journal_path().open("a", encoding="utf-8") as handle:
        handle.write("{not valid json\n")
    with pytest.raises(ct.CloseoutError) as exc:
        _build(closeout_fixture)
    assert str(exc.value) == ct.err_journal_malformed(4)


def test_completed_entry_missing_config_file_raises(closeout_fixture: CloseoutFixture) -> None:
    with closeout_fixture.journal_path().open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"round": 4, "config": "ghost.json", "seed": 99, "metric": 0.003, "status": "completed"}) + "\n"
        )
    with pytest.raises(ct.CloseoutError) as exc:
        _build(closeout_fixture)
    assert str(exc.value) == ct.err_journal_entry_invalid(4, "config")


def test_completed_entry_non_numeric_metric_raises(closeout_fixture: CloseoutFixture) -> None:
    with closeout_fixture.journal_path().open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"round": 4, "config": "config_001.json", "seed": 99, "metric": "n/a", "status": "completed"})
            + "\n"
        )
    with pytest.raises(ct.CloseoutError) as exc:
        _build(closeout_fixture)
    assert str(exc.value) == ct.err_journal_entry_invalid(4, "metric")


def test_no_completed_rounds_raises_leaderboard_empty(closeout_fixture: CloseoutFixture) -> None:
    rows = [
        {
            "round": 1,
            "config": "config_001.json",
            "seed": 42,
            "status": "failed",
            "error": "agentic_research_codex_failed:1",
        },
    ]
    with closeout_fixture.journal_path().open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    with pytest.raises(ct.CloseoutError) as exc:
        _build(closeout_fixture)
    assert str(exc.value) == ct.ERR_LEADERBOARD_EMPTY


def test_believed_best_unresolved_raises(closeout_fixture: CloseoutFixture) -> None:
    state = json.loads(closeout_fixture.state_path().read_text(encoding="utf-8"))
    state["believed_best"] = {"config": "config_999.json"}
    closeout_fixture.state_path().write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ct.CloseoutError) as exc:
        _build(closeout_fixture)
    assert str(exc.value) == ct.ERR_BELIEVED_BEST_UNRESOLVED
