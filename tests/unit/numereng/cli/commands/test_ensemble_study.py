"""CLI tests for `numereng ensemble study freeze|run|finalize|status` (exit codes 0/1/2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from numereng.cli.main import main
from tests.unit.numereng.features.research_portfolio import _portfolio_fixtures as fx


def _write_freeze(store: fx.Store, tmp_path: Path, **kwargs: object) -> Path:
    return fx.write_json_file(tmp_path, fx.freeze_payload(store, **kwargs), name="freeze.json")


def _write_trials(tmp_path: Path, **kwargs: object) -> Path:
    return fx.write_json_file(tmp_path, fx.trials_payload(**kwargs), name="trials.json")


def test_freeze_json_exit_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    store = fx.build_study_store(tmp_path)
    config = _write_freeze(store, tmp_path)
    rc = main(
        ["ensemble", "study", "freeze", "--workspace", str(tmp_path), "--config", str(config), "--format", "json"]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["frozen"] is True
    assert payload["n_members"] == 2


def test_run_finalize_status_flow_exit_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    store = fx.build_study_store(tmp_path)
    config = _write_freeze(store, tmp_path)
    trials = _write_trials(tmp_path)
    assert main(["ensemble", "study", "freeze", "--workspace", str(tmp_path), "--config", str(config)]) == 0
    capsys.readouterr()

    rc = main(["ensemble", "study", "run", "--workspace", str(tmp_path), "--trials", str(trials), "--format", "json"])
    assert rc == 0
    run_payload = json.loads(capsys.readouterr().out)
    assert run_payload["executed"] == 1

    rc = main(
        [
            "ensemble",
            "study",
            "finalize",
            "--workspace",
            str(tmp_path),
            "--study-id",
            "S1",
            "--select",
            "trial_a",
            "--format",
            "json",
        ]
    )
    assert rc == 0
    final_payload = json.loads(capsys.readouterr().out)
    assert final_payload["sealed"] is True
    assert final_payload["selected_trial"] == "trial_a"

    rc = main(["ensemble", "study", "status", "--workspace", str(tmp_path), "--study-id", "S1", "--format", "json"])
    assert rc == 0
    status_payload = json.loads(capsys.readouterr().out)
    assert status_payload["sealed"] is True


def test_status_table_exit_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    store = fx.build_study_store(tmp_path)
    config = _write_freeze(store, tmp_path)
    main(["ensemble", "study", "freeze", "--workspace", str(tmp_path), "--config", str(config)])
    capsys.readouterr()
    rc = main(["ensemble", "study", "status", "--workspace", str(tmp_path), "--study-id", "S1", "--format", "table"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "study_id: S1" in out
    assert "frozen: True" in out


def test_blocked_freeze_exits_one(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    store = fx.build_study_store(tmp_path, policy_filled=False)
    config = _write_freeze(store, tmp_path)
    rc = main(["ensemble", "study", "freeze", "--workspace", str(tmp_path), "--config", str(config)])
    assert rc == 1
    assert "policy_unset" in capsys.readouterr().err


def test_missing_config_exits_two(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fx.build_study_store(tmp_path)
    rc = main(["ensemble", "study", "freeze", "--workspace", str(tmp_path)])
    assert rc == 2
    assert "missing required argument: --config" in capsys.readouterr().err


def test_bad_format_exits_two(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    store = fx.build_study_store(tmp_path)
    config = _write_freeze(store, tmp_path)
    rc = main(["ensemble", "study", "freeze", "--workspace", str(tmp_path), "--config", str(config), "--format", "xml"])
    assert rc == 2
    assert "invalid value for --format" in capsys.readouterr().err


def test_unknown_subcommand_exits_two(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["ensemble", "study", "banana"])
    assert rc == 2
