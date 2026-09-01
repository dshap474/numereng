"""CLI smoke + exit-code contract for `research closeout` and the `research status` memo line."""

from __future__ import annotations

import json

import pytest

from numereng.agentic_research.engine.closeout import types as ct
from numereng.cli.commands.research import handle_research_command

from .conftest import CloseoutFixture, install_fake_llm


def _workspace(fixture: CloseoutFixture) -> str:
    return str(fixture.store_root.parent)


def test_cli_closeout_exit_zero_and_prints_paths(
    closeout_fixture: CloseoutFixture, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install_fake_llm(monkeypatch, closeout_fixture)
    code = handle_research_command(
        ["closeout", "--experiment-id", closeout_fixture.experiment_id, "--workspace", _workspace(closeout_fixture)]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert ct.CLOSEOUT_MEMO_FILENAME in out
    assert ct.CLOSEOUT_EVIDENCE_FILENAME in out


def test_cli_closeout_format_json_emits_machine_readable(
    closeout_fixture: CloseoutFixture, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install_fake_llm(monkeypatch, closeout_fixture)
    code = handle_research_command(
        [
            "closeout",
            "--experiment-id",
            closeout_fixture.experiment_id,
            "--format",
            "json",
            "--workspace",
            _workspace(closeout_fixture),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["experiment_id"] == closeout_fixture.experiment_id
    assert payload["memo_path"].endswith(ct.CLOSEOUT_MEMO_FILENAME)


def test_cli_closeout_running_experiment_exit_one(
    closeout_fixture: CloseoutFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    closeout_fixture.set_run_status("running")
    code = handle_research_command(
        ["closeout", "--experiment-id", closeout_fixture.experiment_id, "--workspace", _workspace(closeout_fixture)]
    )
    assert code == 1
    assert ct.ERR_RUN_ACTIVE in capsys.readouterr().err


def test_cli_closeout_allow_incomplete_exit_zero(
    closeout_fixture: CloseoutFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    closeout_fixture.set_run_status("running")
    install_fake_llm(monkeypatch, closeout_fixture)
    code = handle_research_command(
        [
            "closeout",
            "--experiment-id",
            closeout_fixture.experiment_id,
            "--allow-incomplete",
            "--workspace",
            _workspace(closeout_fixture),
        ]
    )
    assert code == 0


def test_cli_missing_experiment_id_exit_two(closeout_fixture: CloseoutFixture) -> None:
    assert handle_research_command(["closeout"]) == 2


def test_cli_bad_format_exit_two(closeout_fixture: CloseoutFixture) -> None:
    code = handle_research_command(["closeout", "--experiment-id", closeout_fixture.experiment_id, "--format", "bogus"])
    assert code == 2


def test_cli_nonexistent_experiment_exit_one(tmp_path) -> None:
    assert handle_research_command(["closeout", "--experiment-id", "__nope__", "--workspace", str(tmp_path)]) == 1


def test_cli_status_reports_closeout_memo_presence(
    closeout_fixture: CloseoutFixture, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["status", "--experiment-id", closeout_fixture.experiment_id, "--workspace", _workspace(closeout_fixture)]
    assert handle_research_command(args) == 0
    assert "closeout_memo: absent" in capsys.readouterr().out

    install_fake_llm(monkeypatch, closeout_fixture)
    handle_research_command(
        ["closeout", "--experiment-id", closeout_fixture.experiment_id, "--workspace", _workspace(closeout_fixture)]
    )
    capsys.readouterr()

    assert handle_research_command(args) == 0
    assert "closeout_memo: present" in capsys.readouterr().out
