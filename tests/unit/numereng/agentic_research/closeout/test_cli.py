"""CLI smoke + exit-code contract for `research closeout` / `research closeout-status`."""

from __future__ import annotations

import pytest

from numereng.agentic_research.engine import llm
from numereng.agentic_research.engine.closeout import types as ct
from numereng.cli.commands.research import handle_research_command

from .conftest import CloseoutFixture, valid_classification, valid_envelope


def _install_transport(monkeypatch: pytest.MonkeyPatch, fixture: CloseoutFixture) -> None:
    def fake(**kwargs):
        if kwargs.get("round_label") == ct.PHASE_CLASSIFY:
            return valid_classification(), "codex-exec"
        return (
            valid_envelope(experiment_id=fixture.experiment_id, believed_best_config=fixture.believed_best_config),
            "codex-exec",
        )

    monkeypatch.setattr(llm, "call_research_llm", fake)


def test_cli_closeout_until_finalize_exit_zero(
    closeout_fixture: CloseoutFixture, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install_transport(monkeypatch, closeout_fixture)
    code = handle_research_command(
        [
            "closeout",
            "--experiment-id",
            closeout_fixture.experiment_id,
            "--until",
            "finalize",
            "--workspace",
            str(closeout_fixture.store_root.parent),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "finalize | done" in out


def test_cli_closeout_until_classify_exit_zero(
    closeout_fixture: CloseoutFixture,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_transport(monkeypatch, closeout_fixture)

    code = handle_research_command(
        [
            "closeout",
            "--experiment-id",
            closeout_fixture.experiment_id,
            "--until",
            "classify",
            "--workspace",
            str(closeout_fixture.store_root.parent),
        ]
    )

    assert code == 0
    assert "classify | done" in capsys.readouterr().out


def test_cli_bare_closeout_stops_at_extract_exit_one(
    closeout_fixture: CloseoutFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_transport(monkeypatch, closeout_fixture)
    code = handle_research_command(
        [
            "closeout",
            "--experiment-id",
            closeout_fixture.experiment_id,
            "--workspace",
            str(closeout_fixture.store_root.parent),
        ]
    )
    assert code == 1


def test_cli_closeout_status_exit_zero(closeout_fixture: CloseoutFixture) -> None:
    code = handle_research_command(
        [
            "closeout-status",
            "--experiment-id",
            closeout_fixture.experiment_id,
            "--workspace",
            str(closeout_fixture.store_root.parent),
        ]
    )
    assert code == 0


def test_cli_missing_experiment_id_exit_two(closeout_fixture: CloseoutFixture) -> None:
    assert handle_research_command(["closeout"]) == 2


def test_cli_bad_until_exit_two(closeout_fixture: CloseoutFixture) -> None:
    code = handle_research_command(["closeout", "--experiment-id", closeout_fixture.experiment_id, "--until", "bogus"])
    assert code == 2


def test_cli_nonexistent_experiment_exit_one(tmp_path) -> None:
    code = handle_research_command(["closeout-status", "--experiment-id", "__nope__", "--workspace", str(tmp_path)])
    assert code == 1


def test_cli_closeout_format_json_emits_machine_readable(
    closeout_fixture: CloseoutFixture, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    _install_transport(monkeypatch, closeout_fixture)
    code = handle_research_command(
        [
            "closeout",
            "--experiment-id",
            closeout_fixture.experiment_id,
            "--until",
            "finalize",
            "--format",
            "json",
            "--workspace",
            str(closeout_fixture.store_root.parent),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["experiment_id"] == closeout_fixture.experiment_id
    assert payload["error"] is None
    assert any(phase["name"] == "finalize" and phase["status"] == "done" for phase in payload["phases"])


def test_cli_closeout_status_format_json_emits_machine_readable(
    closeout_fixture: CloseoutFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    code = handle_research_command(
        [
            "closeout-status",
            "--experiment-id",
            closeout_fixture.experiment_id,
            "--format",
            "json",
            "--workspace",
            str(closeout_fixture.store_root.parent),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["experiment_id"] == closeout_fixture.experiment_id
    assert [phase["name"] for phase in payload["phases"]]


def test_cli_bad_format_exit_two(closeout_fixture: CloseoutFixture) -> None:
    code = handle_research_command(
        ["closeout-status", "--experiment-id", closeout_fixture.experiment_id, "--format", "bogus"]
    )
    assert code == 2
