"""Gate tests: archived, non-agentic, empty journal, live run, unmet budget.

Every gate failure raises ``CloseoutError``; ``--allow-incomplete`` waives the two run gates.
"""

from __future__ import annotations

import pytest

from numereng.agentic_research.engine.closeout import runner
from numereng.agentic_research.engine.closeout import types as ct

from .conftest import CloseoutFixture, install_fake_llm


def _run(fixture: CloseoutFixture, **kwargs):
    return runner.run_closeout(store_root=fixture.store_root, experiment_id=fixture.experiment_id, **kwargs)


def test_gate_rejects_missing_agentic_state(closeout_fixture: CloseoutFixture) -> None:
    closeout_fixture.state_path().unlink()
    with pytest.raises(ct.CloseoutError) as exc:
        _run(closeout_fixture)
    assert str(exc.value) == ct.ERR_NOT_AGENTIC


def test_gate_rejects_empty_journal(closeout_fixture: CloseoutFixture) -> None:
    closeout_fixture.journal_path().write_text("", encoding="utf-8")
    with pytest.raises(ct.CloseoutError) as exc:
        _run(closeout_fixture)
    assert str(exc.value) == ct.ERR_NO_ROUNDS


def test_gate_rejects_archived_experiment(closeout_fixture: CloseoutFixture) -> None:
    closeout_fixture.set_manifest_status("archived")
    with pytest.raises(ct.CloseoutError) as exc:
        _run(closeout_fixture)
    assert str(exc.value) == ct.ERR_EXPERIMENT_ARCHIVED


def test_gate_rejects_running_experiment(closeout_fixture: CloseoutFixture) -> None:
    closeout_fixture.set_run_status("running")
    with pytest.raises(ct.CloseoutError) as exc:
        _run(closeout_fixture)
    assert str(exc.value) == ct.ERR_RUN_ACTIVE


def test_gate_accepts_running_with_allow_incomplete(
    closeout_fixture: CloseoutFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    closeout_fixture.set_run_status("running")
    install_fake_llm(monkeypatch, closeout_fixture)
    result = _run(closeout_fixture, allow_incomplete=True)
    assert result.memo_path.is_file()


def test_gate_rejects_budget_not_reached(closeout_fixture: CloseoutFixture) -> None:
    closeout_fixture.set_budget(10)
    with pytest.raises(ct.CloseoutError) as exc:
        _run(closeout_fixture)
    assert str(exc.value) == f"{ct.ERROR_PREFIX}budget_not_reached:3/10"


def test_gate_allows_incomplete_budget(closeout_fixture: CloseoutFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    closeout_fixture.set_budget(10)
    install_fake_llm(monkeypatch, closeout_fixture)
    result = _run(closeout_fixture, allow_incomplete=True)
    assert result.memo_path.is_file()
