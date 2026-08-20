"""Gate tests: status, budget, memory root. Gate failures raise CloseoutError (not captured)."""

from __future__ import annotations

import pytest

from numereng.agentic_research.engine.closeout import runner
from numereng.agentic_research.engine.closeout import types as ct

from .conftest import CloseoutFixture


def _run(fixture: CloseoutFixture, **kwargs):
    return runner.run_closeout(
        store_root=fixture.store_root, experiment_id=fixture.experiment_id, until="finalize", **kwargs
    )


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


def test_gate_rejects_running_state_unless_accepted(closeout_fixture: CloseoutFixture) -> None:
    closeout_fixture.set_run_status("running")
    with pytest.raises(ct.CloseoutError) as exc:
        _run(closeout_fixture)
    assert str(exc.value) == ct.ERR_RUN_ACTIVE


def test_gate_accepts_running_with_override(closeout_fixture: CloseoutFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    closeout_fixture.set_run_status("running")
    _install_fake_transport(monkeypatch, closeout_fixture)
    result = _run(closeout_fixture, accept_stale_running=True)
    assert result.error is None


def test_gate_rejects_budget_not_reached(closeout_fixture: CloseoutFixture) -> None:
    closeout_fixture.set_budget(10)
    with pytest.raises(ct.CloseoutError) as exc:
        _run(closeout_fixture)
    assert str(exc.value) == ct.err_budget_not_reached(3, 10)


def test_gate_allows_incomplete_budget_with_override(
    closeout_fixture: CloseoutFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    closeout_fixture.set_budget(10)
    _install_fake_transport(monkeypatch, closeout_fixture)
    result = _run(closeout_fixture, allow_incomplete=True)
    assert result.error is None


def test_gate_rejects_invalid_memory_root(closeout_fixture: CloseoutFixture) -> None:
    (closeout_fixture.memory_root / "CURRENT.md").unlink()
    with pytest.raises(ct.CloseoutError) as exc:
        _run(closeout_fixture)
    assert str(exc.value) == ct.ERR_MEMORY_ROOT_INVALID


def test_gate_rejects_missing_topic_ledger(closeout_fixture: CloseoutFixture) -> None:
    (closeout_fixture.memory_root / "topics" / "targets.md").unlink()
    with pytest.raises(ct.CloseoutError) as exc:
        _run(closeout_fixture)
    assert str(exc.value) == ct.ERR_MEMORY_ROOT_INVALID


def test_until_invalid_raises(closeout_fixture: CloseoutFixture) -> None:
    with pytest.raises(ct.CloseoutError) as exc:
        runner.run_closeout(
            store_root=closeout_fixture.store_root, experiment_id=closeout_fixture.experiment_id, until="bogus"
        )
    assert str(exc.value) == ct.err_until_invalid("bogus")


def test_restart_from_invalid_phase_raises(closeout_fixture: CloseoutFixture) -> None:
    # All phases are implemented; an unknown restart_from phase is the live rejection path.
    with pytest.raises(ct.CloseoutError) as exc:
        runner.run_closeout(
            store_root=closeout_fixture.store_root,
            experiment_id=closeout_fixture.experiment_id,
            restart_from="bogus",
        )
    assert str(exc.value) == ct.err_restart_from_invalid("bogus")


# --------------------------------------------------------------------------- #
# Shared fake transport
# --------------------------------------------------------------------------- #
def _install_fake_transport(monkeypatch: pytest.MonkeyPatch, fixture: CloseoutFixture) -> None:
    from numereng.agentic_research.engine import llm

    from .conftest import valid_envelope

    def fake(**kwargs):
        return (
            valid_envelope(experiment_id=fixture.experiment_id, believed_best_config=fixture.believed_best_config),
            "codex-exec",
        )

    monkeypatch.setattr(llm, "_call_research_llm", fake)
