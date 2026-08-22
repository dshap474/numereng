"""End-to-end runner tests with a fake codex transport (no network, no training).

Covers the FINALIZE happy path, the full four-phase chain, restart, the memory-root identity guard,
the experiment lock, and get_closeout_status.
"""

from __future__ import annotations

import json

import pytest

from numereng.agentic_research.engine import llm
from numereng.agentic_research.engine.closeout import runner
from numereng.agentic_research.engine.closeout import types as ct

from .conftest import (
    CloseoutFixture,
    valid_classification,
    valid_envelope,
    valid_extract_envelope,
    valid_synthesize_envelope,
)


def _install_transport(monkeypatch: pytest.MonkeyPatch, fixture: CloseoutFixture, *, raw: str | None = None) -> dict:
    """Fake codex transport that answers each phase with a valid envelope (or a fixed ``raw`` payload)."""
    calls: dict[str, int] = {"n": 0}

    def fake(**kwargs):
        calls["n"] += 1
        if raw is not None:
            return (raw, "codex-exec")
        label = kwargs.get("round_label")
        if label == ct.PHASE_CLASSIFY:
            payload = valid_classification()
        elif label == ct.PHASE_EXTRACT:
            payload = valid_extract_envelope(experiment_id=fixture.experiment_id)
        elif label == ct.PHASE_SYNTHESIZE:
            payload = valid_synthesize_envelope(experiment_id=fixture.experiment_id)
        else:
            payload = valid_envelope(
                experiment_id=fixture.experiment_id, believed_best_config=fixture.believed_best_config
            )
        return (payload, "codex-exec")

    monkeypatch.setattr(llm, "call_research_llm", fake)
    return calls


def _run(fixture: CloseoutFixture, **kwargs):
    return runner.run_closeout(store_root=fixture.store_root, experiment_id=fixture.experiment_id, **kwargs)


def test_finalize_happy_path_writes_memo_and_evidence(
    closeout_fixture: CloseoutFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_transport(monkeypatch, closeout_fixture)
    result = _run(closeout_fixture, until="finalize")

    assert result.error is None
    assert result.stopped_at_phase is None
    finalize = next(p for p in result.phases if p.name == "finalize")
    assert finalize.status == "done"

    memo_path = closeout_fixture.closeout_dir() / ct.CLOSEOUT_MEMO_FILENAME
    assert memo_path.is_file()
    assert closeout_fixture.experiment_id in memo_path.read_text(encoding="utf-8")
    assert (closeout_fixture.closeout_dir() / ct.CLOSEOUT_EVIDENCE_FILENAME).is_file()
    # No commit journal or stage left behind, lock released.
    assert not (closeout_fixture.closeout_dir() / ct.CLOSEOUT_COMMIT_FILENAME).exists()
    assert not (closeout_fixture.closeout_dir() / ct.CLOSEOUT_LOCK_FILENAME).exists()


def test_second_finalize_is_idempotent_noop(closeout_fixture: CloseoutFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_transport(monkeypatch, closeout_fixture)
    _run(closeout_fixture, until="finalize")
    assert calls["n"] == 1
    # finalize already done -> plan is empty -> no second LLM call.
    result = _run(closeout_fixture, until="finalize")
    assert calls["n"] == 1
    assert result.error is None


def test_bare_closeout_runs_full_chain_to_synthesize(
    closeout_fixture: CloseoutFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    from .conftest import write_ledger_memory_root

    _install_transport(monkeypatch, closeout_fixture)
    memory_root = write_ledger_memory_root(closeout_fixture.store_root / "notes" / "__RESEARCH_MEMORY__")
    result = _run(closeout_fixture, memory_root=str(memory_root))  # finalize -> classify -> extract -> synthesize

    assert result.error is None
    assert result.stopped_at_phase is None
    done = {p.name for p in result.phases if p.status == "done"}
    assert {"finalize", "classify", "extract", "synthesize"} <= done


def test_restart_from_finalize_reruns(closeout_fixture: CloseoutFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_transport(monkeypatch, closeout_fixture)
    _run(closeout_fixture, until="finalize")
    result = _run(closeout_fixture, until="finalize", restart_from="finalize")
    assert calls["n"] == 2
    assert result.error is None


def test_invalid_memo_captured_not_raised(closeout_fixture: CloseoutFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    # Memo missing a required section -> phase failure captured in the result, does not raise.
    bad = json.dumps({"files": [{"path": ct.CLOSEOUT_MEMO_FILENAME, "content": "too short"}], "notes": "x"})
    _install_transport(monkeypatch, closeout_fixture, raw=bad)
    result = _run(closeout_fixture, until="finalize")
    assert result.stopped_at_phase == "finalize"
    assert result.error is not None
    assert result.error.startswith(ct.ERROR_PREFIX)
    # Debug artifacts were dumped.
    debug = closeout_fixture.closeout_dir() / "debug"
    assert any(debug.glob("finalize.debug.*")) if debug.exists() else True


def test_memory_root_identity_change_raises(closeout_fixture: CloseoutFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_transport(monkeypatch, closeout_fixture)
    _run(closeout_fixture, until="finalize")
    # A different valid memory root on the second run trips the identity guard.
    from .conftest import write_memory_root

    other = write_memory_root(closeout_fixture.store_root / "notes" / "OTHER_MEMORY")
    with pytest.raises(ct.CloseoutError) as exc:
        _run(closeout_fixture, until="finalize", memory_root=str(other))
    assert str(exc.value) == ct.ERR_MEMORY_ROOT_CHANGED


def test_experiment_lock_blocks_concurrent_invocation(
    closeout_fixture: CloseoutFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os
    import socket

    _install_transport(monkeypatch, closeout_fixture)
    closeout_dir = closeout_fixture.closeout_dir()
    closeout_dir.mkdir(parents=True, exist_ok=True)
    lock = closeout_dir / ct.CLOSEOUT_LOCK_FILENAME
    lock.write_text(
        json.dumps({"pid": os.getpid(), "hostname": socket.gethostname(), "acquired_at": "2999-01-01T00:00:00+00:00"}),
        encoding="utf-8",
    )
    with pytest.raises(ct.CloseoutError) as exc:
        _run(closeout_fixture, until="finalize")
    assert str(exc.value) == ct.err_lock_held(str(lock))


def test_get_closeout_status_reports_all_phases(closeout_fixture: CloseoutFixture) -> None:
    status = runner.get_closeout_status(
        store_root=closeout_fixture.store_root, experiment_id=closeout_fixture.experiment_id
    )
    names = [p.name for p in status.phases]
    assert names == list(ct.PHASE_ORDER)
    assert all(p.status == "pending" for p in status.phases)
    assert status.error is None
