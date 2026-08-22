"""Runner-level EXTRACT + SYNTHESIZE tests (§3.2/§3.3, §2.4 restart, §2.6 memory-root lock).

Uses a fake codex transport (no network) and a temporary research-memory root. The real
``.numereng/notes/__RESEARCH_MEMORY__/`` is never touched: every test writes into tmp_path.
"""

from __future__ import annotations

import json
import os
import socket

import pytest

from numereng.agentic_research.engine import llm
from numereng.agentic_research.engine.closeout import merge, runner
from numereng.agentic_research.engine.closeout import types as ct

from .conftest import (
    TOPIC_NAMES,
    CloseoutFixture,
    valid_classification,
    valid_envelope,
    valid_extract_envelope,
    valid_synthesize_envelope,
    write_ledger_memory_root,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _install_transport(monkeypatch: pytest.MonkeyPatch, fixture: CloseoutFixture) -> None:
    def fake(**kwargs):
        label = kwargs.get("round_label")
        if label == ct.PHASE_CLASSIFY:
            return (valid_classification(), "codex-exec")
        if label == ct.PHASE_EXTRACT:
            return (valid_extract_envelope(experiment_id=fixture.experiment_id), "codex-exec")
        if label == ct.PHASE_SYNTHESIZE:
            return (valid_synthesize_envelope(experiment_id=fixture.experiment_id), "codex-exec")
        return (
            valid_envelope(experiment_id=fixture.experiment_id, believed_best_config=fixture.believed_best_config),
            "codex-exec",
        )

    monkeypatch.setattr(llm, "call_research_llm", fake)


def _ledger_memory_root(fixture: CloseoutFixture):
    return write_ledger_memory_root(fixture.store_root / "notes" / "__RESEARCH_MEMORY__")


def _run(fixture: CloseoutFixture, memory_root, **kwargs):
    return runner.run_closeout(
        store_root=fixture.store_root,
        experiment_id=fixture.experiment_id,
        memory_root=str(memory_root),
        **kwargs,
    )


def _branch_dir(fixture: CloseoutFixture, memory_root):
    return memory_root / "experiments" / fixture.experiment_id


def _snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


# --------------------------------------------------------------------------- #
# EXTRACT
# --------------------------------------------------------------------------- #
def test_extract_writes_exactly_seven_file_branch(
    closeout_fixture: CloseoutFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_transport(monkeypatch, closeout_fixture)
    memory_root = _ledger_memory_root(closeout_fixture)
    result = _run(closeout_fixture, memory_root, until="extract")

    assert result.error is None
    branch = _branch_dir(closeout_fixture, memory_root)
    files = sorted(p.name for p in branch.iterdir() if p.is_file())
    assert files == sorted(["README.md", *(f"{t}.md" for t in TOPIC_NAMES)])
    assert closeout_fixture.experiment_id in (branch / "README.md").read_text(encoding="utf-8")


def test_extract_onto_existing_branch_is_refused(
    closeout_fixture: CloseoutFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_transport(monkeypatch, closeout_fixture)
    memory_root = _ledger_memory_root(closeout_fixture)
    branch = _branch_dir(closeout_fixture, memory_root)
    branch.mkdir(parents=True, exist_ok=True)
    (branch / "README.md").write_text("PRE-EXISTING", encoding="utf-8")

    result = _run(closeout_fixture, memory_root, until="extract")
    assert result.stopped_at_phase == "extract"
    assert result.error == ct.err_branch_exists(str(branch))
    # The pre-existing branch content is untouched by the refusal.
    assert (branch / "README.md").read_text(encoding="utf-8") == "PRE-EXISTING"


def test_restart_extract_backs_up_then_rewrites(
    closeout_fixture: CloseoutFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_transport(monkeypatch, closeout_fixture)
    memory_root = _ledger_memory_root(closeout_fixture)
    _run(closeout_fixture, memory_root, until="extract")
    branch = _branch_dir(closeout_fixture, memory_root)
    # A stray file plus mutated README simulate an older branch that restart must clear.
    (branch / "stray.md").write_text("stray", encoding="utf-8")
    (branch / "README.md").write_text("OLD README", encoding="utf-8")

    result = _run(closeout_fixture, memory_root, until="extract", restart_from="extract")
    assert result.error is None
    files = sorted(p.name for p in branch.iterdir() if p.is_file())
    assert files == sorted(["README.md", *(f"{t}.md" for t in TOPIC_NAMES)])  # exactly seven, stray gone
    assert "OLD README" not in (branch / "README.md").read_text(encoding="utf-8")

    backups = closeout_fixture.closeout_dir() / "backups"
    backed_up = list(backups.rglob("stray.md"))
    assert backed_up and backed_up[0].read_text(encoding="utf-8") == "stray"


# --------------------------------------------------------------------------- #
# SYNTHESIZE
# --------------------------------------------------------------------------- #
def test_synthesize_only_touches_ledgers_current_md_and_branch(
    closeout_fixture: CloseoutFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_transport(monkeypatch, closeout_fixture)
    memory_root = _ledger_memory_root(closeout_fixture)
    _run(closeout_fixture, memory_root, until="extract")  # create the branch first
    before = _snapshot(memory_root)

    result = _run(closeout_fixture, memory_root, until="synthesize")
    assert result.error is None
    after = _snapshot(memory_root)

    changed = {path for path in after if before.get(path) != after.get(path)}
    for path in changed:
        allowed = (
            path == ct.CURRENT_MD_FILENAME
            or path.startswith("topics/")
            or path.startswith(f"experiments/{closeout_fixture.experiment_id}/")
        )
        assert allowed, f"unexpected change outside the allowed set: {path}"
    # Each ledger gained exactly one entry for this experiment.
    for topic in TOPIC_NAMES:
        text = (memory_root / "topics" / f"{topic}.md").read_text(encoding="utf-8")
        assert merge.count_entries(text, closeout_fixture.experiment_id) == 1


def test_synthesize_backs_up_all_master_memory_before_writing(
    closeout_fixture: CloseoutFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_transport(monkeypatch, closeout_fixture)
    memory_root = _ledger_memory_root(closeout_fixture)
    _run(closeout_fixture, memory_root, until="extract")
    relative_paths = [ct.CURRENT_MD_FILENAME, *(f"topics/{topic}.md" for topic in TOPIC_NAMES)]
    before = {path: (memory_root / path).read_bytes() for path in relative_paths}

    result = _run(closeout_fixture, memory_root, until="synthesize")

    assert result.error is None
    backup_dirs = list((closeout_fixture.closeout_dir() / "backups").iterdir())
    assert len(backup_dirs) == 1
    backed_up = {
        str(path.relative_to(backup_dirs[0])): path.read_bytes() for path in backup_dirs[0].rglob("*") if path.is_file()
    }
    assert backed_up == before


def test_invalid_synthesize_response_creates_no_backup(
    closeout_fixture: CloseoutFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_transport(monkeypatch, closeout_fixture)
    original_call = llm.call_research_llm

    def invalid_current(**kwargs):
        if kwargs.get("round_label") == ct.PHASE_SYNTHESIZE:
            payload = json.loads(valid_synthesize_envelope(experiment_id=closeout_fixture.experiment_id))
            payload["current_md"] = "# CURRENT\n"
            return (json.dumps(payload), "codex-exec")
        return original_call(**kwargs)

    monkeypatch.setattr(llm, "call_research_llm", invalid_current)
    memory_root = _ledger_memory_root(closeout_fixture)

    result = _run(closeout_fixture, memory_root, until="synthesize")

    assert result.stopped_at_phase == ct.PHASE_SYNTHESIZE
    assert not (closeout_fixture.closeout_dir() / "backups").exists()


def test_synthesize_backup_failure_is_captured_without_master_memory_writes(
    closeout_fixture: CloseoutFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_transport(monkeypatch, closeout_fixture)
    memory_root = _ledger_memory_root(closeout_fixture)
    _run(closeout_fixture, memory_root, until="extract")
    before = _snapshot(memory_root)

    def fail_copy(*_args, **_kwargs):
        raise OSError("simulated backup failure")

    monkeypatch.setattr(runner.shutil, "copy2", fail_copy)

    result = _run(closeout_fixture, memory_root, until="synthesize")

    assert result.stopped_at_phase == ct.PHASE_SYNTHESIZE
    assert result.error == ct.ERR_SYNTHESIZE_BACKUP_FAILED
    assert _snapshot(memory_root) == before
    backup_root = closeout_fixture.closeout_dir() / "backups"
    assert backup_root.is_dir()
    assert not list(backup_root.iterdir())


def test_restart_synthesize_replaces_and_keeps_one_entry(
    closeout_fixture: CloseoutFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_transport(monkeypatch, closeout_fixture)
    memory_root = _ledger_memory_root(closeout_fixture)
    _run(closeout_fixture, memory_root, until="synthesize")
    backup_root = closeout_fixture.closeout_dir() / "backups"
    first_backup = next(backup_root.iterdir())
    before_restart = {
        ct.CURRENT_MD_FILENAME: (memory_root / ct.CURRENT_MD_FILENAME).read_bytes(),
        **{f"topics/{topic}.md": (memory_root / "topics" / f"{topic}.md").read_bytes() for topic in TOPIC_NAMES},
    }

    result = _run(closeout_fixture, memory_root, until="synthesize", restart_from="synthesize")
    assert result.error is None
    restart_backups = set(backup_root.iterdir()) - {first_backup}
    assert len(restart_backups) == 1
    restart_backup = restart_backups.pop()
    assert {
        str(path.relative_to(restart_backup)): path.read_bytes() for path in restart_backup.rglob("*") if path.is_file()
    } == before_restart
    for topic in TOPIC_NAMES:
        text = (memory_root / "topics" / f"{topic}.md").read_text(encoding="utf-8")
        assert merge.count_entries(text, closeout_fixture.experiment_id) == 1


@pytest.mark.parametrize("restart_from", ["finalize", "classify", "extract"])
def test_restart_upstream_of_synthesize_is_refused(
    closeout_fixture: CloseoutFixture,
    monkeypatch: pytest.MonkeyPatch,
    restart_from: str,
) -> None:
    _install_transport(monkeypatch, closeout_fixture)
    memory_root = _ledger_memory_root(closeout_fixture)
    _run(closeout_fixture, memory_root, until="synthesize")

    with pytest.raises(ct.CloseoutError) as exc:
        _run(closeout_fixture, memory_root, until="synthesize", restart_from=restart_from)
    assert str(exc.value) == ct.ERR_RESTART_BLOCKED_AFTER_SYNTHESIZE


# --------------------------------------------------------------------------- #
# Memory-root lock (§2.6)
# --------------------------------------------------------------------------- #
def test_memory_root_lock_serializes_concurrent_closeouts(
    closeout_fixture: CloseoutFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_transport(monkeypatch, closeout_fixture)
    memory_root = _ledger_memory_root(closeout_fixture)
    # A concurrent closeout holds the memory-root lock (live pid, fresh timestamp).
    lock = memory_root / ct.MEMORY_ROOT_LOCK_FILENAME
    lock.write_text(
        json.dumps({"pid": os.getpid(), "hostname": socket.gethostname(), "acquired_at": "2999-01-01T00:00:00+00:00"}),
        encoding="utf-8",
    )
    with pytest.raises(ct.CloseoutError) as exc:
        _run(closeout_fixture, memory_root, until="extract")
    assert str(exc.value) == ct.err_lock_held(str(lock))


def test_finalize_only_does_not_take_memory_lock(
    closeout_fixture: CloseoutFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_transport(monkeypatch, closeout_fixture)
    memory_root = _ledger_memory_root(closeout_fixture)
    # Even with the memory-root lock held, a finalize-only run proceeds (it never touches memory).
    lock = memory_root / ct.MEMORY_ROOT_LOCK_FILENAME
    lock.write_text(
        json.dumps({"pid": os.getpid(), "hostname": socket.gethostname(), "acquired_at": "2999-01-01T00:00:00+00:00"}),
        encoding="utf-8",
    )
    result = _run(closeout_fixture, memory_root, until="finalize")
    assert result.error is None
