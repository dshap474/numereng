"""Commit-journal + crash roll-forward: a process death at any write boundary rolls forward clean.

The commit protocol is stage -> commit.json -> apply slots -> mark done -> delete journal+stage. A
kill after any step must leave a consistent state on the next invocation (via _roll_forward), and
re-application must be idempotent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from numereng.agentic_research.engine import types as ar_types
from numereng.agentic_research.engine.closeout import runner
from numereng.agentic_research.engine.closeout import types as ct

PHASE = ct.PHASE_FINALIZE


def _state(tmp_path: Path) -> tuple[ct.CloseoutState, Path]:
    state = ct.CloseoutState.new(experiment_id="exp", memory_root_identity=str(tmp_path / "mem"))
    return state, tmp_path / "state.json"


def _stage_commit(closeout_dir: Path, slots: dict[str, str], *, applied: set[str]) -> dict[str, object]:
    """Write staged files + commit.json exactly as the runner would, then pre-apply `applied` dests."""
    stage_dir = closeout_dir / "stage" / PHASE
    stage_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for index, (dest, content) in enumerate(sorted(slots.items())):
        stage_file = stage_dir / f"slot_{index}"
        ar_types.write_text(stage_file, content)
        records.append(
            {
                "path": dest,
                "stage": str(stage_file.relative_to(closeout_dir)),
                "old_sha256": ct.sha256_file(Path(dest)),
                "new_sha256": ct.sha256_text(content),
            }
        )
    commit = {
        "phase": PHASE,
        "completed_at": "2026-07-13T00:00:00+00:00",
        "duration_seconds": 1.0,
        "notes": "crash-resume",
        "slots": records,
    }
    (closeout_dir / ct.CLOSEOUT_COMMIT_FILENAME).write_text(json.dumps(commit), encoding="utf-8")
    for dest in applied:
        ar_types.write_text(Path(dest), slots[dest])
    return commit


def test_stage_only_no_commit_leaves_phase_pending(tmp_path: Path) -> None:
    closeout_dir = tmp_path / "closeout"
    closeout_dir.mkdir()
    state, state_path = _state(tmp_path)
    # Stage a file but never write commit.json (death before the journal record).
    (closeout_dir / "stage" / PHASE).mkdir(parents=True)
    (closeout_dir / "stage" / PHASE / "slot_0").write_text("staged", encoding="utf-8")
    dest = closeout_dir / ct.CLOSEOUT_MEMO_FILENAME

    runner._roll_forward(closeout_dir, state, state_path)

    assert state.phases[PHASE].status == "pending"
    assert not dest.exists()


def test_commit_written_no_slots_applied_rolls_forward(tmp_path: Path) -> None:
    closeout_dir = tmp_path / "closeout"
    closeout_dir.mkdir()
    state, state_path = _state(tmp_path)
    dest = str(closeout_dir / ct.CLOSEOUT_MEMO_FILENAME)
    _stage_commit(closeout_dir, {dest: "final memo"}, applied=set())

    runner._roll_forward(closeout_dir, state, state_path)

    assert state.phases[PHASE].status == "done"
    assert Path(dest).read_text(encoding="utf-8") == "final memo"
    assert not (closeout_dir / ct.CLOSEOUT_COMMIT_FILENAME).exists()
    assert not (closeout_dir / "stage" / PHASE).exists()


def test_partial_apply_rolls_forward_idempotently(tmp_path: Path) -> None:
    closeout_dir = tmp_path / "closeout"
    closeout_dir.mkdir()
    state, state_path = _state(tmp_path)
    dest_a = str(closeout_dir / "a.md")
    dest_b = str(closeout_dir / "b.md")
    # Death after the first slot was applied but before the second.
    _stage_commit(closeout_dir, {dest_a: "aaa", dest_b: "bbb"}, applied={dest_a})

    runner._roll_forward(closeout_dir, state, state_path)

    assert Path(dest_a).read_text(encoding="utf-8") == "aaa"
    assert Path(dest_b).read_text(encoding="utf-8") == "bbb"
    assert state.phases[PHASE].status == "done"
    assert state.phases[PHASE].outputs == {dest_a: ct.sha256_text("aaa"), dest_b: ct.sha256_text("bbb")}

    # Re-running is a no-op (commit.json already deleted).
    runner._roll_forward(closeout_dir, state, state_path)
    assert Path(dest_a).read_text(encoding="utf-8") == "aaa"


def test_conflicting_destination_raises(tmp_path: Path) -> None:
    closeout_dir = tmp_path / "closeout"
    closeout_dir.mkdir()
    state, state_path = _state(tmp_path)
    dest = str(closeout_dir / ct.CLOSEOUT_MEMO_FILENAME)
    _stage_commit(closeout_dir, {dest: "final memo"}, applied=set())
    # A third party rewrote the destination to content matching neither old nor new.
    ar_types.write_text(Path(dest), "foreign content")

    with pytest.raises(ct.CloseoutError) as exc:
        runner._roll_forward(closeout_dir, state, state_path)
    assert str(exc.value) == ct.err_commit_conflict(dest)


def test_commit_phase_produces_consistent_state(tmp_path: Path) -> None:
    closeout_dir = tmp_path / "closeout"
    closeout_dir.mkdir()
    state, state_path = _state(tmp_path)
    dest = str(closeout_dir / ct.CLOSEOUT_MEMO_FILENAME)

    runner._commit_phase(
        closeout_dir,
        phase=PHASE,
        slots_content={dest: "committed memo"},
        notes="ok",
        duration_seconds=2.5,
        state=state,
        state_path=state_path,
    )

    assert state.phases[PHASE].status == "done"
    assert state.phases[PHASE].notes == "ok"
    assert Path(dest).read_text(encoding="utf-8") == "committed memo"
    assert not (closeout_dir / ct.CLOSEOUT_COMMIT_FILENAME).exists()
    assert not (closeout_dir / "stage" / PHASE).exists()
    # State on disk matches the in-memory object.
    on_disk = ct.CloseoutState.from_dict(json.loads(state_path.read_text(encoding="utf-8")))
    assert on_disk.phases[PHASE].status == "done"
