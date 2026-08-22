"""Selective closeout classification routing and schema-v1 compatibility tests.

USAGE:
    uv run pytest -q tests/unit/numereng/agentic_research/closeout/test_classify.py
"""

from __future__ import annotations

import json

import pytest

from numereng.agentic_research.engine import llm
from numereng.agentic_research.engine.closeout import phases, runner
from numereng.agentic_research.engine.closeout import types as ct

from .conftest import (
    CloseoutFixture,
    valid_classification,
    valid_envelope,
    valid_extract_envelope,
    valid_synthesize_envelope,
    write_ledger_memory_root,
)


def _install_transport(
    monkeypatch: pytest.MonkeyPatch,
    fixture: CloseoutFixture,
    *,
    disposition: str,
    relevant_topics: tuple[str, ...],
) -> None:
    def fake(**kwargs):
        label = kwargs.get("round_label")
        if label == ct.PHASE_CLASSIFY:
            payload = valid_classification(disposition=disposition, relevant_topics=relevant_topics)
        elif label == ct.PHASE_EXTRACT:
            payload = valid_extract_envelope(experiment_id=fixture.experiment_id)
        elif label == ct.PHASE_SYNTHESIZE:
            payload = valid_synthesize_envelope(experiment_id=fixture.experiment_id, topics=relevant_topics)
        else:
            payload = valid_envelope(
                experiment_id=fixture.experiment_id,
                believed_best_config=fixture.believed_best_config,
            )
        return payload, "codex-exec"

    monkeypatch.setattr(llm, "call_research_llm", fake)


@pytest.mark.parametrize(
    ("disposition", "relevant_topics"),
    [
        ("master", ("features", "models")),
        ("master", ()),
        ("branch_only", ()),
        ("exclude", ()),
    ],
)
def test_classification_routes_closeout(
    closeout_fixture: CloseoutFixture,
    monkeypatch: pytest.MonkeyPatch,
    disposition: str,
    relevant_topics: tuple[str, ...],
) -> None:
    memory_root = write_ledger_memory_root(closeout_fixture.store_root / "notes" / "__RESEARCH_MEMORY__")
    original_current = (memory_root / ct.CURRENT_MD_FILENAME).read_bytes()
    original_ledgers = {topic: (memory_root / "topics" / f"{topic}.md").read_bytes() for topic in ct.MEMORY_TOPIC_FILES}
    _install_transport(
        monkeypatch,
        closeout_fixture,
        disposition=disposition,
        relevant_topics=relevant_topics,
    )

    result = runner.run_closeout(
        store_root=closeout_fixture.store_root,
        experiment_id=closeout_fixture.experiment_id,
        memory_root=str(memory_root),
    )

    assert result.error is None
    statuses = {phase.name: phase.status for phase in result.phases}
    assert statuses[ct.PHASE_FINALIZE] == "done"
    assert statuses[ct.PHASE_CLASSIFY] == "done"
    branch_dir = memory_root / "experiments" / closeout_fixture.experiment_id
    if disposition == "master":
        assert statuses[ct.PHASE_EXTRACT] == "done"
        assert statuses[ct.PHASE_SYNTHESIZE] == "done"
        assert branch_dir.is_dir()
        assert (memory_root / ct.CURRENT_MD_FILENAME).read_bytes() != original_current
        for topic in ct.MEMORY_TOPIC_FILES:
            changed = (memory_root / "topics" / f"{topic}.md").read_bytes() != original_ledgers[topic]
            assert changed is (topic in relevant_topics)
    elif disposition == "branch_only":
        assert statuses[ct.PHASE_EXTRACT] == "done"
        assert statuses[ct.PHASE_SYNTHESIZE] == "skipped"
        assert branch_dir.is_dir()
        assert (memory_root / ct.CURRENT_MD_FILENAME).read_bytes() == original_current
        assert all(
            (memory_root / "topics" / f"{topic}.md").read_bytes() == original_ledgers[topic]
            for topic in ct.MEMORY_TOPIC_FILES
        )
    else:
        assert statuses[ct.PHASE_EXTRACT] == "skipped"
        assert statuses[ct.PHASE_SYNTHESIZE] == "skipped"
        assert not branch_dir.exists()
        assert (closeout_fixture.closeout_dir() / ct.CLOSEOUT_MEMO_FILENAME).is_file()
        assert (closeout_fixture.closeout_dir() / ct.CLOSEOUT_CLASSIFICATION_FILENAME).is_file()
        assert (memory_root / ct.CURRENT_MD_FILENAME).read_bytes() == original_current


@pytest.mark.parametrize(
    "payload",
    [
        {"disposition": "invalid", "relevant_topics": [], "rationale": "x"},
        {"disposition": "master", "relevant_topics": ["unknown"], "rationale": "x"},
        {"disposition": "master", "relevant_topics": ["models", "models"], "rationale": "x"},
        {"disposition": "master", "relevant_topics": [], "rationale": ""},
    ],
)
def test_invalid_classification_rejected(payload: dict[str, object]) -> None:
    with pytest.raises(ct.CloseoutError):
        phases.parse_classification(json.dumps(payload))


@pytest.mark.parametrize("disposition", ["branch_only", "exclude"])
def test_non_master_classification_rejects_relevant_topic(disposition: str) -> None:
    raw = valid_classification(disposition=disposition, relevant_topics=("models",))

    with pytest.raises(ct.CloseoutError) as exc:
        phases.parse_classification(raw)

    assert str(exc.value) == ct.err_classification_field_invalid("relevant_topics")


def test_legacy_completed_state_loads_as_master_without_memory_writes(
    closeout_fixture: CloseoutFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory_root = write_ledger_memory_root(closeout_fixture.store_root / "notes" / "__RESEARCH_MEMORY__")
    memory_before = {
        path.relative_to(memory_root): path.read_bytes() for path in memory_root.rglob("*") if path.is_file()
    }
    closeout_dir = closeout_fixture.closeout_dir()
    closeout_dir.mkdir(parents=True, exist_ok=True)
    legacy_phases = {phase: {"status": "done"} for phase in (ct.PHASE_FINALIZE, ct.PHASE_EXTRACT, ct.PHASE_SYNTHESIZE)}
    (closeout_dir / ct.CLOSEOUT_STATE_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_id": closeout_fixture.experiment_id,
                "memory_root_identity": str(memory_root),
                "phases": legacy_phases,
            }
        ),
        encoding="utf-8",
    )

    def fail_transport(**kwargs):
        raise AssertionError("legacy completed closeout must not call the LLM")

    monkeypatch.setattr(llm, "call_research_llm", fail_transport)
    result = runner.run_closeout(
        store_root=closeout_fixture.store_root,
        experiment_id=closeout_fixture.experiment_id,
        memory_root=str(memory_root),
    )

    statuses = {phase.name: phase.status for phase in result.phases}
    assert statuses[ct.PHASE_CLASSIFY] == "done"
    persisted = json.loads((closeout_dir / ct.CLOSEOUT_STATE_FILENAME).read_text(encoding="utf-8"))
    assert persisted["schema_version"] == 2
    assert persisted["phases"][ct.PHASE_CLASSIFY]["notes"] == ct.LEGACY_MASTER_NOTE
    memory_after = {
        path.relative_to(memory_root): path.read_bytes() for path in memory_root.rglob("*") if path.is_file()
    }
    assert memory_after == memory_before


def test_synthesize_rejects_unselected_topic_delta() -> None:
    raw = valid_synthesize_envelope(
        experiment_id="2026-07-14_subset",
        topics=("features", "models"),
    )

    with pytest.raises(ct.CloseoutError):
        phases.parse_synthesize_envelope(raw, relevant_topics=("features",))


@pytest.mark.parametrize("restart_from", [ct.PHASE_FINALIZE, ct.PHASE_CLASSIFY])
def test_restart_upstream_of_completed_extract_is_refused_when_synthesis_was_skipped(
    closeout_fixture: CloseoutFixture,
    monkeypatch: pytest.MonkeyPatch,
    restart_from: str,
) -> None:
    memory_root = write_ledger_memory_root(closeout_fixture.store_root / "notes" / "__RESEARCH_MEMORY__")
    _install_transport(
        monkeypatch,
        closeout_fixture,
        disposition="branch_only",
        relevant_topics=(),
    )
    runner.run_closeout(
        store_root=closeout_fixture.store_root,
        experiment_id=closeout_fixture.experiment_id,
        memory_root=str(memory_root),
    )

    with pytest.raises(ct.CloseoutError) as exc:
        runner.run_closeout(
            store_root=closeout_fixture.store_root,
            experiment_id=closeout_fixture.experiment_id,
            memory_root=str(memory_root),
            restart_from=restart_from,
        )

    assert str(exc.value) == ct.ERR_RESTART_BLOCKED_AFTER_EXTRACT


def test_restart_from_extract_replaces_existing_branch_when_synthesis_was_skipped(
    closeout_fixture: CloseoutFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory_root = write_ledger_memory_root(closeout_fixture.store_root / "notes" / "__RESEARCH_MEMORY__")
    _install_transport(
        monkeypatch,
        closeout_fixture,
        disposition="branch_only",
        relevant_topics=(),
    )
    runner.run_closeout(
        store_root=closeout_fixture.store_root,
        experiment_id=closeout_fixture.experiment_id,
        memory_root=str(memory_root),
    )

    result = runner.run_closeout(
        store_root=closeout_fixture.store_root,
        experiment_id=closeout_fixture.experiment_id,
        memory_root=str(memory_root),
        restart_from=ct.PHASE_EXTRACT,
    )

    assert result.error is None
    statuses = {phase.name: phase.status for phase in result.phases}
    assert statuses[ct.PHASE_EXTRACT] == "done"
    assert statuses[ct.PHASE_SYNTHESIZE] == "skipped"
    backups = closeout_fixture.closeout_dir() / "backups"
    assert any((path / ct.MEMORY_BRANCH_README).is_file() for path in backups.iterdir())
