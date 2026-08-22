"""Context boundedness: the 890 KB prompt that killed a 500-round run must not be reproducible.

A 200-round experiment with fat memos still produces a context capped at MAX_CLOSEOUT_CONTEXT_CHARS,
with the deterministic evidence retained and only round memos dropped (with an explicit marker).
"""

from __future__ import annotations

import json

from numereng.agentic_research.engine.closeout import context as ctx_mod
from numereng.agentic_research.engine.closeout import types as ct

from .conftest import CloseoutFixture


def test_finalize_context_is_bounded_with_200_rounds(closeout_fixture: CloseoutFixture) -> None:
    rounds_dir = closeout_fixture.agentic_dir() / "rounds"
    rounds_dir.mkdir(parents=True, exist_ok=True)
    for index in range(1, 201):
        (rounds_dir / f"r{index:03d}.md").write_text(f"# r{index:03d}\n\n" + ("memo body " * 700), encoding="utf-8")

    experiment = closeout_fixture.experiment()
    state = json.loads(closeout_fixture.state_path().read_text(encoding="utf-8"))
    evidence = {"experiment_id": closeout_fixture.experiment_id, "believed_best": {"config": "config_001.json"}}

    ctx = ctx_mod.build_finalize_context(experiment=experiment, state=state, evidence=evidence)

    total = len(json.dumps(ctx, default=str))
    assert total <= ct.MAX_CLOSEOUT_CONTEXT_CHARS
    # Evidence is placed first and never dropped.
    assert ctx["evidence_summary"] == evidence
    # Some memos were dropped under pressure, and the drop is explicit.
    assert "round_memos_truncation" in ctx
    assert len(ctx["round_memos"]) < 200
    # Kept memos are newest-first.
    kept_labels = [item["round_label"] for item in ctx["round_memos"]]
    assert kept_labels == sorted(kept_labels, reverse=True)


def test_extract_context_drops_rounds_table_under_pressure(closeout_fixture: CloseoutFixture) -> None:
    experiment = closeout_fixture.experiment()
    big_rows = [{"round": i, "note": "x" * 500} for i in range(2000)]
    evidence = {"experiment_id": closeout_fixture.experiment_id, "rounds_table": big_rows}

    ctx = ctx_mod.build_extract_context(experiment=experiment, memo_text="MEMO " * 10_000, evidence=evidence)

    assert len(json.dumps(ctx, default=str)) <= ct.MAX_CLOSEOUT_CONTEXT_CHARS
    assert "rounds_table_truncation" in ctx
    assert len(ctx["rounds_table"]) < len(big_rows)
    # The evidence head keeps everything except the bulk table.
    assert ctx["evidence_summary"]["experiment_id"] == closeout_fixture.experiment_id


def test_synthesize_context_caps_each_ledger_with_marker(closeout_fixture: CloseoutFixture) -> None:
    from .conftest import ledger_text, write_ledger_memory_root

    memory_root = write_ledger_memory_root(closeout_fixture.store_root / "notes" / "__RESEARCH_MEMORY__")
    # Blow one ledger far past the per-ledger cap; the view must be truncated with a marker.
    fat = ledger_text("features").replace(
        "Prior overview for features.", "Prior overview for features. " + ("noise " * 8000)
    )
    (memory_root / "topics" / "features.md").write_text(fat, encoding="utf-8")

    ctx = ctx_mod.build_synthesize_context(experiment_id=closeout_fixture.experiment_id, memory_root=memory_root)

    assert len(json.dumps(ctx, default=str)) <= ct.MAX_CLOSEOUT_CONTEXT_CHARS
    assert len(ctx["ledgers"]["features"]) <= ct.LEDGER_CONTEXT_CAP + len("\n...[truncated]")
    assert ctx["ledgers"]["features"].endswith("...[truncated]")
