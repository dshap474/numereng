"""Context boundedness: the 890 KB prompt that killed a 500-round run must not be reproducible.

A 200-round experiment with fat memos still produces a context capped at MAX_CLOSEOUT_CONTEXT_CHARS,
with the deterministic evidence retained and only round memos dropped (with an explicit marker).
"""

from __future__ import annotations

import json

from numereng.agentic_research.engine.closeout import runner
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

    ctx = runner.build_finalize_context(experiment=experiment, state=state, evidence=evidence)

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


def test_finalize_context_truncates_a_huge_round_memo(closeout_fixture: CloseoutFixture) -> None:
    rounds_dir = closeout_fixture.agentic_dir() / "rounds"
    (rounds_dir / "r003.md").write_text("# r003\n\n" + ("x" * 200_000), encoding="utf-8")

    ctx = runner.build_finalize_context(
        experiment=closeout_fixture.experiment(),
        state=json.loads(closeout_fixture.state_path().read_text(encoding="utf-8")),
        evidence={"experiment_id": closeout_fixture.experiment_id},
    )

    newest = ctx["round_memos"][0]
    assert newest["round_label"] == "r003"
    assert newest["memo"].endswith("...[truncated]")
