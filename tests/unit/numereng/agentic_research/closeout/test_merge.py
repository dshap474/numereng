"""Deterministic ledger-merge unit tests (§3.3): byte preservation, duplicates, per-section splices."""

from __future__ import annotations

import pytest

from numereng.agentic_research.engine.closeout import merge
from numereng.agentic_research.engine.closeout import types as ct

from .conftest import ledger_text, synthesize_entry

EXPERIMENT_ID = "2026-07-13_closeout-exp"


def test_append_preserves_all_prior_bytes() -> None:
    existing = ledger_text("hyperparameters", prior_entry=True)
    entry = synthesize_entry("hyperparameters", EXPERIMENT_ID)
    merged = merge.merge_ledger(
        existing,
        new_entry=entry,
        overview_replacement=None,
        best_understanding_replacement=None,
        experiment_id=EXPERIMENT_ID,
        replace_existing=False,
        topic="hyperparameters",
    )
    # The entire prior ledger is a contiguous prefix of the merged text — no prior byte moved.
    assert merged.startswith(existing.rstrip("\n"))
    assert existing.rstrip("\n") in merged
    assert merge.count_entries(merged, EXPERIMENT_ID) == 1
    assert merge.count_entries(merged, "2026-06-01_prior-exp") == 1


def test_duplicate_entry_refused_on_plain_run() -> None:
    existing = ledger_text("features", prior_entry=True)
    entry = synthesize_entry("features", EXPERIMENT_ID)
    once = merge.merge_ledger(
        existing,
        new_entry=entry,
        overview_replacement=None,
        best_understanding_replacement=None,
        experiment_id=EXPERIMENT_ID,
        replace_existing=False,
        topic="features",
    )
    with pytest.raises(ct.CloseoutError) as exc:
        merge.merge_ledger(
            once,
            new_entry=entry,
            overview_replacement=None,
            best_understanding_replacement=None,
            experiment_id=EXPERIMENT_ID,
            replace_existing=False,
            topic="features",
        )
    assert str(exc.value) == ct.err_duplicate_ledger_entry("features")


def test_restart_replaces_existing_block_in_place() -> None:
    existing = ledger_text("models", prior_entry=True)
    first = merge.merge_ledger(
        existing,
        new_entry=synthesize_entry("models", EXPERIMENT_ID),
        overview_replacement=None,
        best_understanding_replacement=None,
        experiment_id=EXPERIMENT_ID,
        replace_existing=False,
        topic="models",
    )
    replacement = (
        f"### {EXPERIMENT_ID}\n- Source: [branch](../experiments/{EXPERIMENT_ID}/models.md)\n- Learning: revised.\n"
    )
    second = merge.merge_ledger(
        first,
        new_entry=replacement,
        overview_replacement=None,
        best_understanding_replacement=None,
        experiment_id=EXPERIMENT_ID,
        replace_existing=True,
        topic="models",
    )
    assert merge.count_entries(second, EXPERIMENT_ID) == 1
    assert "Learning: revised." in second
    assert "led BMC200 within the seed-noise floor" not in second
    # The prior experiment's block is untouched by the in-place replacement.
    assert merge.count_entries(second, "2026-06-01_prior-exp") == 1


def test_overview_only_change_leaves_other_sections_byte_identical() -> None:
    existing = ledger_text("targets", prior_entry=True)
    _pre, _ov, best_before, learnings_before = merge.parse_ledger(existing, topic="targets")
    merged = merge.merge_ledger(
        existing,
        new_entry=None,
        overview_replacement="Brand new overview body.",
        best_understanding_replacement=None,
        experiment_id=EXPERIMENT_ID,
        replace_existing=False,
        topic="targets",
    )
    _pre2, overview_after, best_after, learnings_after = merge.parse_ledger(merged, topic="targets")
    assert "Brand new overview body." in overview_after
    assert "Prior overview for targets." not in overview_after
    # Best-understanding and learnings sections are preserved byte-for-byte.
    assert best_after == best_before
    assert learnings_after == learnings_before


def test_best_understanding_only_change_leaves_other_sections_byte_identical() -> None:
    existing = ledger_text("targets", prior_entry=True)
    _pre, overview_before, _bu, learnings_before = merge.parse_ledger(existing, topic="targets")
    merged = merge.merge_ledger(
        existing,
        new_entry=None,
        overview_replacement=None,
        best_understanding_replacement="Brand new best understanding body.",
        experiment_id=EXPERIMENT_ID,
        replace_existing=False,
        topic="targets",
    )
    _pre2, overview_after, best_after, learnings_after = merge.parse_ledger(merged, topic="targets")
    assert "Brand new best understanding body." in best_after
    assert "Prior best understanding for targets." not in best_after
    # Overview and learnings sections are preserved byte-for-byte.
    assert overview_after == overview_before
    assert learnings_after == learnings_before


def test_section_replacement_with_heading_is_rejected() -> None:
    existing = ledger_text("features", prior_entry=True)
    with pytest.raises(ct.CloseoutError) as exc:
        merge.merge_ledger(
            existing,
            new_entry=None,
            overview_replacement="## Sneaky Heading\n\nthis is a region, not a body",
            best_understanding_replacement=None,
            experiment_id=EXPERIMENT_ID,
            replace_existing=False,
            topic="features",
        )
    assert str(exc.value) == ct.err_section_replacement_invalid("features", "overview")


def test_best_understanding_replacement_with_heading_is_rejected() -> None:
    existing = ledger_text("features", prior_entry=True)
    with pytest.raises(ct.CloseoutError) as exc:
        merge.merge_ledger(
            existing,
            new_entry=None,
            overview_replacement=None,
            best_understanding_replacement="## Another Section\n\nbody",
            experiment_id=EXPERIMENT_ID,
            replace_existing=False,
            topic="features",
        )
    assert str(exc.value) == ct.err_section_replacement_invalid("features", "best_understanding")


def test_parse_ledger_splits_three_sections() -> None:
    existing = ledger_text("hyperparameters", prior_entry=True)
    preamble, overview, best_understanding, learnings = merge.parse_ledger(existing, topic="hyperparameters")
    assert preamble.startswith("# hyperparameters")
    assert overview.startswith(ct.LEDGER_OVERVIEW_ANCHOR)
    assert best_understanding.startswith(ct.LEDGER_BEST_UNDERSTANDING_ANCHOR)
    assert learnings.startswith(ct.LEDGER_LEARNINGS_ANCHOR)
    # Concatenation is lossless.
    assert preamble + overview + best_understanding + learnings == existing


def test_parse_ledger_rejects_missing_anchors() -> None:
    with pytest.raises(ct.CloseoutError) as exc:
        merge.parse_ledger("# hyperparameters\n\nno anchors here\n", topic="hyperparameters")
    assert str(exc.value) == ct.err_ledger_structure("hyperparameters")


def test_parse_ledger_rejects_missing_best_understanding() -> None:
    two_section = "# hyperparameters\n\n## Current Overview\n\nbody\n\n## Append-Only Experiment Learnings\n"
    with pytest.raises(ct.CloseoutError) as exc:
        merge.parse_ledger(two_section, topic="hyperparameters")
    assert str(exc.value) == ct.err_ledger_structure("hyperparameters")
