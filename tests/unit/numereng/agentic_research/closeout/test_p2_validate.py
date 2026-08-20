"""P2 content-validator tests (§3.2/§3.3): extract slots, synthesize envelope, CURRENT.md, entry blocks."""

from __future__ import annotations

import json

import pytest

from numereng.agentic_research.engine.closeout import phases
from numereng.agentic_research.engine.closeout import types as ct

from .conftest import (
    TOPIC_NAMES,
    extract_topic_file,
    synthesize_current_md,
    synthesize_entry,
    valid_extract_envelope,
    valid_synthesize_envelope,
)

EXPERIMENT_ID = "2026-07-13_closeout-exp"


# --------------------------------------------------------------------------- #
# EXTRACT validators
# --------------------------------------------------------------------------- #
def test_valid_extract_envelope_passes() -> None:
    files, _notes = phases.parse_files_envelope(valid_extract_envelope(experiment_id=EXPERIMENT_ID))
    slots = phases.validate_extract(files, experiment_id=EXPERIMENT_ID)
    assert set(slots) == {"README.md", *(f"{t}.md" for t in TOPIC_NAMES)}


def test_extract_rejects_invalid_evidence_level() -> None:
    bad = extract_topic_file("features").replace("computed metric", "definitely true")
    files = [{"path": "README.md", "content": f"{EXPERIMENT_ID}\n" + "\n".join(f"{t}.md" for t in TOPIC_NAMES)}]
    files += [{"path": f"{t}.md", "content": bad if t == "features" else extract_topic_file(t)} for t in TOPIC_NAMES]
    with pytest.raises(ct.CloseoutError) as exc:
        phases.validate_extract(files, experiment_id=EXPERIMENT_ID)
    assert str(exc.value) == ct.err_evidence_level_invalid("features")


def test_extract_rejects_readme_missing_topic_link() -> None:
    files = [{"path": "README.md", "content": f"Experiment {EXPERIMENT_ID}. features.md only.\n"}]
    files += [{"path": f"{t}.md", "content": extract_topic_file(t)} for t in TOPIC_NAMES]
    with pytest.raises(ct.CloseoutError) as exc:
        phases.validate_extract(files, experiment_id=EXPERIMENT_ID)
    assert str(exc.value) == ct.err_readme_link_missing("ensembling.md")


# --------------------------------------------------------------------------- #
# SYNTHESIZE envelope parsing
# --------------------------------------------------------------------------- #
def test_five_deltas_is_output_slot_missing() -> None:
    payload = json.loads(valid_synthesize_envelope(experiment_id=EXPERIMENT_ID))
    payload["deltas"] = payload["deltas"][:5]  # drop the sixth topic
    with pytest.raises(ct.CloseoutError) as exc:
        phases.parse_synthesize_envelope(json.dumps(payload))
    assert str(exc.value) == ct.err_output_slot_missing("targets")


def test_synthesize_rejects_empty_current_md() -> None:
    payload = json.loads(valid_synthesize_envelope(experiment_id=EXPERIMENT_ID))
    payload["current_md"] = "   "
    with pytest.raises(ct.CloseoutError) as exc:
        phases.parse_synthesize_envelope(json.dumps(payload))
    assert str(exc.value) == ct.err_output_content_empty(ct.CURRENT_MD_FILENAME)


# --------------------------------------------------------------------------- #
# entry-block + CURRENT.md validators
# --------------------------------------------------------------------------- #
def test_entry_block_requires_sole_id_heading() -> None:
    bad = synthesize_entry("features", EXPERIMENT_ID) + "\n## Extra heading\n"
    with pytest.raises(ct.CloseoutError) as exc:
        phases.validate_entry_block("features", bad, experiment_id=EXPERIMENT_ID)
    assert str(exc.value) == ct.err_entry_heading_invalid("features")


def test_entry_block_requires_branch_link() -> None:
    bad = f"### {EXPERIMENT_ID}\n- Learning: no link.\n"
    with pytest.raises(ct.CloseoutError) as exc:
        phases.validate_entry_block("targets", bad, experiment_id=EXPERIMENT_ID)
    assert str(exc.value) == ct.err_entry_link_missing("targets")


def test_current_md_missing_section_rejected() -> None:
    bad = synthesize_current_md(EXPERIMENT_ID).replace("## Current Constraints", "## Something Else")
    with pytest.raises(ct.CloseoutError) as exc:
        phases.validate_current_md(bad, experiment_id=EXPERIMENT_ID)
    assert str(exc.value) == ct.err_current_md_section_missing("Current Constraints")


def test_current_md_missing_full_record_pointer_rejected() -> None:
    bad = synthesize_current_md(EXPERIMENT_ID).replace(
        f"Full record: experiments/{EXPERIMENT_ID}/README.md", "See the branch."
    )
    with pytest.raises(ct.CloseoutError) as exc:
        phases.validate_current_md(bad, experiment_id=EXPERIMENT_ID)
    assert str(exc.value) == ct.err_current_md_reference_missing("full_record")


def test_current_md_too_short_rejected() -> None:
    short = (
        f"# CURRENT\n## Compressed Frontier\n{EXPERIMENT_ID}\n"
        f"Full record: experiments/{EXPERIMENT_ID}/README.md\n"
        "## Comparison Anchors\nx\n## Current Constraints\ny\n"
    )
    with pytest.raises(ct.CloseoutError) as exc:
        phases.validate_current_md(short, experiment_id=EXPERIMENT_ID)
    assert str(exc.value).startswith(ct.ERROR_PREFIX + "current_md_too_short")
