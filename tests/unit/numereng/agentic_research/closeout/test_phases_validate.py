"""FINALIZE envelope-parse + memo-validator tests: no partial writes past a bad memo."""

from __future__ import annotations

import json

import pytest

from numereng.agentic_research.engine.closeout import phases
from numereng.agentic_research.engine.closeout import types as ct

from .conftest import build_finalize_memo

EXPERIMENT_ID = "2026-07-13_closeout-exp"
BELIEVED_BEST = "config_001.json"


def _valid_files() -> list[dict[str, str]]:
    memo = build_finalize_memo(experiment_id=EXPERIMENT_ID, believed_best_config=BELIEVED_BEST)
    return [{"path": ct.CLOSEOUT_MEMO_FILENAME, "content": memo}]


def _validate(files: list[dict[str, str]]) -> dict[str, str]:
    return phases.validate_finalize(files, experiment_id=EXPERIMENT_ID, believed_best_config=BELIEVED_BEST)


def test_valid_memo_passes() -> None:
    slots = _validate(_valid_files())
    assert set(slots) == {ct.CLOSEOUT_MEMO_FILENAME}


def test_parse_envelope_round_trips() -> None:
    raw = json.dumps({"files": _valid_files(), "notes": "ok"})
    files, notes = phases.parse_files_envelope(raw)
    assert files[0]["path"] == ct.CLOSEOUT_MEMO_FILENAME
    assert notes == "ok"


def test_parse_envelope_rejects_missing_files() -> None:
    with pytest.raises(ct.CloseoutError):
        phases.parse_files_envelope(json.dumps({"notes": "x"}))


def test_missing_section_rejected() -> None:
    memo = build_finalize_memo(experiment_id=EXPERIMENT_ID, believed_best_config=BELIEVED_BEST)
    memo = memo.replace("## Metric Conflicts", "## Something Else")
    with pytest.raises(ct.CloseoutError) as exc:
        _validate([{"path": ct.CLOSEOUT_MEMO_FILENAME, "content": memo}])
    assert str(exc.value) == ct.err_memo_section_missing("Metric Conflicts")


def test_missing_experiment_id_reference_rejected() -> None:
    memo = build_finalize_memo(experiment_id="OTHER", believed_best_config=BELIEVED_BEST)
    with pytest.raises(ct.CloseoutError) as exc:
        _validate([{"path": ct.CLOSEOUT_MEMO_FILENAME, "content": memo}])
    assert str(exc.value) == ct.err_memo_reference_missing("experiment_id")


def test_missing_believed_best_reference_rejected() -> None:
    memo = build_finalize_memo(experiment_id=EXPERIMENT_ID, believed_best_config="config_777.json")
    with pytest.raises(ct.CloseoutError) as exc:
        _validate([{"path": ct.CLOSEOUT_MEMO_FILENAME, "content": memo}])
    assert str(exc.value) == ct.err_memo_reference_missing("believed_best")


def test_too_short_memo_rejected() -> None:
    headings = "\n\n".join(f"## {h}" for h in phases.FINALIZE_REQUIRED_HEADINGS)
    memo = f"{headings}\n\n{EXPERIMENT_ID} {BELIEVED_BEST} short"
    with pytest.raises(ct.CloseoutError) as exc:
        _validate([{"path": ct.CLOSEOUT_MEMO_FILENAME, "content": memo}])
    assert str(exc.value).startswith(f"{ct.ERROR_PREFIX}memo_too_short:")


def test_disallowed_path_rejected() -> None:
    files = _valid_files() + [{"path": "../escape.md", "content": "x" * 4000}]
    with pytest.raises(ct.CloseoutError) as exc:
        _validate(files)
    assert str(exc.value) == ct.err_output_path_not_allowed("../escape.md")


def test_empty_content_rejected() -> None:
    with pytest.raises(ct.CloseoutError) as exc:
        _validate([{"path": ct.CLOSEOUT_MEMO_FILENAME, "content": "   "}])
    assert str(exc.value) == ct.err_output_content_empty(ct.CLOSEOUT_MEMO_FILENAME)


def test_missing_required_slot_rejected() -> None:
    with pytest.raises(ct.CloseoutError) as exc:
        _validate([{"path": "not-the-memo.md", "content": "x" * 4000}])
    # unexpected path trips the allowlist before the slot-missing check.
    assert str(exc.value) == ct.err_output_path_not_allowed("not-the-memo.md")
