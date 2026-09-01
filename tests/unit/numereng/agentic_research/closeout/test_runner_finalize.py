"""End-to-end closeout with a stubbed LLM (no network, no training).

Covers the happy path, memo validation, and the re-run-overwrites contract.
"""

from __future__ import annotations

import json

import pytest

from numereng.agentic_research.engine.closeout import runner
from numereng.agentic_research.engine.closeout import types as ct

from .conftest import CloseoutFixture, install_fake_llm


def _run(fixture: CloseoutFixture, **kwargs):
    return runner.run_closeout(store_root=fixture.store_root, experiment_id=fixture.experiment_id, **kwargs)


def test_closeout_writes_evidence_memo_and_raw_response(
    closeout_fixture: CloseoutFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_fake_llm(monkeypatch, closeout_fixture)
    result = _run(closeout_fixture)

    directory = closeout_fixture.closeout_dir()
    assert result.experiment_id == closeout_fixture.experiment_id
    assert result.evidence_path == directory / ct.CLOSEOUT_EVIDENCE_FILENAME
    assert result.memo_path == directory / ct.CLOSEOUT_MEMO_FILENAME
    assert result.holdout_summary is None

    memo = result.memo_path.read_text(encoding="utf-8")
    assert memo.startswith("## Verdict")
    assert closeout_fixture.experiment_id in memo
    assert closeout_fixture.believed_best_config in memo

    evidence = json.loads(result.evidence_path.read_text(encoding="utf-8"))
    assert evidence["experiment_id"] == closeout_fixture.experiment_id
    assert evidence["believed_best"]["config"] == closeout_fixture.believed_best_config
    # The raw response lands beside the evidence file.
    assert (directory / ct.CLOSEOUT_RESPONSE_FILENAME).is_file()


def test_finalize_prompt_carries_the_bounded_context(
    closeout_fixture: CloseoutFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = install_fake_llm(monkeypatch, closeout_fixture)
    _run(closeout_fixture)
    prompt = calls["prompts"][0]
    assert "{{CONTEXT_JSON}}" not in prompt
    assert "# Closeout: Finalize" in prompt
    assert '"evidence_summary"' in prompt
    assert len(prompt) <= ct.MAX_CLOSEOUT_CONTEXT_CHARS + len(runner.FINALIZE_PROMPT_PATH.read_text(encoding="utf-8"))


def test_memo_without_verdict_heading_is_rejected(
    closeout_fixture: CloseoutFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_fake_llm(monkeypatch, closeout_fixture, raw="## Summary\n\n" + ("filler text. " * 400))
    with pytest.raises(ct.CloseoutError) as exc:
        _run(closeout_fixture)
    assert str(exc.value) == f"{ct.ERROR_PREFIX}memo_section_missing:## Verdict"
    # The unusable response is still on disk, the memo is not.
    assert (closeout_fixture.closeout_dir() / ct.CLOSEOUT_RESPONSE_FILENAME).is_file()
    assert not (closeout_fixture.closeout_dir() / ct.CLOSEOUT_MEMO_FILENAME).exists()


def test_short_memo_is_rejected(closeout_fixture: CloseoutFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_llm(monkeypatch, closeout_fixture, raw="## Verdict\n\nToo short.")
    with pytest.raises(ct.CloseoutError) as exc:
        _run(closeout_fixture)
    assert str(exc.value).startswith(f"{ct.ERROR_PREFIX}memo_too_short:")


def test_rerunning_overwrites_both_artifacts(
    closeout_fixture: CloseoutFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = install_fake_llm(monkeypatch, closeout_fixture)
    first = _run(closeout_fixture)
    assert calls["n"] == 1

    second_memo = "## Verdict\n\nRerun verdict. " + ("Fresh evidence text. " * 200)
    install_fake_llm(monkeypatch, closeout_fixture, raw=second_memo)
    second = _run(closeout_fixture)

    assert second.memo_path == first.memo_path
    assert second.memo_path.read_text(encoding="utf-8").startswith("## Verdict\n\nRerun verdict.")
    assert "Rerun verdict" in (closeout_fixture.closeout_dir() / ct.CLOSEOUT_RESPONSE_FILENAME).read_text(
        encoding="utf-8"
    )
    # Exactly the expected artifacts, no stage/lock/commit leftovers.
    assert {path.name for path in closeout_fixture.closeout_dir().iterdir() if path.is_file()} == {
        ct.CLOSEOUT_EVIDENCE_FILENAME,
        ct.CLOSEOUT_MEMO_FILENAME,
        ct.CLOSEOUT_RESPONSE_FILENAME,
    }
