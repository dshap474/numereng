"""Droid-exec transport contract for the agentic-research LLM layer.

Pins the `droid exec` provider path added alongside codex-exec and openrouter:
dispatch selection by ACTIVE_MODEL_SOURCE, command construction, JSON envelope
parsing, and the stable error tokens for each failure mode. `subprocess.run` is
mocked throughout — no droid binary is required.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from numereng.agentic_research.engine import llm
from numereng.agentic_research.engine import types as ar_types
from numereng.platform.clients.openrouter import OpenRouterConfig

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

RESPONSE_PAYLOAD = json.dumps(
    {
        "decision_form": {"action": "run"},
        "round_markdown": "## round",
        "experiment_markdown": None,
    }
)


def _envelope(result: str, *, is_error: bool = False) -> str:
    return json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": is_error,
            "duration_ms": 1,
            "num_turns": 1,
            "result": result,
            "session_id": "s-1",
        }
    )


def _droid_config(**overrides: object) -> OpenRouterConfig:
    values: dict[str, object] = {
        "active_model_source": "droid-exec",
        "active_model": "claude-fable-5",
        "active_model_reasoning_effort": "high",
    }
    values.update(overrides)
    return OpenRouterConfig(**values)  # type: ignore[arg-type]


def _run_ok(captured: dict[str, object], stdout: str) -> object:
    def fake_run(cmd: list[str], **kwargs: object) -> SimpleNamespace:
        captured["cmd"] = cmd
        captured["input"] = kwargs.get("input")
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    return fake_run


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #


def test_droid_source_dispatches_on_auto_and_codex_transports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "load_openrouter_config", _droid_config)
    calls: list[str] = []
    monkeypatch.setattr(llm, "_call_droid_exec", lambda **kwargs: calls.append("droid") or "raw")
    for transport in ("auto", "codex"):
        raw, source = llm._call_research_llm(prompt="p", artifact_dir=tmp_path, round_label="r1", transport=transport)
        assert (raw, source) == ("raw", "droid-exec")
    assert calls == ["droid", "droid"]


# --------------------------------------------------------------------------- #
# Command construction and envelope parsing
# --------------------------------------------------------------------------- #


def test_droid_exec_builds_command_and_returns_envelope_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(llm.subprocess, "run", _run_ok(captured, _envelope(RESPONSE_PAYLOAD)))
    result = llm._call_droid_exec(prompt="the prompt", artifact_dir=tmp_path, round_label="r1", config=_droid_config())
    assert result == RESPONSE_PAYLOAD
    cmd = captured["cmd"]
    assert cmd[1:3] == ["exec", "--output-format"]
    assert "json" in cmd
    assert ["--model", "claude-fable-5"] == cmd[cmd.index("--model") : cmd.index("--model") + 2]
    assert ["--reasoning-effort", "high"] == cmd[cmd.index("--reasoning-effort") : cmd.index("--reasoning-effort") + 2]
    prompt_sent = captured["input"]
    assert isinstance(prompt_sent, str)
    assert prompt_sent.startswith("the prompt")
    assert "JSON Schema" in prompt_sent
    assert '"decision_form"' in prompt_sent


def test_droid_exec_omits_model_flags_when_unset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(llm.subprocess, "run", _run_ok(captured, _envelope(RESPONSE_PAYLOAD)))
    llm._call_droid_exec(
        prompt="p",
        artifact_dir=tmp_path,
        round_label="r1",
        config=_droid_config(active_model=None, active_model_reasoning_effort=None),
    )
    cmd = captured["cmd"]
    assert "--model" not in cmd
    assert "--reasoning-effort" not in cmd


def test_droid_exec_custom_schema_replaces_default_in_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(llm.subprocess, "run", _run_ok(captured, _envelope("{}")))
    llm._call_droid_exec(
        prompt="p",
        artifact_dir=tmp_path,
        round_label="r1",
        config=_droid_config(),
        schema={"type": "object", "properties": {"files": {"type": "object"}}},
    )
    prompt_sent = captured["input"]
    assert isinstance(prompt_sent, str)
    assert '"files"' in prompt_sent
    assert '"decision_form"' not in prompt_sent


# --------------------------------------------------------------------------- #
# Failure modes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("stdout", "token"),
    [
        ("no json here", "agentic_research_droid_envelope_missing"),
        ("{broken}", "agentic_research_droid_envelope_invalid"),
        (_envelope(""), "agentic_research_droid_result_missing"),
        (json.dumps({"type": "result", "result": 7}), "agentic_research_droid_result_missing"),
        (_envelope("boom", is_error=True), "agentic_research_droid_error:boom"),
    ],
)
def test_droid_exec_envelope_failures_raise_stable_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stdout: str, token: str
) -> None:
    monkeypatch.setattr(llm.subprocess, "run", _run_ok({}, stdout))
    with pytest.raises(ar_types.AgenticResearchError) as excinfo:
        llm._call_droid_exec(prompt="p", artifact_dir=tmp_path, round_label="r1", config=_droid_config())
    assert token in str(excinfo.value)
    assert list(tmp_path.glob("*")), "failure debug dump should be written"


def test_droid_exec_nonzero_exit_raises_failed_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        llm.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=3, stdout="", stderr="denied"),
    )
    with pytest.raises(ar_types.AgenticResearchError, match="agentic_research_droid_failed:3:denied"):
        llm._call_droid_exec(prompt="p", artifact_dir=tmp_path, round_label="r1", config=_droid_config())


def test_droid_exec_timeout_raises_timeout_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd="droid", timeout=5)

    monkeypatch.setattr(llm.subprocess, "run", fake_run)
    with pytest.raises(ar_types.AgenticResearchError, match="agentic_research_droid_timeout:5"):
        llm._call_droid_exec(
            prompt="p", artifact_dir=tmp_path, round_label="r1", config=_droid_config(), timeout_seconds=5
        )


def test_droid_exec_missing_executable_raises_missing_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("droid")

    monkeypatch.setattr(llm.subprocess, "run", fake_run)
    with pytest.raises(ar_types.AgenticResearchError, match="agentic_research_droid_executable_missing"):
        llm._call_droid_exec(prompt="p", artifact_dir=tmp_path, round_label="r1", config=_droid_config())
