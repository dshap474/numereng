"""Backend contract for the agentic-research LLM layer.

Pins what ACTIVE_MODEL_SOURCE selects, how the `droid exec` command is built, how its JSON
envelope is parsed, and the stable error token for each failure mode. `subprocess.run` is
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


def test_active_model_source_picks_the_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The config alone decides: droid-exec dispatches to droid, anything else to codex."""
    calls: list[str] = []
    monkeypatch.setattr(llm, "_call_droid_exec", lambda **kwargs: calls.append("droid") or "raw")
    monkeypatch.setattr(llm, "_call_codex_exec", lambda **kwargs: calls.append("codex") or "raw")

    monkeypatch.setattr(llm, "load_openrouter_config", _droid_config)
    assert llm.call_research_llm(prompt="p", artifact_dir=tmp_path, round_label="r1") == (
        "raw",
        "droid-exec:claude-fable-5:high",
    )

    codex_config = OpenRouterConfig(active_model_source="codex-exec", active_model="gpt-5.5")
    monkeypatch.setattr(llm, "load_openrouter_config", lambda: codex_config)
    assert llm.call_research_llm(prompt="p", artifact_dir=tmp_path, round_label="r1") == (
        "raw",
        "codex-exec:gpt-5.5",
    )
    assert calls == ["droid", "codex"]


# --------------------------------------------------------------------------- #
# Model rotation
# --------------------------------------------------------------------------- #

_ROTATION = (("model-a", "medium"), ("model-b", "high"), ("model-c", "xhigh"))


def test_rotation_cycles_models_by_round_number(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "load_openrouter_config", lambda: _droid_config(active_model_rotation=_ROTATION))
    captured_configs: list[OpenRouterConfig] = []

    def fake_droid(**kwargs: object) -> str:
        captured_configs.append(kwargs["config"])  # type: ignore[arg-type]
        return "raw"

    monkeypatch.setattr(llm, "_call_droid_exec", fake_droid)
    sources = [
        llm.call_research_llm(prompt="p", artifact_dir=tmp_path, round_label=label)[1]
        for label in ("r001", "r002", "r003", "r004")
    ]
    assert [(c.active_model, c.active_model_reasoning_effort) for c in captured_configs] == [
        ("model-b", "high"),  # 1 % 3
        ("model-c", "xhigh"),  # 2 % 3
        ("model-a", "medium"),  # 3 % 3
        ("model-b", "high"),  # 4 % 3
    ]
    assert sources == [
        "droid-exec:model-b:high",
        "droid-exec:model-c:xhigh",
        "droid-exec:model-a:medium",
        "droid-exec:model-b:high",
    ]


def test_rotation_skipped_for_non_round_labels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "load_openrouter_config", lambda: _droid_config(active_model_rotation=_ROTATION))
    captured_configs: list[OpenRouterConfig] = []

    def fake_droid(**kwargs: object) -> str:
        captured_configs.append(kwargs["config"])  # type: ignore[arg-type]
        return "raw"

    monkeypatch.setattr(llm, "_call_droid_exec", fake_droid)
    for label in ("closeout-stage-1", "finalize", "r1x"):
        _, source = llm.call_research_llm(prompt="p", artifact_dir=tmp_path, round_label=label)
        assert source == "droid-exec:claude-fable-5:high"
    assert all(c.active_model == "claude-fable-5" for c in captured_configs)


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
    disabled = cmd[cmd.index("--disabled-tools") + 1].split(",")
    assert disabled == list(llm.DROID_DISABLED_TOOLS)
    assert {"WebSearch", "FetchUrl", "Read", "Execute"} <= set(disabled)
    assert "--auto" not in cmd and "--enabled-tools" not in cmd
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


def test_droid_exec_schema_none_sends_plain_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`schema=None` (the closeout memo call) must not append any JSON-Schema instruction."""
    captured: dict[str, object] = {}
    monkeypatch.setattr(llm.subprocess, "run", _run_ok(captured, _envelope("## Verdict\n\nmemo")))
    llm._call_droid_exec(prompt="p", artifact_dir=tmp_path, round_label="closeout", config=_droid_config(), schema=None)
    assert captured["input"] == "p"


def test_codex_exec_schema_none_omits_output_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The codex transport drops `--output-schema` (and its temp file) for a plain-text call."""
    captured: dict[str, object] = {}

    def fake_run(cmd: list[str], **kwargs: object) -> SimpleNamespace:
        captured["cmd"] = cmd
        Path(cmd[cmd.index("-o") + 1]).write_text("## Verdict\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(llm.subprocess, "run", fake_run)
    result = llm._call_codex_exec(
        prompt="p",
        artifact_dir=tmp_path,
        round_label="closeout",
        config=OpenRouterConfig(active_model_source="codex-exec", active_model=None),
        schema=None,
    )
    assert result == "## Verdict\n"
    assert "--output-schema" not in captured["cmd"]
    assert not list(tmp_path.glob(".codex_schema_*"))


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
