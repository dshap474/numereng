"""LLM transport: prompt render, codex-exec + openrouter, static schema, parse/validate."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path

from numereng.features.agentic_research import memory
from numereng.features.agentic_research.types import (
    CODEX_TIMEOUT_SECONDS,
    PROGRAM_PATH,
    AgenticResearchError,
    AgenticResearchValidationError,
    ResearchChange,
    ResearchDecision,
    ResearchLLMResponse,
    as_list,
    optional_str,
    required_str,
)
from numereng.platform.clients.openrouter import OpenRouterClient, OpenRouterConfig, load_openrouter_config
from numereng.platform.errors import OpenRouterClientError


def render_prompt(context: dict[str, object], *, program_path: Path = PROGRAM_PATH) -> str:
    """Render the program file with the context JSON substituted in."""
    context_json = json.dumps(context, indent=2, sort_keys=True, default=str)
    return program_path.read_text(encoding="utf-8").replace("{{CONTEXT_JSON}}", context_json)


def _call_research_llm(*, prompt: str, artifact_dir: Path, round_label: str) -> tuple[str, str]:
    """Call the active backend (openrouter or codex-exec) and return (raw_response, source)."""
    config = load_openrouter_config()
    if config.active_model_source == "openrouter":
        return _call_openrouter(prompt, config=config), "openrouter"
    raw = _call_codex_exec(prompt=prompt, artifact_dir=artifact_dir, round_label=round_label, config=config)
    return raw, "codex-exec"


def _call_openrouter(prompt: str, *, config: OpenRouterConfig) -> str:
    payload: dict[str, object] = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    if config.active_model_reasoning_effort is not None:
        payload["reasoning"] = {"effort": config.active_model_reasoning_effort}
    try:
        response = OpenRouterClient(timeout_seconds=180.0).chat_completions(payload=payload)
    except OpenRouterClientError as exc:
        raise AgenticResearchError(str(exc)) from exc
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise AgenticResearchError("agentic_research_openrouter_response_missing")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise AgenticResearchError("agentic_research_openrouter_content_missing")
    return content


def _call_codex_exec(*, prompt: str, artifact_dir: Path, round_label: str, config: OpenRouterConfig) -> str:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=artifact_dir, prefix=".codex_output_", suffix=".txt", delete=False) as handle:
        output_path = Path(handle.name)
    with tempfile.NamedTemporaryFile(dir=artifact_dir, prefix=".codex_schema_", suffix=".json", delete=False) as handle:
        schema_path = Path(handle.name)
    memory.write_json(schema_path, LLM_RESPONSE_SCHEMA)
    cmd = [_resolve_codex_executable(), "exec"]
    if config.active_model is not None:
        cmd.extend(["--model", config.active_model])
    if config.active_model_reasoning_effort is not None:
        cmd.extend(["-c", f'model_reasoning_effort="{config.active_model_reasoning_effort}"'])
    cmd.extend(
        [
            # The research call only needs JSON back; disabling image_generation removes a
            # known transient failure mode (a 400 on the image model bailed a live run).
            "--disable",
            "image_generation",
            "--skip-git-repo-check",
            "--ephemeral",
            "--output-schema",
            str(schema_path),
            "--json",
            "--color",
            "never",
            "-",
            "-o",
            str(output_path),
        ]
    )
    try:
        try:
            completed = subprocess.run(
                cmd,
                input=prompt,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                check=False,
                timeout=CODEX_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            error = f"agentic_research_codex_timeout:{int(CODEX_TIMEOUT_SECONDS)}"
            memory.write_failure_debug(artifact_dir=artifact_dir, round_label=round_label, prompt=prompt, error=error)
            raise AgenticResearchError(error) from exc
        except FileNotFoundError as exc:
            error = f"agentic_research_codex_executable_missing:{cmd[0]}"
            memory.write_failure_debug(artifact_dir=artifact_dir, round_label=round_label, prompt=prompt, error=error)
            raise AgenticResearchError(error) from exc
        if completed.returncode != 0:
            # The stderr is folded into the stable error token, which the debug dump records.
            error = f"agentic_research_codex_failed:{completed.returncode}:{completed.stderr.strip()}"
            memory.write_failure_debug(artifact_dir=artifact_dir, round_label=round_label, prompt=prompt, error=error)
            raise AgenticResearchError(error)
        return output_path.read_text(encoding="utf-8")
    finally:
        for tmp in (output_path, schema_path):
            try:
                tmp.unlink()
            except OSError:
                pass


def _resolve_codex_executable() -> str:
    if os.name == "nt":
        return shutil.which("codex.cmd") or shutil.which("codex.exe") or shutil.which("codex") or "codex.cmd"
    return shutil.which("codex") or "codex"


def parse_llm_response(raw_response: str) -> ResearchLLMResponse:
    """Parse and validate the raw model response into a structured decision + memo."""
    payload = _extract_json_object(raw_response)
    decision_form = payload.get("decision_form")
    if not isinstance(decision_form, dict):
        raise AgenticResearchValidationError("agentic_research_decision_form_missing")
    experiment_markdown_raw = payload.get("experiment_markdown")
    if experiment_markdown_raw is not None and not isinstance(experiment_markdown_raw, str):
        raise AgenticResearchValidationError("agentic_research_experiment_markdown_invalid")
    return ResearchLLMResponse(
        decision=_parse_decision_object(decision_form),
        round_markdown=required_str(payload, "round_markdown"),
        experiment_markdown=experiment_markdown_raw,
    )


def _parse_decision_object(payload: dict[str, object]) -> ResearchDecision:
    # Budget-bounded design: the only action is `run`. There is no `stop` or `ensemble`.
    if payload.get("action") != "run":
        raise AgenticResearchValidationError("agentic_research_action_invalid")
    changes = tuple(_parse_change(item) for item in as_list(payload.get("changes")))
    decision = ResearchDecision(
        action="run",
        learning=required_str(payload, "learning"),
        belief_update=required_str(payload, "belief_update"),
        next_hypothesis=optional_str(payload.get("next_hypothesis")),
        parent_config=optional_str(payload.get("parent_config")),
        changes=changes,
        stop_reason=optional_str(payload.get("stop_reason")),
    )
    if decision.parent_config is None:
        raise AgenticResearchValidationError("agentic_research_parent_config_missing")
    if not 1 <= len(decision.changes) <= 5:
        raise AgenticResearchValidationError("agentic_research_change_count_invalid")
    return decision


def _parse_change(payload: object) -> ResearchChange:
    if not isinstance(payload, dict):
        raise AgenticResearchValidationError("agentic_research_change_invalid")
    return ResearchChange(
        path=required_str(payload, "path"),
        value=deepcopy(payload.get("value")),
        reason=required_str(payload, "reason"),
    )


# Static output schema: a single `run` action (no dynamic gating). stop_reason is kept
# nullable-and-ignored for output-shape stability across models.
_SCALAR_TYPES = [{"type": t} for t in ("string", "number", "integer", "boolean", "null")]
LLM_RESPONSE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "decision_form": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["run"]},
                "learning": {"type": "string"},
                "belief_update": {"type": "string"},
                "next_hypothesis": {"type": ["string", "null"]},
                "parent_config": {"type": ["string", "null"]},
                "changes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "value": {"anyOf": [*_SCALAR_TYPES, {"type": "array", "items": {"anyOf": _SCALAR_TYPES}}]},
                            "reason": {"type": "string"},
                        },
                        "required": ["path", "value", "reason"],
                        "additionalProperties": False,
                    },
                },
                "stop_reason": {"type": ["string", "null"]},
            },
            "required": [
                "action",
                "learning",
                "belief_update",
                "next_hypothesis",
                "parent_config",
                "changes",
                "stop_reason",
            ],
            "additionalProperties": False,
        },
        "round_markdown": {"type": "string"},
        "experiment_markdown": {"type": ["string", "null"]},
    },
    "required": ["decision_form", "round_markdown", "experiment_markdown"],
    "additionalProperties": False,
}


def _extract_json_object(text: str) -> dict[str, object]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise AgenticResearchValidationError("agentic_research_json_missing")
    try:
        payload = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError as exc:
        raise AgenticResearchValidationError("agentic_research_json_invalid") from exc
    if not isinstance(payload, dict):
        raise AgenticResearchValidationError("agentic_research_json_object_required")
    return payload
