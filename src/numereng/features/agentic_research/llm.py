"""LLM prompt, transport, schema, and parser."""

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
from numereng.features.agentic_research import types as ar_types
from numereng.platform.clients.openrouter import OpenRouterClient, OpenRouterConfig, load_openrouter_config
from numereng.platform.errors import OpenRouterClientError


def render_prompt(context: dict[str, object], *, program_path: Path = ar_types.PROGRAM_PATH) -> str:
    context_json = json.dumps(context, indent=2, sort_keys=True, default=str)
    return program_path.read_text(encoding="utf-8").replace("{{CONTEXT_JSON}}", context_json)


def _call_research_llm(*, prompt: str, artifact_dir: Path, round_label: str) -> tuple[str, str]:
    config = load_openrouter_config()
    if config.active_model_source == "openrouter":
        return _call_openrouter(prompt, config=config), "openrouter"
    return _call_codex_exec(
        prompt=prompt, artifact_dir=artifact_dir, round_label=round_label, config=config
    ), "codex-exec"


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
        raise ar_types.AgenticResearchError(str(exc)) from exc
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ar_types.AgenticResearchError("agentic_research_openrouter_response_missing")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise ar_types.AgenticResearchError("agentic_research_openrouter_content_missing")
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
                timeout=ar_types.CODEX_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            error = f"agentic_research_codex_timeout:{int(ar_types.CODEX_TIMEOUT_SECONDS)}"
            memory.write_failure_debug(artifact_dir=artifact_dir, round_label=round_label, prompt=prompt, error=error)
            raise ar_types.AgenticResearchError(error) from exc
        except FileNotFoundError as exc:
            error = f"agentic_research_codex_executable_missing:{cmd[0]}"
            memory.write_failure_debug(artifact_dir=artifact_dir, round_label=round_label, prompt=prompt, error=error)
            raise ar_types.AgenticResearchError(error) from exc
        if completed.returncode != 0:
            error = f"agentic_research_codex_failed:{completed.returncode}:{completed.stderr.strip()}"
            memory.write_failure_debug(artifact_dir=artifact_dir, round_label=round_label, prompt=prompt, error=error)
            raise ar_types.AgenticResearchError(error)
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


def parse_llm_response(raw_response: str) -> ar_types.ResearchLLMResponse:
    payload = _extract_json_object(raw_response)
    decision_form = payload.get("decision_form")
    if not isinstance(decision_form, dict):
        raise ar_types.AgenticResearchValidationError("agentic_research_decision_form_missing")
    experiment_markdown = payload.get("experiment_markdown")
    if experiment_markdown is not None and not isinstance(experiment_markdown, str):
        raise ar_types.AgenticResearchValidationError("agentic_research_experiment_markdown_invalid")
    return ar_types.ResearchLLMResponse(
        decision=_parse_decision_object(decision_form),
        round_markdown=ar_types.required_str(payload, "round_markdown"),
        experiment_markdown=experiment_markdown,
    )


def _parse_decision_object(payload: dict[str, object]) -> ar_types.ResearchDecision:
    if payload.get("action") != "run":
        raise ar_types.AgenticResearchValidationError("agentic_research_action_invalid")
    decision = ar_types.ResearchDecision(
        action="run",
        learning=ar_types.required_str(payload, "learning"),
        belief_update=ar_types.required_str(payload, "belief_update"),
        next_hypothesis=ar_types.optional_str(payload.get("next_hypothesis")),
        parent_config=ar_types.optional_str(payload.get("parent_config")),
        changes=tuple(_parse_change(item) for item in ar_types.as_list(payload.get("changes"))),
        stop_reason=ar_types.optional_str(payload.get("stop_reason")),
    )
    if decision.parent_config is None:
        raise ar_types.AgenticResearchValidationError("agentic_research_parent_config_missing")
    if not 1 <= len(decision.changes) <= 5:
        raise ar_types.AgenticResearchValidationError("agentic_research_change_count_invalid")
    return decision


def _parse_change(payload: object) -> ar_types.ResearchChange:
    if not isinstance(payload, dict):
        raise ar_types.AgenticResearchValidationError("agentic_research_change_invalid")
    return ar_types.ResearchChange(
        path=ar_types.required_str(payload, "path"),
        value=deepcopy(payload.get("value")),
        reason=ar_types.required_str(payload, "reason"),
    )


_SCALAR_TYPES = [{"type": kind} for kind in ("string", "number", "integer", "boolean", "null")]
_CHANGE_PROPS: dict[str, object] = {
    "path": {"type": "string"},
    "value": {"anyOf": [*_SCALAR_TYPES, {"type": "array", "items": {"anyOf": _SCALAR_TYPES}}]},
    "reason": {"type": "string"},
}
_DECISION_PROPS: dict[str, object] = {
    "action": {"type": "string", "enum": ["run"]},
    "learning": {"type": "string"},
    "belief_update": {"type": "string"},
    "next_hypothesis": {"type": ["string", "null"]},
    "parent_config": {"type": ["string", "null"]},
    "changes": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": _CHANGE_PROPS,
            "required": list(_CHANGE_PROPS),
            "additionalProperties": False,
        },
    },
    "stop_reason": {"type": ["string", "null"]},
}
LLM_RESPONSE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "decision_form": {
            "type": "object",
            "properties": _DECISION_PROPS,
            "required": list(_DECISION_PROPS),
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
    start, end = stripped.find("{"), stripped.rfind("}")
    if start < 0 or end < start:
        raise ar_types.AgenticResearchValidationError("agentic_research_json_missing")
    try:
        payload = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ar_types.AgenticResearchValidationError("agentic_research_json_invalid") from exc
    if not isinstance(payload, dict):
        raise ar_types.AgenticResearchValidationError("agentic_research_json_object_required")
    return payload
