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

from numereng.agentic_research.engine import memory
from numereng.agentic_research.engine import types as ar_types
from numereng.platform.clients.openrouter import OpenRouterClient, OpenRouterConfig, load_openrouter_config
from numereng.platform.errors import OpenRouterClientError


def render_prompt(context: dict[str, object], *, program_path: Path = ar_types.PROGRAM_PATH) -> str:
    context_json = json.dumps(context, indent=2, sort_keys=True, default=str)
    return program_path.read_text(encoding="utf-8").replace("{{CONTEXT_JSON}}", context_json)


def call_research_llm(
    *,
    prompt: str,
    artifact_dir: Path,
    round_label: str,
    schema: dict[str, object] | None = None,
    timeout_seconds: float = ar_types.CODEX_TIMEOUT_SECONDS,
    transport: str = "auto",
) -> tuple[str, str]:
    """Call the active research LLM; return ``(response_text, source)``.

    ``transport`` only gates the openrouter path: ``"auto"`` allows openrouter when it is the
    active model source, anything else (e.g. ``"codex"``) excludes it. The remaining choice is
    made by the config alone — ``active_model_source == "droid-exec"`` dispatches to droid-exec,
    otherwise codex-exec. So ``transport="codex"`` does not force codex.
    """
    config = load_openrouter_config()
    if transport == "auto" and config.active_model_source == "openrouter":
        return _call_openrouter(prompt, config=config), "openrouter"
    if config.active_model_source == "droid-exec":
        return _call_droid_exec(
            prompt=prompt,
            artifact_dir=artifact_dir,
            round_label=round_label,
            config=config,
            schema=schema,
            timeout_seconds=timeout_seconds,
        ), "droid-exec"
    return _call_codex_exec(
        prompt=prompt,
        artifact_dir=artifact_dir,
        round_label=round_label,
        config=config,
        schema=schema,
        timeout_seconds=timeout_seconds,
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


def _call_codex_exec(
    *,
    prompt: str,
    artifact_dir: Path,
    round_label: str,
    config: OpenRouterConfig,
    schema: dict[str, object] | None = None,
    timeout_seconds: float = ar_types.CODEX_TIMEOUT_SECONDS,
) -> str:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=artifact_dir, prefix=".codex_output_", suffix=".txt", delete=False) as handle:
        output_path = Path(handle.name)
    with tempfile.NamedTemporaryFile(dir=artifact_dir, prefix=".codex_schema_", suffix=".json", delete=False) as handle:
        schema_path = Path(handle.name)
    ar_types.write_json(schema_path, LLM_RESPONSE_SCHEMA if schema is None else schema)
    cmd = [_resolve_executable("codex"), "exec"]
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
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            error = f"agentic_research_codex_timeout:{int(timeout_seconds)}"
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


def _call_droid_exec(
    *,
    prompt: str,
    artifact_dir: Path,
    round_label: str,
    config: OpenRouterConfig,
    schema: dict[str, object] | None = None,
    timeout_seconds: float = ar_types.CODEX_TIMEOUT_SECONDS,
) -> str:
    """Call Factory's headless `droid exec` (default read-only autonomy, JSON envelope on stdout).

    droid exec has no --output-schema equivalent, so the schema is appended to the prompt and the
    response is validated downstream by the normal parser.
    """
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cmd = [_resolve_executable("droid"), "exec", "--output-format", "json"]
    if config.active_model is not None:
        cmd.extend(["--model", config.active_model])
    if config.active_model_reasoning_effort is not None:
        cmd.extend(["--reasoning-effort", config.active_model_reasoning_effort])
    schema_text = json.dumps(LLM_RESPONSE_SCHEMA if schema is None else schema, indent=2)
    full_prompt = (
        f"{prompt}\n\n"
        "Respond with ONLY one JSON object (no prose, no code fences) that validates against this "
        f"JSON Schema:\n{schema_text}"
    )
    try:
        completed = subprocess.run(
            cmd,
            input=full_prompt,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        error = f"agentic_research_droid_timeout:{int(timeout_seconds)}"
        memory.write_failure_debug(artifact_dir=artifact_dir, round_label=round_label, prompt=prompt, error=error)
        raise ar_types.AgenticResearchError(error) from exc
    except FileNotFoundError as exc:
        error = f"agentic_research_droid_executable_missing:{cmd[0]}"
        memory.write_failure_debug(artifact_dir=artifact_dir, round_label=round_label, prompt=prompt, error=error)
        raise ar_types.AgenticResearchError(error) from exc
    if completed.returncode != 0:
        # droid exec reports failures inside the stdout JSON envelope, not on stderr.
        detail = completed.stderr.strip() or _droid_failure_detail(completed.stdout)
        error = f"agentic_research_droid_failed:{completed.returncode}:{detail}"
        memory.write_failure_debug(artifact_dir=artifact_dir, round_label=round_label, prompt=prompt, error=error)
        raise ar_types.AgenticResearchError(error)
    try:
        return _parse_droid_envelope(completed.stdout)
    except ar_types.AgenticResearchError as exc:
        memory.write_failure_debug(artifact_dir=artifact_dir, round_label=round_label, prompt=prompt, error=str(exc))
        raise


def _droid_failure_detail(stdout: str) -> str:
    try:
        return str(json.loads(stdout.strip()).get("result", ""))[:500]
    except (json.JSONDecodeError, AttributeError):
        return stdout.strip()[:500]


def _parse_droid_envelope(stdout: str) -> str:
    """Extract the agent's final message from the `droid exec --output-format json` envelope."""
    stripped = stdout.strip()
    start, end = stripped.find("{"), stripped.rfind("}")
    if start < 0 or end < start:
        raise ar_types.AgenticResearchError("agentic_research_droid_envelope_missing")
    try:
        envelope = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ar_types.AgenticResearchError("agentic_research_droid_envelope_invalid") from exc
    if not isinstance(envelope, dict):
        raise ar_types.AgenticResearchError("agentic_research_droid_envelope_invalid")
    if envelope.get("is_error"):
        raise ar_types.AgenticResearchError(f"agentic_research_droid_error:{envelope.get('result')}")
    result = envelope.get("result")
    if not isinstance(result, str) or not result.strip():
        raise ar_types.AgenticResearchError("agentic_research_droid_result_missing")
    return result


def _resolve_executable(name: str) -> str:
    if os.name == "nt":
        return shutil.which(f"{name}.cmd") or shutil.which(f"{name}.exe") or shutil.which(name) or f"{name}.cmd"
    return shutil.which(name) or name


def parse_llm_response(raw_response: str) -> ar_types.ResearchLLMResponse:
    payload = extract_json_object(raw_response)
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
        believed_best=ar_types.optional_str(payload.get("believed_best")),
        seeds=_parse_seeds(payload.get("seeds")),
    )
    if decision.parent_config is None:
        raise ar_types.AgenticResearchValidationError("agentic_research_parent_config_missing")
    if not 1 <= len(decision.changes) <= 5:
        raise ar_types.AgenticResearchValidationError("agentic_research_change_count_invalid")
    return decision


def _parse_seeds(value: object) -> tuple[int, ...]:
    """Optional multi-seed request: absent/null -> single run; otherwise 1-3 integer seeds."""
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ar_types.AgenticResearchValidationError("agentic_research_seeds_invalid")
    seeds: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ar_types.AgenticResearchValidationError("agentic_research_seeds_invalid")
        seeds.append(item)
    if not 1 <= len(seeds) <= 3:
        raise ar_types.AgenticResearchValidationError("agentic_research_seeds_count_invalid")
    return tuple(seeds)


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
    "believed_best": {"type": ["string", "null"]},
    "seeds": {"type": ["array", "null"], "items": {"type": "integer"}, "minItems": 1, "maxItems": 3},
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


def extract_json_object(text: str) -> dict[str, object]:
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
