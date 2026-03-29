"""Planner backend wrappers for the agentic research supervisor."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path

from numereng.features.agentic_research.utils.store import decision_from_dict
from numereng.features.agentic_research.utils.types import (
    CodexPlannerAttempt,
    CodexPlannerExecution,
    RawPlannerExecution,
)
from numereng.platform.clients.openrouter import OpenRouterClient
from numereng.platform.errors import OpenRouterClientError

_REPO_ROOT = Path(__file__).resolve().parents[4]


class AgenticResearchCodexError(Exception):
    """Raised when one headless Codex planning call fails."""


def default_codex_command() -> list[str]:
    """Return the default headless Codex command for the planner."""
    return [
        "codex",
        "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "--json",
        "--color",
        "never",
        "-",
    ]


def run_codex_planner(
    *,
    prompt: str,
    command: list[str],
    schema_path: Path,
    artifact_dir: Path,
) -> CodexPlannerExecution:
    """Run headless Codex once with a structured JSON schema contract."""
    raw_execution, output_path = _run_codex_exec(
        prompt=prompt,
        command=[*command, "--output-schema", str(schema_path)],
        artifact_dir=artifact_dir,
    )
    try:
        last_message = _load_last_message(output_path)
        decision = decision_from_dict(last_message)
    except Exception as exc:
        raise AgenticResearchCodexError("agentic_research_codex_output_invalid") from exc
    finally:
        _unlink_if_exists(output_path)
    return CodexPlannerExecution(
        decision=decision,
        attempts=raw_execution.attempts,
        stdout_jsonl=raw_execution.stdout_jsonl,
        stderr_text=raw_execution.stderr_text,
        last_message=last_message,
        raw_response_text=raw_execution.stdout_jsonl or json.dumps(last_message, indent=2, sort_keys=True),
    )


def run_codex_raw_planner(
    *,
    prompt: str,
    command: list[str],
    artifact_dir: Path,
) -> RawPlannerExecution:
    """Run headless Codex once and capture the raw final text response."""
    raw_execution, output_path = _run_codex_exec(prompt=prompt, command=command, artifact_dir=artifact_dir)
    try:
        response_text = output_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AgenticResearchCodexError("agentic_research_codex_output_invalid") from exc
    finally:
        _unlink_if_exists(output_path)
    return RawPlannerExecution(
        attempts=raw_execution.attempts,
        stdout_jsonl=raw_execution.stdout_jsonl,
        stderr_text=raw_execution.stderr_text,
        raw_response_text=response_text,
    )


def _run_codex_exec(
    *,
    prompt: str,
    command: list[str],
    artifact_dir: Path,
) -> tuple[RawPlannerExecution, Path]:
    """Run the underlying headless Codex command once and capture telemetry."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    attempts: list[CodexPlannerAttempt] = []
    output_suffix = ".json" if "--output-schema" in command else ".txt"
    with tempfile.NamedTemporaryFile(
        dir=artifact_dir,
        prefix=".codex_output_",
        suffix=output_suffix,
        delete=False,
    ) as handle:
        output_path = Path(handle.name)
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [*command, "-o", str(output_path)],
            input=prompt,
            text=True,
            capture_output=True,
            cwd=_REPO_ROOT,
            check=False,
            env=_planner_env(),
        )
    except OSError as exc:  # pragma: no cover - subprocess creation failure is rare
        raise AgenticResearchCodexError("agentic_research_codex_exec_failed") from exc
    elapsed_seconds = time.perf_counter() - started
    parsed_lines = _parse_jsonl(completed.stdout)
    attempts.append(
        CodexPlannerAttempt(
            attempt_number=1,
            elapsed_seconds=elapsed_seconds,
            returncode=completed.returncode,
            thread_id=_find_thread_id(parsed_lines),
            input_tokens=_find_usage_value(parsed_lines, "input_tokens"),
            cached_input_tokens=_find_usage_value(parsed_lines, "cached_input_tokens"),
            output_tokens=_find_usage_value(parsed_lines, "output_tokens"),
            stdout_line_count=sum(1 for line in completed.stdout.splitlines() if line.strip()),
            validation_feedback=None,
        )
    )
    if completed.returncode != 0:
        raise AgenticResearchCodexError(
            f"agentic_research_codex_exec_failed:{completed.returncode}:{completed.stderr.strip()}"
        )
    return (
        RawPlannerExecution(
            attempts=attempts,
            stdout_jsonl=completed.stdout,
            stderr_text=completed.stderr.strip(),
            raw_response_text="",
        ),
        output_path,
    )


def _planner_env() -> dict[str, str]:
    return dict(os.environ)


def _unlink_if_exists(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:
        return


def _load_last_message(path: Path) -> dict[str, object]:
    return _load_json_mapping(
        path,
        invalid_json_error="agentic_research_codex_output_invalid_json",
        invalid_shape_error="agentic_research_codex_output_invalid_shape",
    )


def _parse_jsonl(stdout: str) -> list[dict[str, object]]:
    parsed: list[dict[str, object]] = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            parsed.append(payload)
    return parsed


def _find_thread_id(lines: list[dict[str, object]]) -> str | None:
    for payload in lines:
        thread_id = _deep_find_value(payload, "thread_id")
        if isinstance(thread_id, str) and thread_id.strip():
            return thread_id.strip()
    return None


def _find_usage_value(lines: list[dict[str, object]], key: str) -> int | None:
    for payload in reversed(lines):
        value = _deep_find_usage_value(payload, key)
        if isinstance(value, int):
            return value
    return None


def _deep_find_usage_value(payload: object, key: str) -> int | None:
    if isinstance(payload, dict):
        usage = payload.get("usage")
        if isinstance(usage, dict):
            value = usage.get(key)
            if isinstance(value, int):
                return value
        for value in payload.values():
            found = _deep_find_usage_value(value, key)
            if found is not None:
                return found
    if isinstance(payload, list):
        for item in payload:
            found = _deep_find_usage_value(item, key)
            if found is not None:
                return found
    return None


def _deep_find_value(payload: object, key: str) -> object | None:
    if isinstance(payload, dict):
        if key in payload:
            return payload[key]
        for value in payload.values():
            found = _deep_find_value(value, key)
            if found is not None:
                return found
    if isinstance(payload, list):
        for item in payload:
            found = _deep_find_value(item, key)
            if found is not None:
                return found
    return None


class AgenticResearchOpenRouterError(Exception):
    """Raised when one OpenRouter planning call fails."""


def run_openrouter_planner(
    *,
    prompt: str,
    schema_path: Path,
    artifact_dir: Path,
) -> CodexPlannerExecution:
    """Run the structured planner through OpenRouter once."""
    _ = artifact_dir
    schema = _load_schema(schema_path)
    client = OpenRouterClient()
    started = time.perf_counter()
    try:
        response = client.chat_completions(
            payload={
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "agentic_research_planner",
                        "strict": True,
                        "schema": schema,
                    },
                },
            }
        )
    except OpenRouterClientError as exc:
        raise AgenticResearchOpenRouterError("agentic_research_openrouter_exec_failed") from exc

    elapsed_seconds = time.perf_counter() - started
    usage = response.get("usage")
    attempts = [_openrouter_attempt(response=response, usage=usage, elapsed_seconds=elapsed_seconds)]

    try:
        last_message = _parse_last_message(response)
        decision = decision_from_dict(last_message)
    except Exception as exc:
        raise AgenticResearchOpenRouterError("agentic_research_openrouter_output_invalid") from exc
    return CodexPlannerExecution(
        decision=decision,
        attempts=attempts,
        stdout_jsonl=json.dumps(response) + "\n",
        stderr_text="",
        last_message=last_message,
        raw_response_text=json.dumps(last_message, indent=2, sort_keys=True),
    )


def run_openrouter_raw_planner(*, prompt: str) -> RawPlannerExecution:
    """Run the numerai mutation planner through OpenRouter and capture raw text."""
    client = OpenRouterClient()
    started = time.perf_counter()
    try:
        response = client.chat_completions(payload={"messages": [{"role": "user", "content": prompt}]})
    except OpenRouterClientError as exc:
        raise AgenticResearchOpenRouterError("agentic_research_openrouter_exec_failed") from exc
    elapsed_seconds = time.perf_counter() - started
    usage = response.get("usage")
    return RawPlannerExecution(
        attempts=[_openrouter_attempt(response=response, usage=usage, elapsed_seconds=elapsed_seconds)],
        stdout_jsonl=_extract_content_text(_message_content(response)) + "\n",
        stderr_text="",
        raw_response_text=_extract_content_text(_message_content(response)),
    )


def _load_schema(schema_path: Path) -> dict[str, object]:
    try:
        return _load_json_mapping(
            schema_path,
            invalid_json_error="agentic_research_openrouter_schema_invalid",
            invalid_shape_error="agentic_research_openrouter_schema_invalid",
        )
    except ValueError as exc:
        raise AgenticResearchOpenRouterError(str(exc)) from exc


def _load_json_mapping(path: Path, *, invalid_json_error: str, invalid_shape_error: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(invalid_json_error) from exc
    if not isinstance(payload, dict):
        raise ValueError(invalid_shape_error)
    return payload


def _openrouter_attempt(
    *,
    response: Mapping[str, object],
    usage: object,
    elapsed_seconds: float,
) -> CodexPlannerAttempt:
    return CodexPlannerAttempt(
        attempt_number=1,
        elapsed_seconds=elapsed_seconds,
        returncode=0,
        thread_id=_response_id(response),
        input_tokens=_usage_int(usage, "input_tokens"),
        cached_input_tokens=_cached_input_tokens(usage),
        output_tokens=_usage_int(usage, "output_tokens"),
        stdout_line_count=1,
        validation_feedback=None,
    )


def _response_id(response: Mapping[str, object]) -> str | None:
    value = response.get("id")
    return value if isinstance(value, str) else None


def _usage_int(usage: object, key: str) -> int | None:
    if isinstance(usage, Mapping):
        value = usage.get(key)
        if isinstance(value, int):
            return value
    return None


def _cached_input_tokens(usage: object) -> int | None:
    if not isinstance(usage, Mapping):
        return None
    input_tokens = usage.get("input_tokens_details")
    if isinstance(input_tokens, Mapping):
        value = input_tokens.get("cached_tokens")
        if isinstance(value, int):
            return value
    return None


def _parse_last_message(response: Mapping[str, object]) -> dict[str, object]:
    content = _extract_content_text(_message_content(response))
    try:
        payload = json.loads(_strip_code_fence(content))
    except json.JSONDecodeError as exc:
        raise AgenticResearchOpenRouterError("agentic_research_openrouter_output_invalid") from exc
    if not isinstance(payload, dict):
        raise AgenticResearchOpenRouterError("agentic_research_openrouter_output_invalid")
    return payload


def _message_content(response: Mapping[str, object]) -> object:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise AgenticResearchOpenRouterError("agentic_research_openrouter_output_invalid")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise AgenticResearchOpenRouterError("agentic_research_openrouter_output_invalid")
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise AgenticResearchOpenRouterError("agentic_research_openrouter_output_invalid")
    return message.get("content")


def _extract_content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                if item.get("type") == "text":
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                elif "text" in item:
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
        if parts:
            return "\n".join(parts)
    raise AgenticResearchOpenRouterError("agentic_research_openrouter_output_invalid")


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1]).strip()
    return stripped
