"""OpenRouter-backed planner wrapper for the agentic research supervisor."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path

from numereng.features.agentic_research.contracts import (
    CodexPlannerAttempt,
    CodexPlannerExecution,
    RawPlannerExecution,
)
from numereng.features.agentic_research.state import decision_from_dict
from numereng.platform.clients.openrouter import OpenRouterClient
from numereng.platform.errors import OpenRouterClientError


class AgenticResearchOpenRouterError(Exception):
    """Raised when one OpenRouter planning call fails."""


def run_openrouter_planner(
    *,
    prompt: str,
    schema_path: Path,
    artifact_dir: Path,
) -> CodexPlannerExecution:
    """Run the structured planner through OpenRouter once."""
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
    attempts = [
        CodexPlannerAttempt(
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
    ]

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
        attempts=[
            CodexPlannerAttempt(
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
        ],
        stdout_jsonl=_extract_content_text(_message_content(response)) + "\n",
        stderr_text="",
        raw_response_text=_extract_content_text(_message_content(response)),
    )


def _load_schema(schema_path: Path) -> dict[str, object]:
    try:
        payload = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AgenticResearchOpenRouterError("agentic_research_openrouter_schema_invalid") from exc
    if not isinstance(payload, dict):
        raise AgenticResearchOpenRouterError("agentic_research_openrouter_schema_invalid")
    return payload


def _response_id(response: Mapping[str, object]) -> str | None:
    value = response.get("id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _usage_int(usage: object, key: str) -> int | None:
    if not isinstance(usage, dict):
        return None
    value = usage.get(key)
    if isinstance(value, int):
        return value
    return None


def _cached_input_tokens(usage: object) -> int | None:
    if not isinstance(usage, dict):
        return None
    details = usage.get("prompt_tokens_details")
    if not isinstance(details, dict):
        return None
    value = details.get("cached_tokens")
    if isinstance(value, int):
        return value
    return None


def _parse_last_message(response: Mapping[str, object]) -> dict[str, object]:
    content = _extract_content_text(_message_content(response))
    parsed = json.loads(_strip_code_fence(content))
    if not isinstance(parsed, dict):
        raise ValueError("agentic_research_openrouter_output_invalid_shape")
    return parsed


def _message_content(response: Mapping[str, object]) -> object:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("agentic_research_openrouter_output_missing_choices")
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ValueError("agentic_research_openrouter_output_missing_choices")
    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("agentic_research_openrouter_output_missing_message")
    return message.get("content")


def _extract_content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
        if parts:
            return "".join(parts)
    raise ValueError("agentic_research_openrouter_output_missing_content")


def _strip_code_fence(content: str) -> str:
    stripped = content.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < 3:
        return stripped
    if not lines[-1].startswith("```"):
        return stripped
    return "\n".join(lines[1:-1]).strip()
