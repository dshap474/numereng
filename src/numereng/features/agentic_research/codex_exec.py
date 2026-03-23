"""Headless Codex wrapper for the agentic research supervisor."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Final

from numereng.features.agentic_research.contracts import CodexPlannerAttempt, CodexPlannerExecution
from numereng.features.agentic_research.state import decision_from_dict

_REPO_ROOT = Path(__file__).resolve().parents[4]
_USER_CODEX_HOME: Final[Path] = Path.home() / ".codex"
_LEARNER_CODEX_HOME: Final[Path] = _USER_CODEX_HOME / "agentic_research_learner"
_LEARNER_CONFIG: Final[str] = """model = "gpt-5.4"
model_reasoning_effort = "low"
approval_policy = "never"
sandbox_mode = "read-only"
service_tier = "fast"
web_search = "disabled"
project_doc_fallback_filenames = ["CLAUDE.md"]
suppress_unstable_features_warning = true

[features]
apps = false
multi_agent = false
js_repl = false
shell_snapshot = true
shell_tool = false
"""


class AgenticResearchCodexError(Exception):
    """Raised when one headless Codex planning call fails."""


def default_codex_command() -> list[str]:
    """Return the default lean headless Codex command for the planner."""
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


def learner_codex_home() -> Path:
    """Return the dedicated persistent CODEX_HOME for the learner."""
    return _LEARNER_CODEX_HOME


def ensure_learner_codex_home() -> Path:
    """Create or refresh the learner-specific CODEX_HOME directory."""
    home = learner_codex_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.toml").write_text(_LEARNER_CONFIG, encoding="utf-8")
    source_auth = _USER_CODEX_HOME / "auth.json"
    if source_auth.is_file():
        target_auth = home / "auth.json"
        needs_copy = not target_auth.is_file()
        if not needs_copy:
            try:
                source_stat = source_auth.stat()
                target_stat = target_auth.stat()
            except OSError:
                needs_copy = True
            else:
                needs_copy = (
                    source_stat.st_mtime_ns != target_stat.st_mtime_ns or source_stat.st_size != target_stat.st_size
                )
        if needs_copy:
            shutil.copy2(source_auth, target_auth)
    return home


def run_codex_planner(
    *,
    prompt: str,
    command: list[str],
    schema_path: Path,
    artifact_dir: Path,
    validation_feedback: str | None = None,
) -> CodexPlannerExecution:
    """Run headless Codex once, retrying a single time on invalid structured output."""
    attempts: list[CodexPlannerAttempt] = []
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    current_feedback = validation_feedback
    artifact_dir.mkdir(parents=True, exist_ok=True)
    last_message_path = artifact_dir / "codex_last_message.json"
    current_prompt = _render_attempt_prompt(prompt=prompt, validation_feedback=current_feedback)

    for attempt_number in range(1, 3):
        if last_message_path.exists():
            last_message_path.unlink()
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                [*command, "--output-schema", str(schema_path), "-o", str(last_message_path)],
                input=current_prompt,
                text=True,
                capture_output=True,
                cwd=_REPO_ROOT,
                check=False,
                env=_planner_env(),
            )
        except OSError as exc:  # pragma: no cover - subprocess creation failure is rare
            raise AgenticResearchCodexError("agentic_research_codex_exec_failed") from exc
        elapsed_seconds = time.perf_counter() - started
        stdout_chunks.append(completed.stdout)
        stderr_chunks.append(f"=== attempt {attempt_number} ===\n{completed.stderr}")
        parsed_lines = _parse_jsonl(completed.stdout)
        attempts.append(
            CodexPlannerAttempt(
                attempt_number=attempt_number,
                elapsed_seconds=elapsed_seconds,
                returncode=completed.returncode,
                thread_id=_find_thread_id(parsed_lines),
                input_tokens=_find_usage_value(parsed_lines, "input_tokens"),
                cached_input_tokens=_find_usage_value(parsed_lines, "cached_input_tokens"),
                output_tokens=_find_usage_value(parsed_lines, "output_tokens"),
                stdout_line_count=len([line for line in completed.stdout.splitlines() if line.strip()]),
                validation_feedback=current_feedback,
            )
        )
        if completed.returncode != 0:
            raise AgenticResearchCodexError(
                f"agentic_research_codex_exec_failed:{completed.returncode}:{completed.stderr.strip()}"
            )
        try:
            last_message = _load_last_message(last_message_path)
            decision = decision_from_dict(last_message)
            return CodexPlannerExecution(
                decision=decision,
                attempts=attempts,
                stdout_jsonl="".join(stdout_chunks),
                stderr_text="\n".join(stderr_chunks).strip(),
                last_message=last_message,
            )
        except Exception as exc:
            if attempt_number >= 2:
                raise AgenticResearchCodexError("agentic_research_codex_output_invalid") from exc
            current_feedback = _validation_feedback_from_error(exc)
            current_prompt = _render_attempt_prompt(prompt=prompt, validation_feedback=current_feedback)
    raise AgenticResearchCodexError("agentic_research_codex_output_invalid")


def _planner_env() -> dict[str, str]:
    env = dict(os.environ)
    env["CODEX_HOME"] = str(ensure_learner_codex_home())
    return env


def _render_attempt_prompt(*, prompt: str, validation_feedback: str | None) -> str:
    if validation_feedback is None:
        return prompt
    return f"{prompt}\n\nValidation feedback:\n{validation_feedback}\n"


def _load_last_message(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("agentic_research_codex_output_invalid_json") from exc
    if not isinstance(payload, dict):
        raise ValueError("agentic_research_codex_output_invalid_shape")
    return payload


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


def _validation_feedback_from_error(exc: Exception) -> str:
    if isinstance(exc, json.JSONDecodeError):
        return "Return strict JSON that matches the output schema."
    message = str(exc).strip()
    if message:
        return message
    return "Return valid planner fields only."
