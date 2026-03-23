"""Prompt template loading helpers for agentic research."""

from __future__ import annotations

import re
from pathlib import Path

_PROMPTS_ROOT = Path(__file__).resolve().parent / "prompts"
_VALIDATION_FEEDBACK_TEMPLATE = _PROMPTS_ROOT / "_partials" / "validation_feedback_block.md"
_PLACEHOLDER_RE = re.compile(r"\$([A-Z0-9_]+)")


def prompts_root() -> Path:
    """Return the root directory for prompt templates."""
    return _PROMPTS_ROOT


def render_prompt_template(path: Path, replacements: dict[str, str]) -> str:
    """Render one prompt template with direct placeholder substitution only."""
    template = path.read_text(encoding="utf-8")
    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace(f"${key}", value)
    unresolved = sorted({match.group(1) for match in _PLACEHOLDER_RE.finditer(rendered)})
    if unresolved:
        joined = ",".join(unresolved)
        raise ValueError(f"agentic_research_prompt_placeholders_unresolved:{path}:{joined}")
    return rendered.strip() + "\n"


def render_validation_feedback_block(validation_feedback: str | None) -> str:
    """Render the optional validation feedback block."""
    if validation_feedback is None or not validation_feedback.strip():
        return ""
    return render_prompt_template(
        _VALIDATION_FEEDBACK_TEMPLATE,
        {"VALIDATION_FEEDBACK": validation_feedback.strip()},
    ).strip()
