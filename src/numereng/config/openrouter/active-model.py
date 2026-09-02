"""
Set ACTIVE_MODEL_SOURCE to one of:
ACTIVE_MODEL_SOURCE=codex-exec|openrouter|droid-exec

ACTIVE_MODEL_ROTATION (optional) cycles research rounds through (model, effort)
pairs by round number; ACTIVE_MODEL/ACTIVE_MODEL_REASONING_EFFORT remain the
static fallback for non-round callers (e.g. closeout stages).
"""

ACTIVE_MODEL_SOURCE = "droid-exec"
ACTIVE_MODEL = "claude-opus-5"
ACTIVE_MODEL_REASONING_EFFORT = "high"
ACTIVE_MODEL_ROTATION = [
    ("claude-opus-5", "high"),
    ("gpt-5.6-sol", "high"),
    ("grok-4.6", "xhigh"),
]
