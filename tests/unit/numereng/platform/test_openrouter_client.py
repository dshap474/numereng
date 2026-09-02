"""Contract for the active-model config loader in `numereng.platform.clients.openrouter`.

Covers the two accepted compute sources, the optional round-rotation, and the stable error
tokens a malformed `active-model.py` raises.

USAGE:
    uv run pytest tests/unit/numereng/platform/test_openrouter_client.py -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

import numereng.platform.clients.openrouter as openrouter_module
from numereng.platform.clients.openrouter import (
    OpenRouterConfig,
    active_model_source,
    load_openrouter_config,
)
from numereng.platform.errors import OpenRouterClientError


def test_load_openrouter_config_reads_active_source_and_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "active-model.py"
    config_path.write_text(
        "\n".join(
            [
                'ACTIVE_MODEL_SOURCE = "codex-exec"',
                'ACTIVE_MODEL = "gpt-5.5"',
                'ACTIVE_MODEL_REASONING_EFFORT = "high"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(openrouter_module, "_ACTIVE_MODEL_PATH", config_path)

    assert load_openrouter_config() == OpenRouterConfig(
        active_model_source="codex-exec",
        active_model="gpt-5.5",
        active_model_reasoning_effort="high",
    )
    assert active_model_source() == "codex-exec"


def test_load_openrouter_config_accepts_droid_exec_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "active-model.py"
    config_path.write_text(
        'ACTIVE_MODEL_SOURCE = "droid-exec"\nACTIVE_MODEL = "claude-fable-5"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(openrouter_module, "_ACTIVE_MODEL_PATH", config_path)

    assert load_openrouter_config() == OpenRouterConfig(
        active_model_source="droid-exec",
        active_model="claude-fable-5",
        active_model_reasoning_effort=None,
    )
    assert active_model_source() == "droid-exec"


def test_load_openrouter_config_rejects_unknown_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the two headless CLI backends are accepted; the token names the rejected value."""
    config_path = tmp_path / "active-model.py"
    config_path.write_text('ACTIVE_MODEL_SOURCE = "openrouter"\n', encoding="utf-8")
    monkeypatch.setattr(openrouter_module, "_ACTIVE_MODEL_PATH", config_path)

    with pytest.raises(OpenRouterClientError, match="openrouter_active_model_source_invalid:openrouter"):
        load_openrouter_config()


def test_load_openrouter_config_reads_model_rotation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "active-model.py"
    config_path.write_text(
        "\n".join(
            [
                'ACTIVE_MODEL_SOURCE = "droid-exec"',
                'ACTIVE_MODEL = "claude-opus-5"',
                'ACTIVE_MODEL_REASONING_EFFORT = "medium"',
                "ACTIVE_MODEL_ROTATION = [",
                '    ("claude-opus-5", "medium"),',
                '    ("gpt-5.6-sol", "high"),',
                '    ("grok-4.6", "xhigh"),',
                "]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(openrouter_module, "_ACTIVE_MODEL_PATH", config_path)

    config = load_openrouter_config()
    assert config.active_model_rotation == (
        ("claude-opus-5", "medium"),
        ("gpt-5.6-sol", "high"),
        ("grok-4.6", "xhigh"),
    )
    # for_round cycles by round number and is identity without a round number.
    assert config.for_round(4).active_model == "gpt-5.6-sol"
    assert config.for_round(4).active_model_reasoning_effort == "high"
    assert config.for_round(None) is config


@pytest.mark.parametrize(
    "rotation_line",
    [
        'ACTIVE_MODEL_ROTATION = "claude-opus-5"',
        'ACTIVE_MODEL_ROTATION = [("claude-opus-5",)]',
        'ACTIVE_MODEL_ROTATION = [("", "high")]',
        'ACTIVE_MODEL_ROTATION = [("claude-opus-5", "extreme")]',
    ],
)
def test_load_openrouter_config_rejects_invalid_rotation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rotation_line: str,
) -> None:
    config_path = tmp_path / "active-model.py"
    config_path.write_text(
        f'ACTIVE_MODEL_SOURCE = "droid-exec"\n{rotation_line}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(openrouter_module, "_ACTIVE_MODEL_PATH", config_path)

    with pytest.raises(OpenRouterClientError, match="openrouter_active_model_rotation_invalid"):
        load_openrouter_config()
