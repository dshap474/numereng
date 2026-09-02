"""Loader for the repository-local active-model settings (`config/openrouter/active-model.py`)."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from numereng.platform.errors import OpenRouterClientError

_ACTIVE_MODEL_PATH = Path(__file__).resolve().parents[2] / "config" / "openrouter" / "active-model.py"
# The two headless CLI backends the research loop dispatches to. There is no HTTP transport.
OpenRouterModelSource = Literal["codex-exec", "droid-exec"]
MODEL_SOURCES = ("codex-exec", "droid-exec")
ModelReasoningEffort = Literal["low", "medium", "high", "xhigh"]


@dataclass(frozen=True)
class OpenRouterConfig:
    """Repository-local OpenRouter runtime settings."""

    active_model_source: OpenRouterModelSource
    active_model: str | None
    active_model_reasoning_effort: ModelReasoningEffort | None = None
    active_model_rotation: tuple[tuple[str, ModelReasoningEffort | None], ...] = ()

    def for_round(self, round_number: int | None) -> OpenRouterConfig:
        """Resolve the (model, effort) for one round when a rotation is configured.

        Identity when no rotation is set or the caller has no round number; the
        rotation cycles by round number so a resumed run stays deterministic.
        """
        if not self.active_model_rotation or round_number is None:
            return self
        model, effort = self.active_model_rotation[round_number % len(self.active_model_rotation)]
        return OpenRouterConfig(
            active_model_source=self.active_model_source,
            active_model=model,
            active_model_reasoning_effort=effort,
            active_model_rotation=self.active_model_rotation,
        )


def load_openrouter_config() -> OpenRouterConfig:
    """Load the repository-local OpenRouter model and compute-source settings."""
    if not _ACTIVE_MODEL_PATH.exists():
        return OpenRouterConfig(active_model_source="codex-exec", active_model=None)

    spec = importlib.util.spec_from_file_location("numereng_openrouter_active_model", _ACTIVE_MODEL_PATH)
    if spec is None or spec.loader is None:
        raise OpenRouterClientError("openrouter_config_load_failed")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001
        raise OpenRouterClientError("openrouter_config_load_failed") from exc

    source = getattr(module, "ACTIVE_MODEL_SOURCE", "codex-exec")
    if source not in MODEL_SOURCES:
        raise OpenRouterClientError(f"openrouter_active_model_source_invalid:{source}")

    active_model = getattr(module, "ACTIVE_MODEL", None)
    if active_model is not None and (not isinstance(active_model, str) or not active_model.strip()):
        raise OpenRouterClientError("openrouter_active_model_invalid")

    reasoning_effort = getattr(module, "ACTIVE_MODEL_REASONING_EFFORT", None)
    if reasoning_effort is not None:
        if not isinstance(reasoning_effort, str) or reasoning_effort not in {"low", "medium", "high", "xhigh"}:
            raise OpenRouterClientError("openrouter_active_model_reasoning_effort_invalid")

    return OpenRouterConfig(
        active_model_source=cast(OpenRouterModelSource, source),
        active_model=active_model.strip() if active_model is not None else None,
        active_model_reasoning_effort=cast(ModelReasoningEffort, reasoning_effort)
        if reasoning_effort is not None
        else None,
        active_model_rotation=_parse_model_rotation(getattr(module, "ACTIVE_MODEL_ROTATION", None)),
    )


def _parse_model_rotation(raw: object) -> tuple[tuple[str, ModelReasoningEffort | None], ...]:
    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)):
        raise OpenRouterClientError("openrouter_active_model_rotation_invalid")
    entries: list[tuple[str, ModelReasoningEffort | None]] = []
    for item in raw:
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 2
            or not isinstance(item[0], str)
            or not item[0].strip()
            or (item[1] is not None and item[1] not in {"low", "medium", "high", "xhigh"})
        ):
            raise OpenRouterClientError("openrouter_active_model_rotation_invalid")
        entries.append((item[0].strip(), cast("ModelReasoningEffort | None", item[1])))
    return tuple(entries)


def active_model_source() -> OpenRouterModelSource:
    """Return the configured planner/model compute source."""
    return load_openrouter_config().active_model_source
