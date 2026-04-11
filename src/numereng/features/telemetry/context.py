"""Launch metadata context for opt-in run telemetry."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass

from numereng.platform.run_execution import load_run_execution_from_env

_CURRENT_LAUNCH_METADATA: ContextVar[LaunchMetadata | None] = ContextVar(
    "numereng_launch_telemetry_metadata",
    default=None,
)


@dataclass(frozen=True, slots=True)
class LaunchMetadata:
    """Launch-level metadata used by training telemetry instrumentation."""

    source: str
    operation_type: str
    job_type: str
    execution: dict[str, object] | None = None


@contextmanager
def bind_launch_metadata(
    *,
    source: str,
    operation_type: str = "run",
    job_type: str = "run",
    execution: Mapping[str, object] | None = None,
) -> Iterator[None]:
    """Bind launch metadata for downstream training invocation in current context."""

    resolved_execution = dict(execution) if execution is not None else load_run_execution_from_env()
    token: Token[LaunchMetadata | None] = _CURRENT_LAUNCH_METADATA.set(
        LaunchMetadata(
            source=source,
            operation_type=operation_type,
            job_type=job_type,
            execution=resolved_execution,
        )
    )
    try:
        yield
    finally:
        _CURRENT_LAUNCH_METADATA.reset(token)


def get_launch_metadata() -> LaunchMetadata | None:
    """Return current launch metadata when telemetry is enabled for this context."""

    return _CURRENT_LAUNCH_METADATA.get()


__all__ = [
    "LaunchMetadata",
    "bind_launch_metadata",
    "get_launch_metadata",
]
