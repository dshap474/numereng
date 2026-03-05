"""Launch metadata context for opt-in run telemetry."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass

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


@contextmanager
def bind_launch_metadata(
    *,
    source: str,
    operation_type: str = "run",
    job_type: str = "run",
) -> Iterator[None]:
    """Bind launch metadata for downstream training invocation in current context."""

    token: Token[LaunchMetadata | None] = _CURRENT_LAUNCH_METADATA.set(
        LaunchMetadata(source=source, operation_type=operation_type, job_type=job_type)
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
