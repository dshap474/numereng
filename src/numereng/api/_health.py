"""Health and bootstrap API handlers."""

from __future__ import annotations

from numereng import __version__
from numereng.api.contracts import HealthResponse
from numereng.platform.errors import PackageError


class _BootstrapError(Exception):
    """Internal bootstrap error to exercise boundary translation."""


def get_health() -> HealthResponse:
    """Return package bootstrap health metadata."""
    return HealthResponse(version=__version__)


def run_bootstrap_check(*, fail: bool = False) -> HealthResponse:
    """Run a minimal API boundary check."""
    try:
        if fail:
            raise _BootstrapError("bootstrap_check_failed")
        return get_health()
    except _BootstrapError as exc:
        raise PackageError("bootstrap_check_failed") from exc


__all__ = ["get_health", "run_bootstrap_check"]
