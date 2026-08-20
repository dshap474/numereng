"""Read-only loader for the human-maintained portfolio registry (v1).

Missing file -> empty portfolio (returns None). Malformed JSON or schema
violations raise a clear feature error. v1 never writes the registry.

USAGE:
    from numereng.features.research_portfolio.registry import load_registry
    registry = load_registry(store_root=".numereng")   # None when absent
"""

from __future__ import annotations

import json
from pathlib import Path

from numereng.config.research_portfolio import (
    RegistryConfig,
    RegistryConfigError,
    load_registry_config,
)
from numereng.features.research_portfolio.types import PortfolioValidationError
from numereng.features.store import resolve_portfolio_registry_path

# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #


def registry_path(*, store_root: str | Path = ".numereng") -> Path:
    """Return the canonical registry.json path for one store root."""

    return resolve_portfolio_registry_path(store_root=store_root)


def load_registry(*, store_root: str | Path = ".numereng") -> RegistryConfig | None:
    """Load and validate registry.json; return None when the file is absent."""

    path = registry_path(store_root=store_root)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PortfolioValidationError(f"registry_read_failed:{path}") from exc
    try:
        return load_registry_config(payload)
    except RegistryConfigError as exc:
        raise PortfolioValidationError(str(exc)) from exc


__all__ = ["load_registry", "registry_path"]
