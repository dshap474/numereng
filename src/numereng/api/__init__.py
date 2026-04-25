"""Stable public API surface for numereng consumers."""

from __future__ import annotations

from importlib import import_module

from numereng.platform.errors import PackageError

from . import contracts as _contracts
from ._lazy_exports import (
    LAZY_EXPORTS,
    RESEARCH_API_EXPORTS,
    RESEARCH_FEATURE_EXPORTS,
    ROOT_CONTRACT_EXPORT_EXCLUDES,
)

# === Contract re-exports - keep request/response models eager and explicit ===
CONTRACT_EXPORTS = tuple(name for name in _contracts.__all__ if name not in ROOT_CONTRACT_EXPORT_EXCLUDES)
for contract_name in CONTRACT_EXPORTS:
    globals()[contract_name] = getattr(_contracts, contract_name)


def __getattr__(name: str):
    target = LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing and not missing.startswith("numereng"):
            raise PackageError(f"runtime_dependency_missing:{missing}:run_uv_sync_extra_dev") from exc
        raise
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def _agentic_research_api_module():
    return import_module("numereng.api._agentic_research")


def _agentic_research_feature_module():
    return import_module("numereng.features.agentic_research")


def research_run(*args, **kwargs):
    return _agentic_research_api_module().research_run(*args, **kwargs)


def research_status(*args, **kwargs):
    return _agentic_research_api_module().research_status(*args, **kwargs)


def get_research_status(*args, **kwargs):
    return _agentic_research_feature_module().get_research_status(*args, **kwargs)


def run_research(*args, **kwargs):
    return _agentic_research_feature_module().run_research(*args, **kwargs)


__all__ = list(
    dict.fromkeys(
        [
            *CONTRACT_EXPORTS,
            "PackageError",
            *LAZY_EXPORTS,
            *RESEARCH_API_EXPORTS,
            *RESEARCH_FEATURE_EXPORTS,
        ]
    )
)
