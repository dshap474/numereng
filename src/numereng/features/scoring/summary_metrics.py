"""Shared scalar run-metric normalization for dashboard-aligned summaries."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any

SHARED_RUN_METRIC_NAMES: tuple[str, ...] = (
    "bmc_last_200_eras_mean",
    "bmc_mean",
    "corr_sharpe",
    "corr_mean",
    "mmc_mean",
    "cwmm_mean",
    "fnc_mean",
    "max_drawdown",
    "mmc_coverage_ratio_rows",
)

_SHARED_RUN_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "bmc_last_200_eras_mean": (
        "bmc_last_200_eras_mean",
        "bmc_ender20_last_200_eras.mean",
        "bmc_last_200_eras.mean",
    ),
    "bmc_mean": ("bmc_mean", "bmc_ender20.mean", "bmc.mean"),
    "corr_sharpe": ("corr_sharpe", "corr.sharpe", "corr20v2_sharpe", "sharpe"),
    "corr_mean": ("corr_mean", "corr.mean", "corr20v2_mean"),
    "mmc_mean": ("mmc_mean", "mmc_ender20.mean", "mmc.mean"),
    "cwmm_mean": ("cwmm_mean", "cwmm.mean"),
    "fnc_mean": ("fnc_mean", "fnc.mean"),
    "max_drawdown": ("max_drawdown", "corr.max_drawdown"),
}


def expand_shared_metric_query_names(metric_names: list[str]) -> list[str]:
    """Expand canonical shared metric names into persisted alias names."""

    expanded: list[str] = []
    seen: set[str] = set()
    for name in metric_names:
        aliases = _SHARED_RUN_METRIC_ALIASES.get(name, (name,))
        for alias in aliases:
            if alias in seen:
                continue
            seen.add(alias)
            expanded.append(alias)
    return expanded


def normalize_shared_run_metrics(
    metrics_payload: dict[str, Any],
    *,
    score_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize nested/raw metrics into a shared scalar summary payload."""

    flattened = _flatten_metrics(_sanitize_metrics(metrics_payload))
    normalized: dict[str, Any] = {}
    for metric_name in SHARED_RUN_METRIC_NAMES:
        if metric_name == "mmc_coverage_ratio_rows":
            ratio = _coverage_ratio_from_provenance(score_provenance)
            if ratio is not None:
                normalized[metric_name] = ratio
            continue
        value = _extract_numeric_metric(flattened, *_SHARED_RUN_METRIC_ALIASES[metric_name])
        if value is not None:
            normalized[metric_name] = value
    return normalized


def _sanitize_metric_value(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _sanitize_metrics(metrics: Any) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        return {}
    return {str(key): _sanitize_metric_value(value) for key, value in metrics.items()}


def _flatten_metrics(metrics: dict[str, Any], *, prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key_raw, value in metrics.items():
        key = str(key_raw)
        full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            flattened.update(_flatten_metrics(_sanitize_metrics(value), prefix=full_key))
            continue
        flattened[full_key] = _sanitize_metric_value(value)
    return flattened


def _extract_numeric_metric(metrics: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key not in metrics:
            continue
        value = _sanitize_metric_value(metrics.get(key))
        if isinstance(value, bool):
            continue
        if isinstance(value, Real):
            number = float(value)
            if math.isfinite(number):
                return number
    return None


def _coverage_ratio_from_provenance(score_provenance: dict[str, Any] | None) -> float | None:
    if not isinstance(score_provenance, dict):
        return None
    joins_raw = score_provenance.get("joins")
    joins = joins_raw if isinstance(joins_raw, dict) else {}
    predictions_rows = joins.get("predictions_rows")
    meta_overlap_rows = joins.get("meta_overlap_rows")
    if not isinstance(predictions_rows, Real) or isinstance(predictions_rows, bool):
        return None
    if not isinstance(meta_overlap_rows, Real) or isinstance(meta_overlap_rows, bool):
        return None
    total = float(predictions_rows)
    overlap = float(meta_overlap_rows)
    if total <= 0:
        return None
    ratio = overlap / total
    return ratio if math.isfinite(ratio) else None


__all__ = [
    "SHARED_RUN_METRIC_NAMES",
    "expand_shared_metric_query_names",
    "normalize_shared_run_metrics",
]
