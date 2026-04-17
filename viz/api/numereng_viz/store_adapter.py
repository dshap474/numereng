"""Store adapter for the viz API.

The adapter isolates all access to `.numereng` filesystem and SQLite state so the
HTTP layer is path-agnostic and easy to rewire when store layout evolves.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from numbers import Integral, Real
from pathlib import Path, PureWindowsPath
from typing import Any
from urllib.parse import urlsplit

import pandas as pd
import yaml

from numereng.assets import docs_assets_root, docs_root
from numereng.features.scoring.summary_metrics import (
    expand_shared_metric_query_names,
    normalize_shared_run_metrics,
)
from numereng.features.store import resolve_workspace_layout, resolve_workspace_layout_from_store_root

logger = logging.getLogger(__name__)

_SAFE_ID = re.compile(r"^[\w\-.]+$")
_ROUND_CONFIG_STEM_RE = re.compile(r"^(r\d+)_\d+_(.+)$")
_ROUND_INDEX_RE = re.compile(r"^r(\d+)_")
_FORUM_INDEX_RELATIVE_PATH = "forum/INDEX.md"
_FORUM_POSTS_RELATIVE_PATH = "forum/posts"

_TERMINAL_JOB_STATUSES = {"completed", "failed", "canceled", "stale"}
_ACTIVE_JOB_STATUSES = {"queued", "starting", "running", "canceling"}
_HIDDEN_NOTE_STEMS = {"CLAUDE", "AGENTS"}
_EXPERIMENT_ARCHIVE_DIRNAME = "_archive"
_ATTENTION_STATUS_RANK: dict[str, int] = {
    "none": 0,
    "canceled": 1,
    "stale": 2,
    "failed": 3,
}
_METRIC_QUERY_ALIASES: dict[str, tuple[str, ...]] = {
    "corr_mean": ("corr_mean", "corr.mean"),
    "corr_payout_mean": ("corr_payout_mean", "corr_ender20.mean", "corr_ender20"),
    "corr_std": ("corr_std", "corr.std"),
    "corr_sharpe": ("corr_sharpe", "corr.sharpe", "sharpe"),
    "corr_n_eras": ("corr_n_eras", "corr.n_eras", "n_eras"),
    "fnc_mean": ("fnc_mean", "fnc.mean"),
    "fnc_std": ("fnc_std", "fnc.std"),
    "fnc_sharpe": ("fnc_sharpe", "fnc.sharpe"),
    "mmc_mean": ("mmc_mean", "mmc_ender20.mean", "mmc.mean"),
    "mmc_payout_mean": ("mmc_payout_mean", "mmc.mean", "mmc_ender20.mean", "mmc_ender20"),
    "mmc_std": ("mmc_std", "mmc_ender20.std", "mmc.std"),
    "mmc_sharpe": ("mmc_sharpe", "mmc_ender20.sharpe", "mmc.sharpe"),
    "mmc_n_eras": ("mmc_n_eras", "mmc_ender20.n_eras", "mmc.n_eras"),
    "bmc_mean": ("bmc_mean", "bmc_ender20.mean", "bmc.mean"),
    "bmc_std": ("bmc_std", "bmc_ender20.std", "bmc.std"),
    "bmc_sharpe": ("bmc_sharpe", "bmc_ender20.sharpe", "bmc.sharpe"),
    "bmc_n_eras": ("bmc_n_eras", "bmc_ender20.n_eras", "bmc.n_eras"),
    "bmc_last_200_eras_mean": (
        "bmc_last_200_eras_mean",
        "bmc_ender20_last_200_eras.mean",
        "bmc_last_200_eras.mean",
    ),
    "cwmm_mean": ("cwmm_mean", "cwmm.mean"),
    "cwmm_std": ("cwmm_std", "cwmm.std"),
    "cwmm_sharpe": ("cwmm_sharpe", "cwmm.sharpe"),
    "max_drawdown": ("max_drawdown", "corr.max_drawdown"),
}
_LEGACY_SCORING_SERIES_SPECS: tuple[tuple[str, str, str], ...] = (
    ("corr_per_era.parquet", "corr_native", "per_era"),
    ("corr_cumulative.parquet", "corr_native", "cumulative"),
    ("corr_ender20_per_era.parquet", "corr_ender20", "per_era"),
    ("corr_ender20_cumulative.parquet", "corr_ender20", "cumulative"),
    ("bmc_per_era.parquet", "bmc_native", "per_era"),
    ("bmc_cumulative.parquet", "bmc_native", "cumulative"),
    ("bmc_ender20_per_era.parquet", "bmc_ender20", "per_era"),
    ("bmc_ender20_cumulative.parquet", "bmc_ender20", "cumulative"),
    ("corr_with_benchmark_per_era.parquet", "corr_with_benchmark", "per_era"),
    ("corr_with_benchmark_cumulative.parquet", "corr_with_benchmark", "cumulative"),
    ("baseline_corr_per_era.parquet", "baseline_corr_native", "per_era"),
    ("baseline_corr_cumulative.parquet", "baseline_corr_native", "cumulative"),
    ("baseline_corr_ender20_per_era.parquet", "baseline_corr_ender20", "per_era"),
    ("baseline_corr_ender20_cumulative.parquet", "baseline_corr_ender20", "cumulative"),
    ("corr_delta_vs_baseline_per_era.parquet", "corr_delta_vs_baseline_native", "per_era"),
    ("corr_delta_vs_baseline_cumulative.parquet", "corr_delta_vs_baseline_native", "cumulative"),
    ("corr_delta_vs_baseline_ender20_per_era.parquet", "corr_delta_vs_baseline_ender20", "per_era"),
    ("corr_delta_vs_baseline_ender20_cumulative.parquet", "corr_delta_vs_baseline_ender20", "cumulative"),
)

_CANONICAL_DASHBOARD_METRIC_KEY_ALIASES: dict[str, str] = {
    "bmc_ender20": "bmc",
    "mmc_ender20": "mmc",
    "corr_delta_vs_baseline_ender20": "corr_delta_vs_baseline",
}
_HIDDEN_DASHBOARD_METRIC_PREFIXES: tuple[str, ...] = (
    "baseline_corr",
    "bmc_",
    "mmc_",
    "corr_delta_vs_baseline_",
)
_REMOTE_PULLS_RELATIVE_DIR = ("cache", "remote_ops", "pulls")


@lru_cache(maxsize=256)
def _read_parquet_records_cached(path_str: str, mtime_ns: int) -> list[dict[str, Any]]:
    """Read parquet records from mtime-keyed cache."""

    _ = mtime_ns
    frame = pd.read_parquet(path_str)
    return _frame_to_records(frame)


@lru_cache(maxsize=32)
def _read_parquet_frame_cached(
    path_str: str,
    mtime_ns: int,
    columns: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Read parquet frame with optional column subset from mtime-keyed cache."""

    _ = mtime_ns
    return pd.read_parquet(path_str, columns=list(columns) if columns else None)


@lru_cache(maxsize=128)
def _read_parquet_matrix_cached(path_str: str, mtime_ns: int) -> tuple[list[str], list[list[float | None]]]:
    """Read correlation matrix parquet into labels + matrix payload."""

    _ = mtime_ns
    frame = pd.read_parquet(path_str)
    labels = [str(value) for value in frame.columns.tolist()]
    matrix: list[list[float | None]] = []
    for row in frame.values.tolist():
        normalized_row = []
        for value in row:
            normalized_row.append(_sanitize_metric_value(value))
        matrix.append(normalized_row)
    return labels, matrix


def _frame_to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Normalize pandas records into JSON-serializable dicts."""

    records = frame.to_dict(orient="records")
    normalized: list[dict[str, Any]] = []
    for item in records:
        row: dict[str, Any] = {}
        for key, value in item.items():
            row[str(key)] = _coerce_json_value(value)
        normalized.append(row)
    return normalized


def _coerce_json_value(value: Any) -> Any:
    """Convert pandas/scalar values into JSON-safe values."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else None
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    return value


def _to_non_empty_str(value: Any) -> str | None:
    """Normalize non-empty string values."""

    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned if cleaned else None


def _normalize_dashboard_metric_key(metric_key: str) -> str | None:
    canonical = _CANONICAL_DASHBOARD_METRIC_KEY_ALIASES.get(metric_key)
    if canonical is not None:
        return canonical
    if any(metric_key.startswith(prefix) for prefix in _HIDDEN_DASHBOARD_METRIC_PREFIXES):
        return None
    return metric_key


def _normalize_scoring_dashboard_series(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_metric_keys = {str(row.get("metric_key")) for row in rows if isinstance(row.get("metric_key"), str)}
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        raw_metric_key = row.get("metric_key")
        if not isinstance(raw_metric_key, str):
            continue
        metric_key = _normalize_dashboard_metric_key(raw_metric_key)
        if metric_key is None:
            continue
        if raw_metric_key != metric_key and metric_key in raw_metric_keys:
            continue
        normalized_row = dict(row)
        normalized_row["metric_key"] = metric_key
        dedupe_key = (
            metric_key,
            str(normalized_row.get("series_type", "")),
            str(normalized_row.get("era", "")),
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        normalized.append(normalized_row)
    return normalized


def _normalize_scoring_dashboard_summary_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    normalized = dict(row)
    legacy_prefixes: dict[str, tuple[str, ...]] = {
        "bmc": ("bmc_ender20",),
        "mmc": ("mmc_ender20",),
        "corr_delta_vs_baseline": ("corr_delta_vs_baseline_ender20",),
        "bmc_last_200_eras": ("bmc_last_200_eras_ender20", "bmc_ender20_last_200_eras"),
    }
    for metric_name, prefixes in legacy_prefixes.items():
        for suffix in ("mean", "std", "sharpe", "max_drawdown"):
            canonical_key = f"{metric_name}_{suffix}"
            if canonical_key in normalized:
                continue
            for legacy_prefix in prefixes:
                legacy_key = f"{legacy_prefix}_{suffix}"
                if legacy_key in normalized:
                    normalized[canonical_key] = normalized[legacy_key]
                    break
    return normalized


def _normalize_scoring_dashboard_fold_rows(
    rows: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    if rows is None:
        return None
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        normalized = dict(row)
        if "bmc_fold_mean" not in normalized and "bmc_ender20_fold_mean" in normalized:
            normalized["bmc_fold_mean"] = normalized["bmc_ender20_fold_mean"]
        normalized_rows.append(normalized)
    return normalized_rows


def _parse_json_object(value: Any) -> dict[str, Any]:
    """Parse JSON object payloads defensively."""

    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_json_array(value: Any) -> list[Any]:
    """Parse JSON list payloads defensively."""

    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


def _parse_json_list(value: Any) -> list[Any]:
    """Backward-compatible alias for JSON list parsing."""

    return _parse_json_array(value)


def _sanitize_metric_value(value: Any) -> Any:
    """Normalize metrics values for JSON response payloads."""

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _sanitize_metrics(metrics: Any) -> dict[str, Any]:
    """Normalize metric dict payloads."""

    if not isinstance(metrics, dict):
        return {}
    return {str(key): _sanitize_metric_value(value) for key, value in metrics.items()}


def _read_json_dict(path: Path) -> dict[str, Any]:
    """Read JSON file as dict, returning empty dict when invalid."""

    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _first_present_str(payload: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    """Return first present non-empty string value from dict for keys."""

    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _resolve_run_targets(manifest: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    """Resolve canonical run target fields from manifest payload."""

    data_raw = manifest.get("data")
    data_info = data_raw if isinstance(data_raw, dict) else {}
    target_train = _first_present_str(data_info, ("target_train", "target_col", "target")) or _first_present_str(
        manifest, ("target_col", "target_train", "target")
    )
    target_payout = _first_present_str(data_info, ("target_payout",))
    target = target_payout or target_train
    return target_train, target_payout, target


def _extract_run_seed(payload: dict[str, Any]) -> int | None:
    """Resolve canonical run seed from resolved/manifest payload."""

    model_raw = payload.get("model")
    model = model_raw if isinstance(model_raw, dict) else {}
    params_raw = model.get("params")
    params = params_raw if isinstance(params_raw, dict) else {}
    for candidate in (params.get("random_state"), params.get("seed"), model.get("seed")):
        if isinstance(candidate, bool):
            continue
        if isinstance(candidate, Integral):
            return int(candidate)
        if isinstance(candidate, Real):
            number = float(candidate)
            if math.isfinite(number):
                return int(number)
    return None


@lru_cache(maxsize=1)
def _load_scoring_functions() -> tuple[Callable[..., pd.Series] | None, Callable[..., pd.Series] | None]:
    """Load Numerai scoring functions lazily."""

    try:
        from numerai_tools.scoring import (
            correlation_contribution,
            numerai_corr,
        )
    except Exception:
        logger.debug("numerai_tools unavailable; per-era fallback derivation disabled", exc_info=True)
        return None, None
    return correlation_contribution, numerai_corr


def _metric_row_value(value: Any, value_json: Any) -> Any:
    """Decode metric table row value, supporting scalar + serialized JSON payloads."""

    scalar = _sanitize_metric_value(value)
    if scalar is not None:
        return scalar

    if not isinstance(value_json, str) or not value_json.strip():
        return None
    try:
        parsed = json.loads(value_json)
    except json.JSONDecodeError:
        return None
    return _coerce_json_value(parsed)


def _flatten_metrics(metrics: dict[str, Any], *, prefix: str = "") -> dict[str, Any]:
    """Flatten nested metric payloads into dotted keys for canonical lookup."""

    flattened: dict[str, Any] = {}
    for key_raw, value in metrics.items():
        key = str(key_raw)
        full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            flattened.update(_flatten_metrics(_sanitize_metrics(value), prefix=full_key))
            continue
        flattened[full_key] = _sanitize_metric_value(value)
    return flattened


def _expand_metric_query_names(metric_names: list[str]) -> list[str]:
    """Expand canonical metric names into all known persisted aliases."""

    expanded: list[str] = []
    seen: set[str] = set()
    for name in metric_names:
        aliases = _METRIC_QUERY_ALIASES.get(name)
        if aliases is None:
            aliases = tuple(expand_shared_metric_query_names([name])) or (name,)
        for alias in aliases:
            if alias in seen:
                continue
            seen.add(alias)
            expanded.append(alias)
    return expanded


def _iso_from_ts(ts: float) -> str:
    """UTC ISO-8601 timestamp from unix seconds."""

    return datetime.fromtimestamp(ts, tz=UTC).isoformat()


def _ensure_safe_id(value: str, *, label: str) -> str:
    """Validate an identifier used in route params."""

    if not _SAFE_ID.match(value):
        raise ValueError(f"Invalid {label}: {value}")
    return value


def _normalize_relative_path(path: str, *, label: str) -> str:
    """Normalize and validate a safe repository-relative path."""

    raw = path.strip().replace("\\", "/")
    if not raw:
        raise ValueError(f"Invalid {label}: empty path")
    if raw.startswith("/"):
        raise ValueError(f"Invalid {label}: absolute path not allowed")
    if "\x00" in raw:
        raise ValueError(f"Invalid {label}: contains null byte")

    segments: list[str] = []
    for segment in raw.split("/"):
        if segment in {"", "."}:
            continue
        if segment == "..":
            raise ValueError(f"Invalid {label}: path traversal not allowed")
        segments.append(segment)

    if not segments:
        raise ValueError(f"Invalid {label}: empty path")
    return "/".join(segments)


def _normalize_markdown_doc_path(path: str) -> str:
    """Normalize docs content path and require markdown extension."""

    normalized = _normalize_relative_path(path, label="doc path")
    if not normalized.lower().endswith(".md"):
        raise ValueError(f"Invalid doc path: {path}")
    return normalized


def _resolve_summary_href(base_dir: str, href: str) -> str | None:
    """Resolve SUMMARY.md href into normalized repo-relative markdown path."""

    raw = href.strip()
    if not raw:
        return None
    if raw.startswith("<") and raw.endswith(">"):
        raw = raw[1:-1].strip()
        if not raw:
            return None

    parsed = urlsplit(raw)
    if parsed.scheme or parsed.netloc:
        return None

    href_path = parsed.path.strip().replace("\\", "/")
    if not href_path:
        return None

    if href_path.startswith("/"):
        candidate = href_path.lstrip("/")
    else:
        candidate = f"{base_dir}/{href_path}" if base_dir else href_path

    stack: list[str] = []
    for segment in candidate.split("/"):
        if segment in {"", "."}:
            continue
        if segment == "..":
            if not stack:
                return None
            stack.pop()
            continue
        stack.append(segment)

    if not stack:
        return None
    return "/".join(stack)


def _round_key_from_config_stem(stem: str) -> str:
    match = _ROUND_CONFIG_STEM_RE.match(stem)
    if not match:
        return stem
    return f"{match.group(1)}_{match.group(2)}"


def _round_index_from_key(round_key: str) -> int | None:
    match = _ROUND_INDEX_RE.match(round_key)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _infer_round_model_type(round_key: str, model_type: str | None) -> str:
    if model_type:
        return model_type
    if "_neut_" in round_key:
        return "neutralized"
    if "core_plus" in round_key or "_blend_" in round_key:
        return "blend"
    return "derived"


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


def _normalize_round_metrics(
    metrics: dict[str, Any],
    *,
    target_col: str | None = None,
) -> dict[str, Any]:
    _ = target_col
    flattened = _flatten_metrics(_sanitize_metrics(metrics))
    normalized = normalize_shared_run_metrics(metrics)

    scalar_metrics: tuple[tuple[str, tuple[str, ...], bool], ...] = (
        ("corr_payout_mean", ("corr_ender20.mean", "corr_ender20", "corr_payout_mean"), False),
        (
            "corr_with_benchmark",
            (
                "corr_with_benchmark",
                "avg_corr_with_benchmark",
                "bmc.avg_corr_with_benchmark",
                "bmc_last_200_eras.avg_corr_with_benchmark",
            ),
            False,
        ),
        ("corr_std", ("corr.std", "corr_std", "corr20v2_std"), False),
        ("corr_n_eras", ("corr.n_eras", "corr_n_eras", "corr20v2_n_eras", "n_eras"), True),
        ("fnc_std", ("fnc.std", "fnc_std"), False),
        ("fnc_sharpe", ("fnc.sharpe", "fnc_sharpe"), False),
        ("mmc_std", ("mmc_ender20.std", "mmc.std", "mmc_std"), False),
        ("mmc_sharpe", ("mmc_ender20.sharpe", "mmc.sharpe", "mmc_sharpe"), False),
        ("mmc_n_eras", ("mmc_ender20.n_eras", "mmc.n_eras", "mmc_n_eras"), True),
        ("bmc_std", ("bmc_ender20.std", "bmc.std", "bmc_std"), False),
        ("bmc_sharpe", ("bmc_ender20.sharpe", "bmc.sharpe", "bmc_sharpe"), False),
        ("bmc_n_eras", ("bmc_ender20.n_eras", "bmc.n_eras", "bmc_n_eras"), True),
        ("cwmm_std", ("cwmm.std", "cwmm_std"), False),
        ("cwmm_sharpe", ("cwmm.sharpe", "cwmm_sharpe"), False),
    )

    for output_key, aliases, cast_int in scalar_metrics:
        value = _extract_numeric_metric(flattened, *aliases)
        if value is None:
            continue
        normalized[output_key] = int(value) if cast_int else value

    mmc_payout_mean = normalized.get("mmc_mean")
    if isinstance(mmc_payout_mean, Real) and not isinstance(mmc_payout_mean, bool):
        normalized["mmc_payout_mean"] = float(mmc_payout_mean)
    else:
        value = _extract_numeric_metric(flattened, "mmc_payout_mean", "mmc_ender20.mean", "mmc_ender20")
        if value is not None:
            normalized["mmc_payout_mean"] = value

    return normalized


def _coverage_ratio_from_provenance(score_provenance: dict[str, Any]) -> float | None:
    """Derive MMC coverage ratio from score provenance joins payload."""

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


def _normalize_per_era_corr_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize persisted per-era correlation rows to canonical field names."""

    normalized: list[dict[str, Any]] = []
    for row in records:
        corr = _extract_numeric_metric(row, "corr", "corr20v2")
        if corr is None:
            continue
        normalized.append(
            {
                "era": row.get("era"),
                "corr": corr,
            }
        )
    return _sort_records_by_era(normalized)


def _resolve_series_value(value: Any, preferred_key: str | None = None) -> float | None:
    """Resolve a scalar float from scorer outputs (Series/scalar)."""

    if isinstance(value, pd.Series):
        if preferred_key and preferred_key in value.index:
            candidate = _sanitize_metric_value(value[preferred_key])
            if isinstance(candidate, Real) and not isinstance(candidate, bool):
                number = float(candidate)
                if math.isfinite(number):
                    return number
        for candidate in value.to_list():
            scalar = _sanitize_metric_value(candidate)
            if isinstance(scalar, Real) and not isinstance(scalar, bool):
                number = float(scalar)
                if math.isfinite(number):
                    return number
        return None

    scalar = _sanitize_metric_value(value)
    if isinstance(scalar, Real) and not isinstance(scalar, bool):
        number = float(scalar)
        return number if math.isfinite(number) else None
    return None


def _era_sort_key(value: Any) -> tuple[int, float, str]:
    """Stable sort key for era-like identifiers."""

    if isinstance(value, bool):
        return (2, 0.0, str(value))
    if isinstance(value, Integral):
        return (0, float(int(value)), str(int(value)))
    if isinstance(value, Real):
        numeric = float(value)
        if math.isfinite(numeric):
            return (0, numeric, str(numeric))
        return (2, 0.0, str(value))
    text = str(value)
    try:
        return (0, float(text), text)
    except Exception:
        return (1, 0.0, text)


def _sort_records_by_era(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort per-era record payload by era value."""

    return sorted(records, key=lambda row: _era_sort_key(row.get("era")))


def repository_root() -> Path:
    """Return repository root (`numereng/`)."""

    return Path(__file__).resolve().parents[3]


def resolve_store_root(
    explicit: str | Path | None = None,
    *,
    workspace_root: str | Path | None = None,
) -> Path:
    """Resolve store root using explicit value, env, then repo default."""

    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    if workspace_root is not None:
        return resolve_workspace_layout(workspace_root).store_root
    env_workspace_root = os.getenv("NUMERENG_WORKSPACE_ROOT")
    if env_workspace_root:
        return resolve_workspace_layout(env_workspace_root).store_root
    env_root = os.getenv("NUMERENG_STORE_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return (repository_root() / ".numereng").resolve()


@dataclass(frozen=True)
class VizStoreConfig:
    """Configuration for viz store adapter."""

    store_root: Path
    workspace_root: Path | None = None
    repo_root: Path | None = None
    numerai_docs_root: Path | None = None
    numereng_docs_root: Path | None = None
    shared_docs_assets_root: Path | None = None

    def __post_init__(self) -> None:
        store_root = self.store_root.expanduser().resolve()
        workspace_root = (
            self.workspace_root.expanduser().resolve()
            if self.workspace_root is not None
            else resolve_workspace_layout_from_store_root(store_root).workspace_root
        )
        repo_root = self.repo_root.expanduser().resolve() if self.repo_root is not None else repository_root()
        repo_docs_root = repo_root / "docs"
        workspace_docs_root = workspace_root / "docs"
        numerai_root = (
            self.numerai_docs_root.expanduser().resolve()
            if self.numerai_docs_root is not None
            else (
                workspace_docs_root / "numerai"
                if (workspace_docs_root / "numerai").is_dir()
                else (repo_docs_root / "numerai" if (repo_docs_root / "numerai").is_dir() else docs_root("numerai"))
            )
        )
        numereng_root = (
            self.numereng_docs_root.expanduser().resolve()
            if self.numereng_docs_root is not None
            else (repo_docs_root / "numereng" if (repo_docs_root / "numereng").is_dir() else docs_root("numereng"))
        )
        shared_assets_root = (
            self.shared_docs_assets_root.expanduser().resolve()
            if self.shared_docs_assets_root is not None
            else (repo_docs_root / "assets" if (repo_docs_root / "assets").is_dir() else docs_assets_root())
        )
        object.__setattr__(self, "store_root", store_root)
        object.__setattr__(self, "workspace_root", workspace_root)
        object.__setattr__(self, "repo_root", repo_root)
        object.__setattr__(self, "numerai_docs_root", numerai_root)
        object.__setattr__(self, "numereng_docs_root", numereng_root)
        object.__setattr__(self, "shared_docs_assets_root", shared_assets_root)

    @classmethod
    def from_env(
        cls,
        *,
        workspace_root: str | Path | None = None,
        store_root: str | Path | None = None,
    ) -> VizStoreConfig:
        return cls(
            store_root=resolve_store_root(store_root, workspace_root=workspace_root),
            workspace_root=Path(workspace_root).expanduser().resolve() if workspace_root is not None else None,
        )


@dataclass(frozen=True)
class PerEraCorrLoadResult:
    """Per-era corr payload plus lightweight timing metadata for viz routes."""

    payload: list[dict[str, Any]] | None
    persisted_read_ms: float
    materialize_ms: float
    wrote_artifact: bool


class VizStoreAdapter:
    """Read-oriented adapter over `.numereng` SQLite + artifact filesystem."""

    def __init__(self, config: VizStoreConfig) -> None:
        self.config = config
        self.store_root = config.store_root
        self.repo_root = config.repo_root
        self.db_path = self.store_root / "numereng.db"
        self._lock = threading.RLock()
        self._conn = self._connect_read_only()
        self._table_exists_cache: dict[str, bool] = {}
        self._resolved_run_dir_cache: dict[str, Path] = {}

    def _connect_read_only(self) -> sqlite3.Connection | None:
        if not self.db_path.exists():
            return None
        try:
            conn = sqlite3.connect(
                f"file:{self.db_path}?mode=ro",
                uri=True,
                check_same_thread=False,
                timeout=2.0,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout=2000")
            return conn
        except Exception as exc:
            logger.warning("Unable to open store DB %s: %s", self.db_path, exc)
            return None

    def _query(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        if self._conn is None:
            return []
        with self._lock:
            try:
                return self._conn.execute(sql, params).fetchall()
            except sqlite3.Error as exc:
                logger.debug("SQLite query failed: %s (%s)", exc, sql)
                return []

    def _query_one(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        rows = self._query(sql, params)
        return rows[0] if rows else None

    def _table_exists(self, name: str) -> bool:
        cached = self._table_exists_cache.get(name)
        if cached is not None:
            return cached
        row = self._query_one(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
            (name,),
        )
        exists = row is not None
        self._table_exists_cache[name] = exists
        return exists

    def _in_store(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self.store_root.resolve())
            return True
        except ValueError:
            return False

    def _sha256(self, path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    def _safe_read_yaml(self, path: Path) -> dict[str, Any]:
        try:
            with open(path) as handle:
                payload = yaml.safe_load(handle)
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _config_summary(self, payload: dict[str, Any]) -> dict[str, Any]:
        compute_raw = payload.get("compute")
        compute: dict[str, Any] = compute_raw if isinstance(compute_raw, dict) else {}
        model_raw = payload.get("model")
        model: dict[str, Any] = model_raw if isinstance(model_raw, dict) else {}
        data_raw = payload.get("data")
        data_cfg: dict[str, Any] = data_raw if isinstance(data_raw, dict) else {}
        stage_list = payload.get("stages")
        stages = stage_list if isinstance(stage_list, list) else None
        return {
            "run_id": payload.get("run_id"),
            "backend": compute.get("backend"),
            "tier": compute.get("tier"),
            "model_type": model.get("type"),
            "target": data_cfg.get("target") or data_cfg.get("target_train") or data_cfg.get("target_col"),
            "target_payout": data_cfg.get("target_payout"),
            "feature_set": data_cfg.get("feature_set"),
            "stages": stages,
        }

    def _runnable_status(self, payload: dict[str, Any]) -> tuple[bool, str | None]:
        if not payload:
            return False, "Config YAML must be a mapping"
        stages = payload.get("stages")
        if stages is not None and not isinstance(stages, list):
            return False, "Config 'stages' must be a list"
        return True, None

    def _resolve_config_id(self, config_id: str) -> Path:
        if config_id.startswith("store:"):
            config_id = config_id[len("store:") :]
        elif config_id.startswith("repo:"):
            raise ValueError("repo: config IDs are not supported")

        candidate = Path(config_id)
        if not config_id or candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("Invalid config path")

        path = (self.store_root / candidate).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {config_id}")
        if path.suffix.lower() not in {".json", ".yaml", ".yml"}:
            raise ValueError("Config must be JSON or YAML")
        if not self._in_store(path):
            raise ValueError("Config path escapes store root")
        return path

    def _experiments_root(self) -> Path:
        return resolve_workspace_layout_from_store_root(self.store_root).experiments_root

    def _workspace_root(self) -> Path:
        return resolve_workspace_layout_from_store_root(self.store_root).workspace_root

    def _legacy_experiments_root(self) -> Path:
        return self._workspace_root() / "experiments"

    def _archived_experiments_root(self) -> Path:
        return self._experiments_root() / _EXPERIMENT_ARCHIVE_DIRNAME

    def _legacy_archived_experiments_root(self) -> Path:
        return self._legacy_experiments_root() / _EXPERIMENT_ARCHIVE_DIRNAME

    def _remote_pulls_root(self) -> Path:
        return (
            self.store_root
            / _REMOTE_PULLS_RELATIVE_DIR[0]
            / _REMOTE_PULLS_RELATIVE_DIR[1]
            / _REMOTE_PULLS_RELATIVE_DIR[2]
        )

    def _iter_remote_pull_target_roots(self) -> list[Path]:
        root = self._remote_pulls_root()
        if not root.is_dir():
            return []
        return sorted([child for child in root.iterdir() if child.is_dir()], key=lambda path: path.name)

    def _manifest_run_ids(self, experiment_id: str) -> list[str]:
        manifest = _read_json_dict(self._resolve_experiment_dir(experiment_id) / "experiment.json")
        runs = manifest.get("runs")
        if not isinstance(runs, list):
            return []
        payload: list[str] = []
        seen: set[str] = set()
        for item in runs:
            if not isinstance(item, str):
                continue
            run_id = item.strip()
            if not run_id or run_id in seen:
                continue
            seen.add(run_id)
            payload.append(run_id)
        return payload

    def _resolve_pulled_run_dir(self, run_id: str) -> Path | None:
        for target_root in self._iter_remote_pull_target_roots():
            candidate = target_root / "runs" / run_id
            if (candidate / "run.json").is_file():
                return candidate
        return None

    def _experiment_path_candidates(self, experiment_id: str) -> tuple[Path, ...]:
        return (
            self._experiments_root() / experiment_id,
            self._archived_experiments_root() / experiment_id,
            self._legacy_experiments_root() / experiment_id,
            self._legacy_archived_experiments_root() / experiment_id,
        )

    def _resolve_experiment_dir(self, experiment_id: str) -> Path:
        _ensure_safe_id(experiment_id, label="experiment_id")
        candidates = self._experiment_path_candidates(experiment_id)
        existing = [path for path in candidates if (path / "experiment.json").is_file()]
        live_existing = [path for path in existing if _EXPERIMENT_ARCHIVE_DIRNAME not in path.parts]
        archived_existing = [path for path in existing if _EXPERIMENT_ARCHIVE_DIRNAME in path.parts]
        if live_existing and archived_existing:
            raise ValueError(f"Experiment path conflict: {experiment_id}")
        if live_existing:
            return live_existing[0]
        if archived_existing:
            return archived_existing[0]
        existing_dirs = [path for path in candidates if path.is_dir()]
        if existing_dirs:
            return existing_dirs[0]
        return candidates[0]

    def _iter_experiment_ids(self, *, include_archived: bool) -> list[str]:
        ids: set[str] = set()
        for experiment_dir in (self._experiments_root(), self._legacy_experiments_root()):
            if not experiment_dir.exists():
                continue
            for child in experiment_dir.iterdir():
                if child.is_dir() and not child.name.startswith(".") and child.name != _EXPERIMENT_ARCHIVE_DIRNAME:
                    ids.add(child.name)
        if include_archived:
            for archive_dir in (self._archived_experiments_root(), self._legacy_archived_experiments_root()):
                if not archive_dir.exists():
                    continue
                for child in archive_dir.iterdir():
                    if child.is_dir() and not child.name.startswith("."):
                        ids.add(child.name)
        if self._table_exists("experiments"):
            if include_archived:
                rows = self._query("SELECT experiment_id FROM experiments")
            else:
                rows = self._query("SELECT experiment_id FROM experiments WHERE status != 'archived'")
            for row in rows:
                value = row["experiment_id"]
                if isinstance(value, str) and value:
                    ids.add(value)
        return sorted(ids)

    def _iter_configs(self, experiment_id: str) -> list[Path]:
        root = self._resolve_experiment_dir(experiment_id)
        paths: list[Path] = []
        if (root / "configs").exists():
            paths.extend(sorted((root / "configs").glob("*.json")))
            paths.extend(sorted((root / "configs").glob("*.yaml")))
            paths.extend(sorted((root / "configs").glob("*.yml")))
        if (root / "config.json").exists():
            paths.append(root / "config.json")
        if (root / "config.yaml").exists():
            paths.append(root / "config.yaml")
        return paths

    def _build_config_item(self, path: Path, experiment_id: str) -> dict[str, Any]:
        payload = self._safe_read_yaml(path)
        summary = self._config_summary(payload)
        runnable, reason = self._runnable_status(payload)
        rel = path.relative_to(self._workspace_root()).as_posix()
        return {
            "config_id": rel,
            "relative_path": rel,
            "abs_path": str(path),
            "experiment_id": experiment_id,
            "sha256": self._sha256(path),
            "mtime": path.stat().st_mtime,
            "summary": summary,
            "is_runnable": runnable,
            "runnable_reason": reason,
        }

    def list_experiment_configs(
        self,
        experiment_id: str,
        *,
        q: str | None = None,
        runnable_only: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        _ensure_safe_id(experiment_id, label="experiment_id")
        items: list[dict[str, Any]] = []
        for path in self._iter_configs(experiment_id):
            item = self._build_config_item(path, experiment_id)
            if runnable_only and not item["is_runnable"]:
                continue
            if q:
                query = q.strip().lower()
                summary = item["summary"]
                haystack = " ".join(
                    [
                        item["relative_path"],
                        str(summary.get("run_id") or ""),
                        str(summary.get("model_type") or ""),
                        str(summary.get("target") or ""),
                        str(summary.get("feature_set") or ""),
                    ]
                ).lower()
                if query not in haystack:
                    continue
            items.append(item)

        items.sort(key=lambda row: row["relative_path"])
        total = len(items)
        sliced = items[offset : offset + limit]
        return {
            "experiment_id": experiment_id,
            "items": sliced,
            "total": total,
        }

    def list_all_configs(
        self,
        *,
        q: str | None = None,
        experiment_id: str | None = None,
        model_type: str | None = None,
        target: str | None = None,
        runnable_only: bool = False,
        limit: int = 500,
        offset: int = 0,
    ) -> dict[str, Any]:
        exp_ids = [experiment_id] if experiment_id else self._iter_experiment_ids(include_archived=False)
        items: list[dict[str, Any]] = []
        for exp_id in exp_ids:
            for path in self._iter_configs(exp_id):
                item = self._build_config_item(path, exp_id)
                summary = item["summary"]
                if runnable_only and not item["is_runnable"]:
                    continue
                if model_type and summary.get("model_type") != model_type:
                    continue
                if target and summary.get("target") != target:
                    continue
                if q:
                    query = q.strip().lower()
                    haystack = " ".join(
                        [
                            item["relative_path"],
                            exp_id,
                            str(summary.get("run_id") or ""),
                            str(summary.get("model_type") or ""),
                            str(summary.get("target") or ""),
                            str(summary.get("feature_set") or ""),
                        ]
                    ).lower()
                    if query not in haystack:
                        continue
                item.pop("abs_path", None)
                items.append(item)

        items.sort(key=lambda row: row["relative_path"])
        total = len(items)
        sliced = items[offset : offset + limit]
        return {"items": sliced, "total": total}

    def read_config_yaml(self, config_id: str) -> str:
        path = self._resolve_config_id(config_id)
        return path.read_text()

    def _format_experiment(self, row: dict[str, Any], run_count: int | None) -> dict[str, Any]:
        metadata = _parse_json_object(row.get("metadata_json"))
        status = row.get("status")
        if status == "concluded":
            status = "complete"
        tags = metadata.get("tags")
        return {
            "experiment_id": row.get("experiment_id"),
            "name": row.get("name") or row.get("experiment_id"),
            "status": status,
            "created_at": row.get("created_at"),
            "updated_at": row.get("updated_at"),
            "preset": metadata.get("preset"),
            "hypothesis": metadata.get("hypothesis"),
            "tags": tags if isinstance(tags, list) else [],
            "champion_run_id": metadata.get("champion_run_id"),
            "run_count": run_count,
            "metadata": metadata,
        }

    def _decode_lifecycle_row(self, row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        item = dict(row)
        item["completed_stages"] = _parse_json_list(item.pop("completed_stages_json", None))
        item["terminal_detail"] = _parse_json_object(item.pop("terminal_detail_json", None))
        item["latest_metrics"] = _parse_json_object(item.pop("latest_metrics_json", None))
        item["latest_sample"] = _parse_json_object(item.pop("latest_sample_json", None))
        item["cancel_requested"] = bool(item.get("cancel_requested"))
        item["reconciled"] = bool(item.get("reconciled"))
        return item

    def _attention_state_for_status(self, status: Any) -> str:
        normalized = _to_non_empty_str(status)
        if normalized in {"failed", "stale", "canceled"}:
            return normalized
        return "none"

    def _overview_activity_at(self, *values: Any) -> str | None:
        timestamps = [str(value) for value in values if isinstance(value, str) and value.strip()]
        if not timestamps:
            return None
        return max(timestamps)

    def _overview_config_label(self, job: dict[str, Any], lifecycle: dict[str, Any] | None) -> str:
        for raw_value in (
            job.get("config_path"),
            lifecycle.get("config_path") if lifecycle else None,
            job.get("config_id"),
            lifecycle.get("config_id") if lifecycle else None,
        ):
            text = _to_non_empty_str(raw_value)
            if text:
                return self._overview_path_label(text)
        return "unknown-config"

    def _overview_path_label(self, value: str) -> str:
        if "\\" in value:
            return PureWindowsPath(value).name
        return Path(value).name

    def _overview_summary_from_experiments(self, experiments: list[dict[str, Any]]) -> dict[str, int]:
        return {
            "total_experiments": len(experiments),
            "active_experiments": sum(1 for item in experiments if item.get("status") == "active"),
            "completed_experiments": sum(1 for item in experiments if item.get("status") == "complete"),
            "live_experiments": 0,
            "live_runs": 0,
            "queued_runs": 0,
            "attention_count": 0,
        }

    def _overview_synthesized_experiment_status(
        self,
        *,
        job: dict[str, Any],
        lifecycle: dict[str, Any] | None,
    ) -> str:
        status = _to_non_empty_str(lifecycle.get("status")) if lifecycle else None
        if status is None:
            status = _to_non_empty_str(job.get("status"))
        if status in _TERMINAL_JOB_STATUSES:
            return "complete"
        return "active"

    def _overview_synthesized_experiment_item(
        self,
        *,
        experiment_id: str,
        job: dict[str, Any],
        lifecycle: dict[str, Any] | None,
    ) -> dict[str, Any]:
        created_at = self._overview_activity_at(job.get("created_at"))
        updated_at = self._overview_activity_at(
            lifecycle.get("updated_at") if lifecycle else None,
            job.get("updated_at"),
            job.get("finished_at"),
            job.get("started_at"),
            job.get("created_at"),
        )
        return {
            "experiment_id": experiment_id,
            "name": experiment_id,
            "status": self._overview_synthesized_experiment_status(job=job, lifecycle=lifecycle),
            "created_at": created_at,
            "updated_at": updated_at,
            "run_count": 0,
            "tags": [],
            "has_live": False,
            "live_run_count": 0,
            "attention_state": "none",
            "latest_activity_at": updated_at or created_at,
        }

    def get_experiments_overview(self) -> dict[str, Any]:
        experiments = [dict(item) for item in self.list_experiments()]
        experiment_items: dict[str, dict[str, Any]] = {}
        for item in experiments:
            experiment_id = _to_non_empty_str(item.get("experiment_id"))
            if experiment_id is None:
                continue
            experiment_items[experiment_id] = {
                **item,
                "has_live": False,
                "live_run_count": 0,
                "attention_state": "none",
                "latest_activity_at": item.get("updated_at") or item.get("created_at"),
            }

        if not self._table_exists("run_jobs"):
            ordered = sorted(
                experiment_items.values(),
                key=lambda item: (
                    int(bool(item.get("has_live"))),
                    _ATTENTION_STATUS_RANK.get(str(item.get("attention_state") or "none"), 0),
                    str(item.get("updated_at") or item.get("created_at") or ""),
                    str(item.get("experiment_id") or ""),
                ),
                reverse=True,
            )
            return {
                "summary": self._overview_summary_from_experiments(ordered),
                "experiments": ordered,
                "live_experiments": [],
                "recent_activity": [],
            }

        latest_job_rows = self._query(
            """
            WITH ranked AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY COALESCE(logical_run_id, job_id)
                        ORDER BY COALESCE(attempt_no, 0) DESC, created_at DESC
                    ) AS __rn
                FROM run_jobs
            )
            SELECT *
            FROM ranked
            WHERE __rn = 1
            ORDER BY created_at DESC
            """
        )
        latest_jobs = [self._decode_job_row(row) for row in latest_job_rows]

        lifecycle_by_run_id: dict[str, dict[str, Any]] = {}
        if self._table_exists("run_lifecycles"):
            for row in self._query("SELECT * FROM run_lifecycles ORDER BY updated_at DESC"):
                lifecycle = self._decode_lifecycle_row(row)
                run_id = _to_non_empty_str(lifecycle.get("run_id"))
                if run_id is None:
                    continue
                lifecycle_by_run_id[run_id] = lifecycle

        latest_terminal_by_experiment: dict[str, dict[str, Any]] = {}
        live_runs_by_experiment: dict[str, list[dict[str, Any]]] = {}
        recent_activity: list[dict[str, Any]] = []
        synthesized_experiment_ids: set[str] = set()

        for job in latest_jobs:
            experiment_id = _to_non_empty_str(job.get("experiment_id"))
            if experiment_id is None:
                continue

            run_id = _to_non_empty_str(job.get("canonical_run_id")) or _to_non_empty_str(job.get("job_id"))
            lifecycle = lifecycle_by_run_id.get(run_id or "")
            if experiment_id not in experiment_items:
                experiment_items[experiment_id] = self._overview_synthesized_experiment_item(
                    experiment_id=experiment_id,
                    job=job,
                    lifecycle=lifecycle,
                )
                synthesized_experiment_ids.add(experiment_id)
            if experiment_id in synthesized_experiment_ids:
                experiment_items[experiment_id]["run_count"] = (
                    int(experiment_items[experiment_id].get("run_count") or 0) + 1
                )
            activity_at = self._overview_activity_at(
                lifecycle.get("last_heartbeat_at") if lifecycle else None,
                lifecycle.get("updated_at") if lifecycle else None,
                job.get("updated_at"),
                job.get("finished_at"),
                job.get("started_at"),
                job.get("created_at"),
            )
            experiment_items[experiment_id]["latest_activity_at"] = self._overview_activity_at(
                experiment_items[experiment_id].get("latest_activity_at"),
                activity_at,
            )

            job_status = _to_non_empty_str(job.get("status")) or "unknown"
            lifecycle_status = _to_non_empty_str(lifecycle.get("status")) if lifecycle else None
            # Mission control should reflect actual in-flight runs only. The lifecycle
            # table is the canonical live-truth surface; stale run_jobs rows without an
            # active lifecycle must not appear as live operations.
            is_live = lifecycle_status in _ACTIVE_JOB_STATUSES if lifecycle_status else False
            config_label = self._overview_config_label(job, lifecycle)
            run_status = lifecycle_status or job_status
            run_progress = lifecycle.get("progress_percent") if lifecycle else None
            run_progress_label = lifecycle.get("progress_label") if lifecycle else None
            current_stage = lifecycle.get("current_stage") if lifecycle else None
            terminal_reason = (
                _to_non_empty_str(lifecycle.get("terminal_reason"))
                if lifecycle
                else _to_non_empty_str(job.get("terminal_reason"))
            )

            if is_live and run_id:
                run_item = {
                    "run_id": run_id,
                    "job_id": _to_non_empty_str(job.get("job_id")),
                    "config_id": _to_non_empty_str(job.get("config_id")),
                    "config_label": config_label,
                    "status": run_status,
                    "current_stage": current_stage,
                    "progress_percent": run_progress,
                    "progress_label": run_progress_label,
                    "updated_at": activity_at,
                    "terminal_reason": terminal_reason,
                }
                live_runs_by_experiment.setdefault(experiment_id, []).append(run_item)

            if job_status in _TERMINAL_JOB_STATUSES:
                terminal_item = {
                    "experiment_id": experiment_id,
                    "experiment_name": experiment_items[experiment_id].get("name") or experiment_id,
                    "run_id": run_id,
                    "job_id": _to_non_empty_str(job.get("job_id")),
                    "config_id": _to_non_empty_str(job.get("config_id")),
                    "config_label": config_label,
                    "status": run_status,
                    "current_stage": current_stage,
                    "progress_percent": run_progress,
                    "progress_label": run_progress_label,
                    "updated_at": activity_at,
                    "finished_at": self._overview_activity_at(
                        lifecycle.get("finished_at") if lifecycle else None,
                        job.get("finished_at"),
                    ),
                    "terminal_reason": terminal_reason,
                }
                recent_activity.append(terminal_item)
                previous = latest_terminal_by_experiment.get(experiment_id)
                if previous is None:
                    latest_terminal_by_experiment[experiment_id] = terminal_item
                else:
                    previous_activity = self._overview_activity_at(
                        previous.get("updated_at"), previous.get("finished_at")
                    )
                    next_activity = self._overview_activity_at(
                        terminal_item.get("updated_at"), terminal_item.get("finished_at")
                    )
                    if (next_activity or "") > (previous_activity or ""):
                        latest_terminal_by_experiment[experiment_id] = terminal_item
                    elif (next_activity or "") == (previous_activity or ""):
                        previous_rank = _ATTENTION_STATUS_RANK.get(
                            self._attention_state_for_status(previous.get("status")),
                            0,
                        )
                        next_rank = _ATTENTION_STATUS_RANK.get(
                            self._attention_state_for_status(terminal_item.get("status")),
                            0,
                        )
                        if next_rank > previous_rank:
                            latest_terminal_by_experiment[experiment_id] = terminal_item

        live_experiments: list[dict[str, Any]] = []
        for experiment_id, item in experiment_items.items():
            latest_terminal = latest_terminal_by_experiment.get(experiment_id)
            attention_state = self._attention_state_for_status(
                latest_terminal.get("status") if latest_terminal else None
            )
            live_runs = live_runs_by_experiment.get(experiment_id, [])
            live_runs.sort(
                key=lambda row: (
                    int((_to_non_empty_str(row.get("status")) or "") == "running"),
                    int((_to_non_empty_str(row.get("status")) or "") == "starting"),
                    int((_to_non_empty_str(row.get("status")) or "") == "queued"),
                    str(row.get("updated_at") or ""),
                ),
                reverse=True,
            )
            item["has_live"] = len(live_runs) > 0
            item["live_run_count"] = len(live_runs)
            item["attention_state"] = attention_state

            if not live_runs:
                continue

            progress_values: list[float] = []
            queued_count = 0
            for run_item in live_runs:
                status = _to_non_empty_str(run_item.get("status")) or "unknown"
                if status == "queued":
                    queued_count += 1
                progress_value = run_item.get("progress_percent")
                if isinstance(progress_value, (int, float)):
                    progress_values.append(float(progress_value))
                elif status == "queued":
                    progress_values.append(0.0)

            aggregate_progress_percent = None
            if progress_values:
                aggregate_progress_percent = max(0.0, min(100.0, sum(progress_values) / len(progress_values)))

            live_experiments.append(
                {
                    "experiment_id": experiment_id,
                    "name": item.get("name") or experiment_id,
                    "status": item.get("status"),
                    "tags": item.get("tags") if isinstance(item.get("tags"), list) else [],
                    "live_run_count": len(live_runs),
                    "queued_run_count": queued_count,
                    "attention_state": attention_state,
                    "latest_activity_at": item.get("latest_activity_at"),
                    "aggregate_progress_percent": aggregate_progress_percent,
                    "runs": live_runs,
                }
            )

        recent_activity.sort(
            key=lambda item: (
                self._overview_activity_at(item.get("updated_at"), item.get("finished_at")) or "",
                _ATTENTION_STATUS_RANK.get(self._attention_state_for_status(item.get("status")), 0),
            ),
            reverse=True,
        )
        recent_activity = recent_activity[:8]

        for experiment_id, item in experiment_items.items():
            latest_terminal = latest_terminal_by_experiment.get(experiment_id)
            if latest_terminal is not None:
                item["attention_state"] = self._attention_state_for_status(latest_terminal.get("status"))

            ordered_experiments = sorted(
                experiment_items.values(),
                key=lambda item: (
                    int(bool(item.get("has_live"))),
                    _ATTENTION_STATUS_RANK.get(str(item.get("attention_state") or "none"), 0),
                    str(item.get("latest_activity_at") or item.get("updated_at") or item.get("created_at") or ""),
                    str(item.get("experiment_id") or ""),
                ),
                reverse=True,
            )

        summary = self._overview_summary_from_experiments(ordered_experiments)
        summary["live_experiments"] = len(live_experiments)
        summary["live_runs"] = sum(int(item.get("live_run_count") or 0) for item in live_experiments)
        summary["queued_runs"] = sum(int(item.get("queued_run_count") or 0) for item in live_experiments)
        summary["attention_count"] = sum(
            1 for item in ordered_experiments if str(item.get("attention_state") or "none") != "none"
        )

        live_experiments.sort(
            key=lambda item: (
                _ATTENTION_STATUS_RANK.get(str(item.get("attention_state") or "none"), 0),
                str(item.get("latest_activity_at") or ""),
                str(item.get("experiment_id") or ""),
            ),
            reverse=True,
        )

        return {
            "summary": summary,
            "experiments": ordered_experiments,
            "live_experiments": live_experiments,
            "recent_activity": recent_activity,
        }

    def list_experiments(self) -> list[dict[str, Any]]:
        if self._table_exists("experiments"):
            rows = self._query(
                """
                SELECT experiment_id, name, status, created_at, updated_at, metadata_json
                FROM experiments
                WHERE status != 'archived'
                ORDER BY created_at DESC
                """
            )
            counts: dict[str, int] = {}
            if self._table_exists("runs"):
                for row in self._query("SELECT experiment_id, COUNT(*) AS cnt FROM runs GROUP BY experiment_id"):
                    exp_id = row["experiment_id"]
                    if isinstance(exp_id, str):
                        counts[exp_id] = int(row["cnt"])
            payload: list[dict[str, Any]] = []
            for row in rows:
                row_dict = dict(row)
                exp_id = row_dict.get("experiment_id")
                run_count = 0
                if isinstance(exp_id, str):
                    run_count = max(counts.get(exp_id, 0), len(self._manifest_run_ids(exp_id)))
                payload.append(self._format_experiment(row_dict, run_count=run_count))
            return payload

        # Filesystem fallback when DB is absent.
        items: list[dict[str, Any]] = []
        for exp_id in self._iter_experiment_ids(include_archived=False):
            exp_dir = self._experiments_root() / exp_id
            manifest = _read_json_dict(exp_dir / "experiment.json")
            created_at = None
            updated_at = None
            if exp_dir.exists():
                stat = exp_dir.stat()
                created_at = datetime.fromtimestamp(stat.st_ctime, tz=UTC).isoformat()
                updated_at = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()
            runs_raw = manifest.get("runs")
            items.append(
                {
                    "experiment_id": exp_id,
                    "name": _to_non_empty_str(manifest.get("name")) or exp_id,
                    "status": _to_non_empty_str(manifest.get("status")) or "draft",
                    "created_at": created_at,
                    "updated_at": updated_at,
                    "preset": None,
                    "hypothesis": _to_non_empty_str(manifest.get("hypothesis")),
                    "tags": manifest.get("tags") if isinstance(manifest.get("tags"), list) else [],
                    "champion_run_id": _to_non_empty_str(manifest.get("champion_run_id")),
                    "run_count": len(runs_raw) if isinstance(runs_raw, list) else 0,
                    "metadata": manifest.get("metadata") if isinstance(manifest.get("metadata"), dict) else {},
                }
            )
        return items

    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        _ensure_safe_id(experiment_id, label="experiment_id")
        if self._table_exists("experiments"):
            row = self._query_one(
                """
                SELECT experiment_id, name, status, created_at, updated_at, metadata_json
                FROM experiments
                WHERE experiment_id = ?
                """,
                (experiment_id,),
            )
            if row is not None:
                run_count = 0
                if self._table_exists("runs"):
                    count_row = self._query_one(
                        "SELECT COUNT(*) AS cnt FROM runs WHERE experiment_id = ?",
                        (experiment_id,),
                    )
                    run_count = int(count_row["cnt"]) if count_row else 0
                run_count = max(run_count, len(self._manifest_run_ids(experiment_id)))
                payload = self._format_experiment(dict(row), run_count=run_count)
                payload["study_count"] = self.count_studies(experiment_id=experiment_id)
                payload["ensemble_count"] = self.count_ensembles(experiment_id=experiment_id)
                return payload

        resolved_dir = self._resolve_experiment_dir(experiment_id)
        manifest_path = resolved_dir / "experiment.json"
        manifest = _read_json_dict(manifest_path)
        if resolved_dir.exists() and resolved_dir.is_dir():
            created_at = datetime.fromtimestamp(resolved_dir.stat().st_ctime, tz=UTC).isoformat()
            updated_at = datetime.fromtimestamp(resolved_dir.stat().st_mtime, tz=UTC).isoformat()
            runs_raw = manifest.get("runs")
            return {
                "experiment_id": experiment_id,
                "name": _to_non_empty_str(manifest.get("name")) or experiment_id,
                "status": _to_non_empty_str(manifest.get("status")) or "draft",
                "created_at": created_at,
                "updated_at": updated_at,
                "preset": None,
                "hypothesis": _to_non_empty_str(manifest.get("hypothesis")),
                "tags": manifest.get("tags") if isinstance(manifest.get("tags"), list) else [],
                "champion_run_id": _to_non_empty_str(manifest.get("champion_run_id")),
                "run_count": len(runs_raw) if isinstance(runs_raw, list) else 0,
                "metadata": manifest.get("metadata") if isinstance(manifest.get("metadata"), dict) else {},
                "study_count": 0,
                "ensemble_count": 0,
            }
        return None

    def _run_metrics_map(self, run_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not run_ids or not self._table_exists("metrics"):
            return {}
        placeholders = ",".join("?" for _ in run_ids)
        rows = self._query(
            f"SELECT run_id, name, value, value_json FROM metrics WHERE run_id IN ({placeholders})",
            tuple(run_ids),
        )
        raw_metrics: dict[str, dict[str, Any]] = {}
        for row in rows:
            run_id = row["run_id"]
            name = row["name"]
            if not isinstance(run_id, str) or not isinstance(name, str):
                continue
            raw_metrics.setdefault(run_id, {})[name] = _metric_row_value(row["value"], row["value_json"])
        return raw_metrics

    def _run_target_map(self, run_ids: list[str]) -> dict[str, str | None]:
        if not run_ids:
            return {}
        target_map: dict[str, str | None] = {}

        if self._table_exists("runs"):
            placeholders = ",".join("?" for _ in run_ids)
            rows = self._query(
                f"SELECT run_id, manifest_json FROM runs WHERE run_id IN ({placeholders})",
                tuple(run_ids),
            )
            for row in rows:
                run_id = row["run_id"]
                if not isinstance(run_id, str):
                    continue
                manifest = _parse_json_object(row["manifest_json"])
                target_train, _, _ = _resolve_run_targets(manifest)
                target_map[run_id] = target_train

        for run_id in run_ids:
            if run_id in target_map:
                continue
            manifest = _read_json_dict(self._resolve_run_dir(run_id) / "run.json")
            target_train, _, _ = _resolve_run_targets(manifest)
            target_map[run_id] = target_train

        return target_map

    def _format_run(
        self,
        row: dict[str, Any],
        *,
        champion_run_id: str | None,
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        manifest = _parse_json_object(row.get("manifest_json"))
        run_id = row.get("run_id")
        resolved = _read_json_dict(self._resolve_run_dir(run_id) / "resolved.json") if isinstance(run_id, str) else {}
        model_raw = manifest.get("model")
        model_info: dict[str, Any] = model_raw if isinstance(model_raw, dict) else {}
        data_raw = manifest.get("data")
        data_info: dict[str, Any] = data_raw if isinstance(data_raw, dict) else {}
        lineage_raw = manifest.get("lineage")
        lineage: dict[str, Any] = lineage_raw if isinstance(lineage_raw, dict) else {}
        target_train, target_payout, target = _resolve_run_targets(manifest)
        return {
            "run_id": run_id,
            "experiment_id": row.get("experiment_id"),
            "created_at": row.get("created_at"),
            "status": row.get("status"),
            "is_champion": bool(champion_run_id and run_id == champion_run_id),
            "metrics": _normalize_round_metrics(metrics, target_col=target_train),
            "config_hash": manifest.get("config_hash"),
            "notes": None,
            "round_id": lineage.get("round_id"),
            "round_index": lineage.get("round_index"),
            "sweep_dimension": lineage.get("sweep_dimension"),
            "run_name": manifest.get("run_name"),
            "model_type": model_info.get("type") if model_info else manifest.get("model_type"),
            "seed": _extract_run_seed(resolved) or _extract_run_seed(manifest),
            "target_train": target_train,
            "target_payout": target_payout,
            "target_col": data_info.get("target_col"),
            "target": target,
            "feature_set": data_info.get("feature_set"),
        }

    def list_experiment_runs(self, experiment_id: str) -> list[dict[str, Any]]:
        _ensure_safe_id(experiment_id, label="experiment_id")
        champion_run_id: str | None = None
        experiment = self.get_experiment(experiment_id)
        if experiment:
            metadata = experiment.get("metadata")
            if isinstance(metadata, dict):
                champion = metadata.get("champion_run_id")
                if isinstance(champion, str):
                    champion_run_id = champion

        payload: list[dict[str, Any]] = []
        seen_run_ids: set[str] = set()
        if self._table_exists("runs"):
            rows = self._query(
                """
                SELECT run_id, experiment_id, status, created_at, manifest_json
                FROM runs
                WHERE experiment_id = ?
                ORDER BY created_at DESC
                """,
                (experiment_id,),
            )
            run_ids = [row["run_id"] for row in rows if isinstance(row["run_id"], str)]
            metrics_map = self._run_metrics_map(run_ids)
            for row in rows:
                row_dict = dict(row)
                run_id = row_dict.get("run_id")
                if not isinstance(run_id, str):
                    continue
                seen_run_ids.add(run_id)
                payload.append(
                    self._format_run(
                        row_dict,
                        champion_run_id=champion_run_id,
                        metrics=metrics_map.get(run_id, {}),
                    )
                )

        candidate_run_ids = self._manifest_run_ids(experiment_id)
        if not candidate_run_ids:
            runs_dir = self.store_root / "runs"
            if runs_dir.exists():
                candidate_run_ids = sorted(
                    [child.name for child in runs_dir.iterdir() if child.is_dir()],
                )

        for run_id in candidate_run_ids:
            if run_id in seen_run_ids:
                continue
            run_dir = self._resolve_run_dir(run_id)
            manifest_path = run_dir / "run.json"
            if not manifest_path.exists():
                continue
            try:
                payload_raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if not isinstance(payload_raw, dict):
                continue
            lineage_raw = payload_raw.get("lineage")
            lineage: dict[str, Any] = lineage_raw if isinstance(lineage_raw, dict) else {}
            exp_id = payload_raw.get("experiment_id") or lineage.get("experiment_id")
            if exp_id != experiment_id:
                continue
            manifest_run_id = payload_raw.get("run_id")
            if not isinstance(manifest_run_id, str):
                continue
            metrics = {}
            metrics_path = run_dir / "metrics.json"
            if metrics_path.exists():
                try:
                    metric_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    metric_payload = {}
                if isinstance(metric_payload, dict):
                    metrics = metric_payload
            model_raw = payload_raw.get("model")
            model_info: dict[str, Any] = model_raw if isinstance(model_raw, dict) else {}
            data_raw = payload_raw.get("data")
            data_info: dict[str, Any] = data_raw if isinstance(data_raw, dict) else {}
            target_train, target_payout, target = _resolve_run_targets(payload_raw)
            resolved_payload = _read_json_dict(run_dir / "resolved.json")
            payload.append(
                {
                    "run_id": manifest_run_id,
                    "experiment_id": experiment_id,
                    "created_at": payload_raw.get("created_at"),
                    "status": payload_raw.get("status"),
                    "is_champion": bool(champion_run_id and champion_run_id == manifest_run_id),
                    "metrics": _normalize_round_metrics(metrics, target_col=target_train),
                    "config_hash": payload_raw.get("config_hash"),
                    "notes": None,
                    "round_id": lineage.get("round_id"),
                    "round_index": lineage.get("round_index"),
                    "sweep_dimension": lineage.get("sweep_dimension"),
                    "run_name": payload_raw.get("run_name"),
                    "model_type": model_info.get("type") or payload_raw.get("model_type"),
                    "seed": _extract_run_seed(resolved_payload) or _extract_run_seed(payload_raw),
                    "target_train": target_train,
                    "target_payout": target_payout,
                    "target_col": data_info.get("target_col"),
                    "target": target,
                    "feature_set": data_info.get("feature_set"),
                }
            )
        payload.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return payload

    def list_experiment_round_results(self, experiment_id: str) -> list[dict[str, Any]]:
        _ensure_safe_id(experiment_id, label="experiment_id")
        exp_dir = self._resolve_experiment_dir(experiment_id)
        results_dir = exp_dir / "results"
        if not results_dir.exists():
            return []

        config_payload = self.list_experiment_configs(
            experiment_id,
            q=None,
            runnable_only=False,
            limit=5000,
            offset=0,
        )
        config_by_round_key: dict[str, dict[str, Any]] = {}
        for item in config_payload.get("items", []):
            rel_path = item.get("relative_path")
            if not isinstance(rel_path, str):
                continue
            config_stem = Path(rel_path).stem
            round_key = _round_key_from_config_stem(config_stem)
            summary = item.get("summary")
            if not isinstance(summary, dict):
                summary = {}
            if round_key not in config_by_round_key:
                config_by_round_key[round_key] = summary

        payload: list[dict[str, Any]] = []
        for metrics_path in sorted(results_dir.glob("*_metrics.json")):
            try:
                raw = json.loads(metrics_path.read_text())
            except Exception:
                continue
            if not isinstance(raw, dict):
                continue
            stem = metrics_path.stem
            round_key = stem[:-8] if stem.endswith("_metrics") else stem
            summary = config_by_round_key.get(round_key, {})
            target_train = summary.get("target")
            target_payout = summary.get("target_payout")

            normalized_metrics = _normalize_round_metrics(
                raw,
                target_col=_to_non_empty_str(target_train),
            )
            created_at = datetime.fromtimestamp(metrics_path.stat().st_mtime, tz=UTC).isoformat()
            try:
                source_file = metrics_path.relative_to(self._workspace_root()).as_posix()
            except ValueError:
                source_file = str(metrics_path)

            payload.append(
                {
                    "result_id": round_key,
                    "run_id": f"derived:{round_key}",
                    "name": round_key.replace("_", " "),
                    "round_index": _round_index_from_key(round_key),
                    "created_at": created_at,
                    "metrics": normalized_metrics,
                    "model_type": _infer_round_model_type(round_key, summary.get("model_type")),
                    "target_train": target_train,
                    "target_payout": target_payout,
                    "target": target_payout or target_train,
                    "feature_set": summary.get("feature_set"),
                    "source_file": source_file,
                }
            )

        payload.sort(
            key=lambda item: (
                item.get("round_index") is None,
                item.get("round_index") if item.get("round_index") is not None else 10_000,
                item.get("result_id"),
            )
        )
        return payload

    def linked_runs_for_configs(self, config_ids: list[str]) -> dict[str, str]:
        if not config_ids or not self._table_exists("run_jobs"):
            return {}
        placeholders = ",".join("?" for _ in config_ids)
        rows = self._query(
            f"""
            SELECT config_id, canonical_run_id
            FROM run_jobs
            WHERE config_id IN ({placeholders})
              AND status = 'completed'
              AND canonical_run_id IS NOT NULL
            ORDER BY finished_at DESC, created_at DESC
            """,
            tuple(config_ids),
        )
        linked: dict[str, str] = {}
        for row in rows:
            config_id = row["config_id"]
            canonical_run_id = row["canonical_run_id"]
            if not isinstance(config_id, str) or not isinstance(canonical_run_id, str):
                continue
            if config_id not in linked:
                linked[config_id] = canonical_run_id
        return linked

    def get_metrics_for_runs(
        self,
        run_ids: list[str],
        metric_names: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        if not run_ids or not self._table_exists("metrics"):
            return {}
        placeholders = ",".join("?" for _ in run_ids)
        params: list[Any] = list(run_ids)
        sql = f"SELECT run_id, name, value, value_json FROM metrics WHERE run_id IN ({placeholders})"
        if metric_names:
            expanded_metric_names = _expand_metric_query_names(metric_names)
            metric_placeholders = ",".join("?" for _ in expanded_metric_names)
            sql += f" AND name IN ({metric_placeholders})"
            params.extend(expanded_metric_names)
        rows = self._query(sql, tuple(params))
        raw_metrics: dict[str, dict[str, Any]] = {}
        for row in rows:
            run_id = row["run_id"]
            name = row["name"]
            if not isinstance(run_id, str) or not isinstance(name, str):
                continue
            raw_metrics.setdefault(run_id, {})[name] = _metric_row_value(row["value"], row["value_json"])
        target_map = self._run_target_map(list(raw_metrics))
        metrics: dict[str, dict[str, Any]] = {}
        for run_id, payload in raw_metrics.items():
            normalized = _normalize_round_metrics(payload, target_col=target_map.get(run_id))
            if metric_names:
                filtered: dict[str, Any] = {}
                for metric_name in metric_names:
                    if metric_name in normalized:
                        filtered[metric_name] = normalized[metric_name]
                metrics[run_id] = filtered
                continue
            metrics[run_id] = normalized
        return metrics

    def _resolve_run_dir(self, run_id: str) -> Path:
        _ensure_safe_id(run_id, label="run_id")
        cached = self._resolved_run_dir_cache.get(run_id)
        if cached is not None and cached.exists() and cached.is_dir():
            return cached
        if cached is not None:
            self._resolved_run_dir_cache.pop(run_id, None)
        if self._table_exists("runs"):
            row = self._query_one("SELECT run_path FROM runs WHERE run_id = ?", (run_id,))
            if row is not None:
                run_path = row["run_path"]
                if isinstance(run_path, str) and run_path:
                    path = Path(run_path)
                    if path.exists() and path.is_dir():
                        self._resolved_run_dir_cache[run_id] = path
                        return path

        if self._table_exists("run_jobs"):
            row = self._query_one(
                """
                SELECT run_dir
                FROM run_jobs
                WHERE canonical_run_id = ?
                  AND run_dir IS NOT NULL
                ORDER BY finished_at DESC, created_at DESC
                LIMIT 1
                """,
                (run_id,),
            )
            if row is not None:
                run_dir = row["run_dir"]
                if isinstance(run_dir, str) and run_dir:
                    path = Path(run_dir)
                    if path.exists() and path.is_dir():
                        self._resolved_run_dir_cache[run_id] = path
                        return path

        local_run_dir = self.store_root / "runs" / run_id
        if local_run_dir.is_dir():
            self._resolved_run_dir_cache[run_id] = local_run_dir
            return local_run_dir

        pulled_run_dir = self._resolve_pulled_run_dir(run_id)
        if pulled_run_dir is not None:
            self._resolved_run_dir_cache[run_id] = pulled_run_dir
            return pulled_run_dir

        return local_run_dir

    def _read_artifact_records(self, *, parquet_path: Path) -> list[dict[str, Any]] | None:
        if parquet_path.exists():
            try:
                return _read_parquet_records_cached(str(parquet_path), parquet_path.stat().st_mtime_ns)
            except Exception:
                logger.debug("Failed parquet read for %s", parquet_path, exc_info=True)
        return None

    def _resolve_run_artifact_file(self, run_dir: Path, raw_path: Any) -> Path | None:
        artifact_path = _to_non_empty_str(raw_path)
        if artifact_path is None:
            return None
        candidate = Path(artifact_path).expanduser()
        if not candidate.is_absolute():
            candidate = (run_dir / candidate).resolve()
        else:
            candidate = candidate.resolve()
        if candidate.exists() and candidate.is_file():
            return candidate
        return None

    def _score_provenance(self, run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
        artifacts_raw = manifest.get("artifacts")
        artifacts = artifacts_raw if isinstance(artifacts_raw, dict) else {}
        explicit_path = self._resolve_run_artifact_file(run_dir, artifacts.get("score_provenance"))
        if explicit_path is not None:
            return _read_json_dict(explicit_path)
        return _read_json_dict(run_dir / "score_provenance.json")

    def _read_table_frame(self, path: Path, *, columns: tuple[str, ...] | None = None) -> pd.DataFrame | None:
        normalized_columns = tuple(dict.fromkeys(columns)) if columns else None
        suffix = path.suffix.lower()
        try:
            if suffix == ".parquet":
                return _read_parquet_frame_cached(str(path), path.stat().st_mtime_ns, normalized_columns)
        except Exception:
            logger.debug("Failed table read for %s", path, exc_info=True)
            return None
        return None

    def _read_scoring_manifest(self, run_dir: Path) -> dict[str, Any]:
        path = run_dir / "artifacts" / "scoring" / "manifest.json"
        return _read_json_dict(path)

    def _summary_row_from_frame(self, path: Path) -> dict[str, Any] | None:
        frame = self._read_table_frame(path)
        if frame is None or frame.empty:
            return None
        return _frame_to_records(frame.iloc[[0]])[0]

    def _summary_row_from_any(self, *paths: Path) -> dict[str, Any] | None:
        for path in paths:
            payload = self._summary_row_from_frame(path)
            if payload is not None:
                return payload
        return None

    def _legacy_scoring_dashboard_series(self, run_dir: Path) -> list[dict[str, Any]]:
        scoring_dir = run_dir / "artifacts" / "scoring"
        rows: list[dict[str, Any]] = []
        for filename, metric_key, series_type in _LEGACY_SCORING_SERIES_SPECS:
            path = scoring_dir / filename
            frame = self._read_table_frame(path)
            if frame is None or frame.empty:
                continue
            normalized = frame.copy()
            if "value" not in normalized.columns and "corr" in normalized.columns:
                normalized["value"] = normalized["corr"]
            if "era" not in normalized.columns or "value" not in normalized.columns:
                continue
            normalized = normalized[["era", "value"]]
            for row in _frame_to_records(normalized):
                rows.append(
                    {
                        "metric_key": metric_key,
                        "series_type": series_type,
                        "era": row.get("era"),
                        "value": row.get("value"),
                    }
                )
        rows.sort(
            key=lambda row: (
                str(row.get("metric_key", "")),
                str(row.get("series_type", "")),
                str(row.get("era", "")),
            )
        )
        return rows

    def _resolve_predictions_artifact_path(
        self,
        run_dir: Path,
        manifest: dict[str, Any],
        score_provenance: dict[str, Any],
    ) -> Path | None:
        sources_raw = score_provenance.get("sources")
        sources = sources_raw if isinstance(sources_raw, dict) else {}
        predictions_source = sources.get("predictions")
        if isinstance(predictions_source, dict):
            source_path = self._resolve_run_artifact_file(run_dir, predictions_source.get("path"))
            if source_path is not None:
                return source_path

        artifacts_raw = manifest.get("artifacts")
        artifacts = artifacts_raw if isinstance(artifacts_raw, dict) else {}
        from_manifest = self._resolve_run_artifact_file(run_dir, artifacts.get("predictions"))
        if from_manifest is not None:
            return from_manifest

        results_payload = _read_json_dict(run_dir / "results.json")
        output_raw = results_payload.get("output")
        output_payload = output_raw if isinstance(output_raw, dict) else {}
        from_results = self._resolve_run_artifact_file(run_dir, output_payload.get("predictions_file"))
        if from_results is not None:
            return from_results

        artifacts_predictions = run_dir / "artifacts" / "predictions"
        if not artifacts_predictions.exists():
            return None

        parquet_files = sorted(artifacts_predictions.glob("*.parquet"))
        if parquet_files:
            return parquet_files[0]
        return None

    def _resolve_meta_model_artifact_path(
        self,
        run_dir: Path,
        manifest: dict[str, Any],
        score_provenance: dict[str, Any],
    ) -> Path | None:
        sources_raw = score_provenance.get("sources")
        sources = sources_raw if isinstance(sources_raw, dict) else {}
        meta_source = sources.get("meta_model")
        if isinstance(meta_source, dict):
            source_path = self._resolve_run_artifact_file(run_dir, meta_source.get("path"))
            if source_path is not None:
                return source_path

        results_payload = _read_json_dict(run_dir / "results.json")
        meta_raw = results_payload.get("meta_model")
        meta_payload = meta_raw if isinstance(meta_raw, dict) else {}
        meta_file = _to_non_empty_str(meta_payload.get("file"))
        if meta_file is not None:
            candidate = (self.store_root / "datasets" / meta_file).resolve()
            if candidate.exists() and candidate.is_file():
                return candidate

        data_raw = manifest.get("data")
        data_payload = data_raw if isinstance(data_raw, dict) else {}
        data_version = _to_non_empty_str(data_payload.get("version")) or _to_non_empty_str(
            data_payload.get("data_version")
        )
        if data_version is None:
            data_raw_results = results_payload.get("data")
            data_payload_results = data_raw_results if isinstance(data_raw_results, dict) else {}
            data_version = _to_non_empty_str(data_payload_results.get("data_version"))
        if data_version is not None:
            candidate = (self.store_root / "datasets" / data_version / "meta_model.parquet").resolve()
            if candidate.exists() and candidate.is_file():
                return candidate
        return None

    def _scoring_context(
        self,
        run_id: str,
        run_dir: Path,
    ) -> tuple[pd.DataFrame, dict[str, Any]] | None:
        manifest = self.get_run_manifest(run_id) or _read_json_dict(run_dir / "run.json")
        score_provenance = self._score_provenance(run_dir, manifest)
        predictions_path = self._resolve_predictions_artifact_path(run_dir, manifest, score_provenance)
        if predictions_path is None:
            return None

        columns_raw = score_provenance.get("columns")
        columns = columns_raw if isinstance(columns_raw, dict) else {}
        prediction_cols_raw = columns.get("prediction_cols")
        prediction_cols = prediction_cols_raw if isinstance(prediction_cols_raw, list) else []
        prediction_col = next(
            (str(col) for col in prediction_cols if isinstance(col, str) and col.strip()),
            "prediction",
        )

        data_raw = manifest.get("data")
        data_payload = data_raw if isinstance(data_raw, dict) else {}
        target_train, _, _ = _resolve_run_targets(manifest)
        target_col = (
            _to_non_empty_str(columns.get("target_col"))
            or _to_non_empty_str(data_payload.get("target_col"))
            or _to_non_empty_str(data_payload.get("target_train"))
            or target_train
            or "target"
        )
        era_col = _to_non_empty_str(columns.get("era_col")) or "era"
        id_col = _to_non_empty_str(columns.get("id_col")) or "id"
        meta_model_col = _to_non_empty_str(columns.get("meta_model_col")) or "numerai_meta_model"

        requested = tuple(dict.fromkeys([era_col, id_col, target_col, prediction_col]))
        frame = self._read_table_frame(predictions_path, columns=requested)
        if frame is None:
            frame = self._read_table_frame(predictions_path)
        if frame is None or frame.empty:
            return None

        frame_columns = {str(col): col for col in frame.columns}
        if era_col not in frame_columns:
            fallback = next((name for name in frame_columns if name.lower() == "era"), None)
            if fallback is None:
                fallback = next((name for name in frame_columns if "era" in name.lower()), None)
            if fallback is None:
                return None
            era_col = fallback

        if target_col not in frame_columns:
            fallback_targets = [name for name in frame_columns if name.lower().startswith("target")]
            if not fallback_targets:
                return None
            target_col = fallback_targets[0]

        if prediction_col not in frame_columns:
            if "prediction" in frame_columns:
                prediction_col = "prediction"
            else:
                ignored = {era_col, id_col, target_col}
                fallback = next((name for name in frame_columns if name not in ignored), None)
                if fallback is None:
                    return None
                prediction_col = fallback

        context = {
            "manifest": manifest,
            "score_provenance": score_provenance,
            "predictions_path": predictions_path,
            "target_col": target_col,
            "era_col": era_col,
            "id_col": id_col if id_col in frame_columns else None,
            "prediction_col": prediction_col,
            "meta_model_col": meta_model_col,
        }
        return frame, context

    def _derive_per_era_corr_fallback(self, run_id: str, run_dir: Path) -> list[dict[str, Any]] | None:
        _, numerai_corr = _load_scoring_functions()
        if numerai_corr is None:
            return None

        loaded = self._scoring_context(run_id, run_dir)
        if loaded is None:
            return None
        frame, context = loaded
        era_col = str(context["era_col"])
        target_col = str(context["target_col"])
        prediction_col = str(context["prediction_col"])

        filtered = frame.dropna(subset=[era_col, target_col, prediction_col])
        if filtered.empty:
            return None

        records: list[dict[str, Any]] = []
        for era, group in filtered.groupby(era_col):
            raw_score = numerai_corr(group[[prediction_col]], group[target_col])
            score = _resolve_series_value(raw_score, prediction_col)
            if score is None:
                continue
            records.append(
                {
                    "era": _coerce_json_value(era),
                    "corr": score,
                }
            )
        return _sort_records_by_era(records) if records else None

    def get_per_era_corr_result(self, run_id: str) -> PerEraCorrLoadResult:
        _ensure_safe_id(run_id, label="run_id")
        run_dir = self._resolve_run_dir(run_id)
        parquet_path = run_dir / "artifacts" / "scoring" / "corr_per_era.parquet"
        read_start = time.perf_counter()
        records = self._read_artifact_records(parquet_path=parquet_path)
        persisted_read_ms = (time.perf_counter() - read_start) * 1000.0
        if records is None:
            materialize_start = time.perf_counter()
            payload = self._derive_per_era_corr_fallback(run_id, run_dir)
            return PerEraCorrLoadResult(
                payload=payload,
                persisted_read_ms=persisted_read_ms,
                materialize_ms=(time.perf_counter() - materialize_start) * 1000.0,
                wrote_artifact=False,
            )

        normalized: list[dict[str, Any]] = []
        for row in records:
            item = dict(row)
            if "corr" not in item and "value" in item:
                item["corr"] = item["value"]
            item.pop("value", None)
            normalized.append(item)
        return PerEraCorrLoadResult(
            payload=normalized,
            persisted_read_ms=persisted_read_ms,
            materialize_ms=0.0,
            wrote_artifact=False,
        )

    def get_per_era_corr(self, run_id: str) -> list[dict[str, Any]] | None:
        return self.get_per_era_corr_result(run_id).payload

    def get_scoring_dashboard(self, run_id: str) -> dict[str, Any] | None:
        _ensure_safe_id(run_id, label="run_id")
        run_dir = self._resolve_run_dir(run_id)
        manifest = self.get_run_manifest(run_id) or _read_json_dict(run_dir / "run.json")
        scoring_manifest = self._read_scoring_manifest(run_dir)
        scoring_dir = run_dir / "artifacts" / "scoring"

        chart_path = scoring_dir / "run_metric_series.parquet"
        if chart_path.exists():
            chart_frame = self._read_table_frame(chart_path)
            series = _frame_to_records(chart_frame) if chart_frame is not None else []
            source = "canonical"
        else:
            series = self._legacy_scoring_dashboard_series(run_dir)
            source = "legacy_fallback"
        series = _normalize_scoring_dashboard_series(series)

        fold_snapshots = None
        fold_path = scoring_dir / "post_fold_snapshots.parquet"
        if fold_path.exists():
            fold_frame = self._read_table_frame(fold_path)
            if fold_frame is not None and not fold_frame.empty:
                fold_snapshots = _frame_to_records(fold_frame)
        fold_snapshots = _normalize_scoring_dashboard_fold_rows(fold_snapshots)

        summary = _normalize_scoring_dashboard_summary_row(
            self._summary_row_from_any(
                scoring_dir / "post_training_core_summary.parquet",
                scoring_dir / "post_training_summary.parquet",
            )
        )
        feature_summary = _normalize_scoring_dashboard_summary_row(
            self._summary_row_from_any(
                scoring_dir / "post_training_full_summary.parquet",
                scoring_dir / "post_training_features_summary.parquet",
            )
        )

        target_train, target_payout, target_col = _resolve_run_targets(manifest)
        meta_target_col = target_col
        meta_payout_col = target_payout or "target_ender_20"
        if summary is not None:
            meta_target_col = _to_non_empty_str(summary.get("target_col")) or meta_target_col
            meta_payout_col = _to_non_empty_str(summary.get("payout_target_col")) or meta_payout_col

        available_metric_keys = sorted(
            {
                str(row.get("metric_key"))
                for row in series
                if isinstance(row, dict) and isinstance(row.get("metric_key"), str)
            }
        )
        stages_raw = scoring_manifest.get("stages")
        stages = stages_raw if isinstance(stages_raw, dict) else {}
        omissions_raw = stages.get("omissions")
        omissions = omissions_raw if isinstance(omissions_raw, dict) else {}

        if not series and fold_snapshots is None and summary is None and feature_summary is None:
            return None

        return {
            "series": series,
            "fold_snapshots": fold_snapshots,
            "summary": summary,
            "feature_summary": feature_summary,
            "meta": {
                "target_col": meta_target_col or target_train,
                "payout_target_col": meta_payout_col,
                "available_metric_keys": available_metric_keys,
                "source": source,
                "omissions": {str(key): _coerce_json_value(value) for key, value in omissions.items()},
            },
        }

    def get_trials(self, run_id: str) -> list[dict[str, Any]] | None:
        _ensure_safe_id(run_id, label="run_id")
        run_dir = self._resolve_run_dir(run_id)
        path = run_dir / "artifacts" / "reports" / "trials.parquet"
        if not path.exists():
            return None
        return _read_parquet_records_cached(str(path), path.stat().st_mtime_ns)

    def get_best_params(self, run_id: str) -> dict[str, Any] | None:
        _ensure_safe_id(run_id, label="run_id")
        run_dir = self._resolve_run_dir(run_id)
        path = run_dir / "artifacts" / "reports" / "best_params.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def get_resolved_config(self, run_id: str) -> dict[str, Any] | None:
        _ensure_safe_id(run_id, label="run_id")
        run_dir = self._resolve_run_dir(run_id)
        json_path = run_dir / "resolved.json"
        if json_path.exists():
            return {"yaml": json_path.read_text()}

        yaml_path = run_dir / "resolved.yaml"
        if not yaml_path.exists():
            return None
        return {"yaml": yaml_path.read_text()}

    def get_run_manifest(self, run_id: str) -> dict[str, Any] | None:
        _ensure_safe_id(run_id, label="run_id")
        if self._table_exists("runs"):
            row = self._query_one("SELECT manifest_json FROM runs WHERE run_id = ?", (run_id,))
            if row is not None:
                manifest = _parse_json_object(row["manifest_json"])
                if manifest:
                    return manifest
        run_dir = self._resolve_run_dir(run_id)
        path = run_dir / "run.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def get_run_metrics(self, run_id: str) -> dict[str, Any] | None:
        _ensure_safe_id(run_id, label="run_id")
        run_dir = self._resolve_run_dir(run_id)
        manifest = self.get_run_manifest(run_id) or _read_json_dict(run_dir / "run.json")
        target_train, _, _ = _resolve_run_targets(manifest)

        def _enrich_metrics(metrics_payload: dict[str, Any]) -> dict[str, Any]:
            normalized = _normalize_round_metrics(metrics_payload, target_col=target_train)
            if "mmc_coverage_ratio_rows" in normalized:
                return normalized
            provenance = self._score_provenance(run_dir, manifest)
            ratio = _coverage_ratio_from_provenance(provenance)
            if ratio is not None:
                normalized["mmc_coverage_ratio_rows"] = ratio
            return normalized

        if self._table_exists("metrics"):
            rows = self._query("SELECT name, value, value_json FROM metrics WHERE run_id = ?", (run_id,))
            if rows:
                payload: dict[str, Any] = {}
                for row in rows:
                    name = row["name"]
                    if isinstance(name, str):
                        payload[name] = _metric_row_value(row["value"], row["value_json"])
                return _enrich_metrics(payload)
        path = run_dir / "metrics.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            return None
        return _enrich_metrics(payload) if isinstance(payload, dict) else None

    def get_diagnostics_sources(self, run_id: str) -> dict[str, Any] | None:
        _ensure_safe_id(run_id, label="run_id")
        run_dir = self._resolve_run_dir(run_id)
        manifest = self.get_run_manifest(run_id) or _read_json_dict(run_dir / "run.json")
        provenance = self._score_provenance(run_dir, manifest)
        if not provenance:
            return None

        payload: dict[str, Any] = {}
        columns_raw = provenance.get("columns")
        if isinstance(columns_raw, dict):
            payload["columns"] = {str(key): _coerce_json_value(value) for key, value in columns_raw.items()}
        joins_raw = provenance.get("joins")
        if isinstance(joins_raw, dict):
            payload["joins"] = {str(key): _coerce_json_value(value) for key, value in joins_raw.items()}

        sources_payload: dict[str, Any] = {}
        sources_raw = provenance.get("sources")
        sources = sources_raw if isinstance(sources_raw, dict) else {}
        for source_name in ("predictions", "meta_model", "benchmark"):
            source_raw = sources.get(source_name)
            source = source_raw if isinstance(source_raw, dict) else {}
            source_path = self._resolve_run_artifact_file(run_dir, source.get("path"))
            item: dict[str, Any] = {}
            if source_path is not None:
                item["path"] = str(source_path)
                item["exists"] = True
            else:
                path_value = _to_non_empty_str(source.get("path"))
                item["path"] = path_value
                item["exists"] = False if path_value else None
            for key in ("sha256", "size_bytes"):
                if key in source:
                    item[key] = _coerce_json_value(source.get(key))
            sources_payload[source_name] = item
        payload["sources"] = sources_payload
        payload["score_provenance_path"] = str((run_dir / "score_provenance.json").resolve())
        return payload

    def list_run_events(self, run_id: str, *, limit: int) -> list[dict[str, Any]]:
        _ensure_safe_id(run_id, label="run_id")
        if not self._table_exists("events"):
            return []
        rows = self._query(
            """
            SELECT id, run_id, event_id, event_type, payload_json, created_at
            FROM events
            WHERE run_id = ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (run_id, limit),
        )
        payload: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["payload"] = _parse_json_object(item.pop("payload_json", None))
            payload.append(item)
        return payload

    def list_run_resources(self, run_id: str, *, limit: int) -> list[dict[str, Any]]:
        _ensure_safe_id(run_id, label="run_id")
        if not self._table_exists("resource_samples"):
            return []
        rows = self._query(
            """
            SELECT id, run_id, cpu, ram, gpu, created_at
            FROM resource_samples
            WHERE run_id = ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (run_id, limit),
        )
        return [dict(row) for row in rows]

    def _decode_job_row(self, row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        item = dict(row)
        item["request"] = _parse_json_object(item.pop("request_json", None)) or None
        item["error"] = _parse_json_object(item.pop("error_json", None)) or None
        item.pop("__rn", None)
        return item

    def _queue_position(self, job_id: str) -> int | None:
        if not self._table_exists("run_jobs"):
            return None
        row = self._query_one(
            "SELECT queue_name, status, priority, created_at FROM run_jobs WHERE job_id = ?",
            (job_id,),
        )
        if row is None or row["status"] != "queued":
            return None
        rank_row = self._query_one(
            """
            SELECT COUNT(*) AS rank
            FROM run_jobs
            WHERE queue_name = ?
              AND status = 'queued'
              AND (
                priority < ?
                OR (priority = ? AND created_at < ?)
              )
            """,
            (row["queue_name"], row["priority"], row["priority"], row["created_at"]),
        )
        if rank_row is None:
            return 1
        return int(rank_row["rank"]) + 1

    def _run_job_filters(
        self,
        *,
        experiment_id: str | None,
        status: str | None,
        batch_id: str | None,
    ) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if experiment_id:
            clauses.append("experiment_id = ?")
            params.append(experiment_id)
        if batch_id:
            clauses.append("batch_id = ?")
            params.append(batch_id)
        if status:
            clauses.append("status = ?")
            params.append(status)
        if not clauses:
            return "", params
        return " WHERE " + " AND ".join(clauses), params

    def list_run_jobs(
        self,
        *,
        experiment_id: str | None = None,
        status: str | None = None,
        batch_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
        include_attempts: bool = False,
    ) -> list[dict[str, Any]]:
        if not self._table_exists("run_jobs"):
            return []

        if include_attempts:
            where_sql, filter_params = self._run_job_filters(
                experiment_id=experiment_id,
                status=status,
                batch_id=batch_id,
            )
            sql = "SELECT * FROM run_jobs" + where_sql + " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            filter_params.extend([limit, offset])
            rows = self._query(sql, tuple(filter_params))
        else:
            filter_clauses: list[str] = []
            latest_params: list[Any] = []
            if experiment_id:
                filter_clauses.append("experiment_id = ?")
                latest_params.append(experiment_id)
            if batch_id:
                filter_clauses.append("batch_id = ?")
                latest_params.append(batch_id)
            where_sql = ""
            if filter_clauses:
                where_sql = "WHERE " + " AND ".join(filter_clauses)
            sql = f"""
                WITH ranked AS (
                    SELECT
                        *,
                        ROW_NUMBER() OVER (
                            PARTITION BY COALESCE(logical_run_id, job_id)
                            ORDER BY COALESCE(attempt_no, 0) DESC, created_at DESC
                        ) AS __rn
                    FROM run_jobs
                    {where_sql}
                )
                SELECT *
                FROM ranked
                WHERE __rn = 1
            """
            if status:
                sql += " AND status = ?"
                latest_params.append(status)
            sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            latest_params.extend([limit, offset])
            rows = self._query(sql, tuple(latest_params))

        items = [self._decode_job_row(row) for row in rows]
        for item in items:
            if item.get("status") == "queued":
                item["queue_position"] = self._queue_position(str(item.get("job_id") or ""))
            else:
                item["queue_position"] = None
        return items

    def count_run_jobs(
        self,
        *,
        experiment_id: str | None = None,
        status: str | None = None,
        batch_id: str | None = None,
        include_attempts: bool = False,
    ) -> int:
        if not self._table_exists("run_jobs"):
            return 0

        if include_attempts:
            where_sql, filter_params = self._run_job_filters(
                experiment_id=experiment_id,
                status=status,
                batch_id=batch_id,
            )
            row = self._query_one("SELECT COUNT(*) AS cnt FROM run_jobs" + where_sql, tuple(filter_params))
            return int(row["cnt"]) if row else 0

        filter_clauses: list[str] = []
        latest_params: list[Any] = []
        if experiment_id:
            filter_clauses.append("experiment_id = ?")
            latest_params.append(experiment_id)
        if batch_id:
            filter_clauses.append("batch_id = ?")
            latest_params.append(batch_id)
        where_sql = ""
        if filter_clauses:
            where_sql = "WHERE " + " AND ".join(filter_clauses)
        sql = f"""
            WITH ranked AS (
                SELECT
                    status,
                    ROW_NUMBER() OVER (
                        PARTITION BY COALESCE(logical_run_id, job_id)
                        ORDER BY COALESCE(attempt_no, 0) DESC, created_at DESC
                    ) AS __rn
                FROM run_jobs
                {where_sql}
            )
            SELECT COUNT(*) AS cnt
            FROM ranked
            WHERE __rn = 1
        """
        if status:
            sql += " AND status = ?"
            latest_params.append(status)
        row = self._query_one(sql, tuple(latest_params))
        return int(row["cnt"]) if row else 0

    def get_run_job(self, job_id: str) -> dict[str, Any] | None:
        _ensure_safe_id(job_id, label="job_id")
        if not self._table_exists("run_jobs"):
            return None
        row = self._query_one("SELECT * FROM run_jobs WHERE job_id = ?", (job_id,))
        if row is None:
            return None
        item = self._decode_job_row(row)
        item["queue_position"] = self._queue_position(job_id) if item.get("status") == "queued" else None
        return item

    def get_run_job_batch(self, batch_id: str) -> list[dict[str, Any]]:
        _ensure_safe_id(batch_id, label="batch_id")
        return self.list_run_jobs(batch_id=batch_id, limit=1000, offset=0, include_attempts=False)

    def list_run_job_events(
        self,
        job_id: str,
        *,
        after_id: int | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        _ensure_safe_id(job_id, label="job_id")
        if not self._table_exists("run_job_events"):
            return []
        sql = """
            SELECT id, job_id, sequence, event_type, source, payload_json, created_at
            FROM run_job_events
            WHERE job_id = ?
        """
        params: list[Any] = [job_id]
        if after_id is not None:
            sql += " AND id > ?"
            params.append(after_id)
        sql += " ORDER BY id ASC LIMIT ?"
        params.append(limit)
        rows = self._query(sql, tuple(params))
        payload: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["payload"] = _parse_json_object(item.pop("payload_json", None))
            payload.append(item)
        return payload

    def list_run_job_logs(
        self,
        job_id: str,
        *,
        after_id: int | None,
        limit: int,
        stream: str,
    ) -> list[dict[str, Any]]:
        _ensure_safe_id(job_id, label="job_id")
        if not self._table_exists("run_job_logs"):
            return []
        sql = """
            SELECT id, job_id, line_no, stream, line, created_at
            FROM run_job_logs
            WHERE job_id = ?
        """
        params: list[Any] = [job_id]
        if after_id is not None:
            sql += " AND id > ?"
            params.append(after_id)
        if stream and stream != "all":
            sql += " AND stream = ?"
            params.append(stream)
        sql += " ORDER BY id ASC LIMIT ?"
        params.append(limit)
        rows = self._query(sql, tuple(params))
        return [dict(row) for row in rows]

    def list_run_job_samples(
        self,
        job_id: str,
        *,
        after_id: int | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        _ensure_safe_id(job_id, label="job_id")
        if not self._table_exists("run_job_samples"):
            return []
        sql = """
            SELECT
                id,
                job_id,
                cpu_percent,
                rss_gb,
                ram_available_gb,
                gpu_percent,
                gpu_mem_gb,
                process_cpu_percent,
                process_rss_gb,
                host_cpu_percent,
                host_ram_available_gb,
                host_ram_used_gb,
                host_gpu_percent,
                host_gpu_mem_used_gb,
                scope,
                status,
                created_at
            FROM run_job_samples
            WHERE job_id = ?
        """
        params: list[Any] = [job_id]
        if after_id is not None:
            sql += " AND id > ?"
            params.append(after_id)
        sql += " ORDER BY id ASC LIMIT ?"
        params.append(limit)
        rows = self._query(sql, tuple(params))

        payload: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            process_cpu_percent = item.get("process_cpu_percent")
            if process_cpu_percent is None:
                process_cpu_percent = item.get("cpu_percent")

            process_rss_gb = item.get("process_rss_gb")
            if process_rss_gb is None:
                process_rss_gb = item.get("rss_gb")

            host_cpu_percent = item.get("host_cpu_percent")

            host_ram_available_gb = item.get("host_ram_available_gb")
            if host_ram_available_gb is None:
                host_ram_available_gb = item.get("ram_available_gb")

            host_ram_used_gb = item.get("host_ram_used_gb")

            host_gpu_percent = item.get("host_gpu_percent")
            if host_gpu_percent is None:
                host_gpu_percent = item.get("gpu_percent")

            host_gpu_mem_used_gb = item.get("host_gpu_mem_used_gb")
            if host_gpu_mem_used_gb is None:
                host_gpu_mem_used_gb = item.get("gpu_mem_gb")

            scope = item.get("scope")
            if not scope:
                if process_cpu_percent is not None or process_rss_gb is not None:
                    scope = "launcher_wrapper_only"
                elif host_ram_available_gb is not None or host_cpu_percent is not None:
                    scope = "launcher_host_only"
                else:
                    scope = "unavailable"

            status = item.get("status")
            if not status:
                status = "partial" if scope != "unavailable" else "unavailable"

            payload.append(
                {
                    "id": item["id"],
                    "job_id": item["job_id"],
                    "cpu_percent": process_cpu_percent,
                    "rss_gb": process_rss_gb,
                    "ram_available_gb": host_ram_available_gb,
                    "gpu_percent": host_gpu_percent,
                    "gpu_mem_gb": host_gpu_mem_used_gb,
                    "process_cpu_percent": process_cpu_percent,
                    "process_rss_gb": process_rss_gb,
                    "host_cpu_percent": host_cpu_percent,
                    "host_ram_available_gb": host_ram_available_gb,
                    "host_ram_used_gb": host_ram_used_gb,
                    "host_gpu_percent": host_gpu_percent,
                    "host_gpu_mem_used_gb": host_gpu_mem_used_gb,
                    "scope": scope,
                    "status": status,
                    "created_at": item["created_at"],
                }
            )
        return payload

    def list_operations(
        self,
        *,
        experiment_id: str,
        operation_type: str | None,
        status: str | None,
        limit: int,
        offset: int,
    ) -> list[dict[str, Any]]:
        if not self._table_exists("logical_runs"):
            return []
        clauses: list[str] = ["experiment_id = ?"]
        params: list[Any] = [experiment_id]
        if operation_type:
            clauses.append("operation_type = ?")
            params.append(operation_type)
        if status:
            clauses.append("status = ?")
            params.append(status)
        where_sql = " WHERE " + " AND ".join(clauses)
        rows = self._query(
            "SELECT * FROM logical_runs" + where_sql + " ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            tuple(params + [limit, offset]),
        )
        return [dict(row) for row in rows]

    def count_operations(
        self,
        *,
        experiment_id: str,
        operation_type: str | None,
        status: str | None,
    ) -> int:
        if not self._table_exists("logical_runs"):
            return 0
        clauses: list[str] = ["experiment_id = ?"]
        params: list[Any] = [experiment_id]
        if operation_type:
            clauses.append("operation_type = ?")
            params.append(operation_type)
        if status:
            clauses.append("status = ?")
            params.append(status)
        where_sql = " WHERE " + " AND ".join(clauses)
        row = self._query_one("SELECT COUNT(*) AS cnt FROM logical_runs" + where_sql, tuple(params))
        return int(row["cnt"]) if row else 0

    def get_operation(self, logical_run_id: str) -> dict[str, Any] | None:
        _ensure_safe_id(logical_run_id, label="logical_run_id")
        if not self._table_exists("logical_runs"):
            return None
        row = self._query_one("SELECT * FROM logical_runs WHERE logical_run_id = ?", (logical_run_id,))
        return dict(row) if row else None

    def list_operation_attempts(self, logical_run_id: str, *, limit: int, offset: int) -> list[dict[str, Any]]:
        _ensure_safe_id(logical_run_id, label="logical_run_id")
        if not self._table_exists("run_attempts") or not self._table_exists("run_jobs"):
            return []
        rows = self._query(
            """
            SELECT
                ra.attempt_id,
                ra.logical_run_id,
                ra.job_id,
                ra.attempt_no,
                ra.status,
                ra.created_at,
                ra.started_at,
                ra.finished_at,
                ra.updated_at,
                ra.worker_id,
                ra.pid,
                ra.exit_code,
                ra.signal,
                ra.error_json,
                ra.canonical_run_id,
                ra.external_run_id,
                ra.run_dir,
                rj.experiment_id,
                rj.config_id,
                rj.job_type
            FROM run_attempts AS ra
            JOIN run_jobs AS rj ON rj.job_id = ra.job_id
            WHERE ra.logical_run_id = ?
            ORDER BY ra.attempt_no DESC, ra.created_at DESC
            LIMIT ? OFFSET ?
            """,
            (logical_run_id, limit, offset),
        )
        payload: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["error"] = _parse_json_object(item.pop("error_json", None)) or None
            payload.append(item)
        return payload

    def count_operation_attempts(self, logical_run_id: str) -> int:
        _ensure_safe_id(logical_run_id, label="logical_run_id")
        if not self._table_exists("run_attempts"):
            return 0
        row = self._query_one(
            "SELECT COUNT(*) AS cnt FROM run_attempts WHERE logical_run_id = ?",
            (logical_run_id,),
        )
        return int(row["cnt"]) if row else 0

    def list_studies(
        self,
        *,
        experiment_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        if not self._table_exists("hpo_studies"):
            return []
        clauses: list[str] = []
        params: list[Any] = []
        if experiment_id:
            clauses.append("experiment_id = ?")
            params.append(experiment_id)
        if status:
            clauses.append("status = ?")
            params.append(status)
        where_sql = ""
        if clauses:
            where_sql = " WHERE " + " AND ".join(clauses)
        rows = self._query(
            "SELECT * FROM hpo_studies" + where_sql + " ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            tuple(params + [limit, offset]),
        )
        payload: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["config"] = _parse_json_object(item.pop("config_json", None)) or None
            payload.append(item)
        return payload

    def count_studies(self, *, experiment_id: str | None = None) -> int:
        if not self._table_exists("hpo_studies"):
            return 0
        if experiment_id:
            row = self._query_one(
                "SELECT COUNT(*) AS cnt FROM hpo_studies WHERE experiment_id = ?",
                (experiment_id,),
            )
        else:
            row = self._query_one("SELECT COUNT(*) AS cnt FROM hpo_studies")
        return int(row["cnt"]) if row else 0

    def get_study(self, study_id: str) -> dict[str, Any] | None:
        _ensure_safe_id(study_id, label="study_id")
        if not self._table_exists("hpo_studies"):
            return None
        row = self._query_one("SELECT * FROM hpo_studies WHERE study_id = ?", (study_id,))
        if row is None:
            return None
        item = dict(row)
        item["config"] = _parse_json_object(item.pop("config_json", None)) or None
        return item

    def get_study_trials(self, study_id: str) -> list[dict[str, Any]] | None:
        study = self.get_study(study_id)
        if not study:
            return None
        storage_path = study.get("storage_path")
        if not isinstance(storage_path, str) or not storage_path:
            return None
        trials_path = Path(storage_path) / "trials_live.parquet"
        if trials_path.exists():
            try:
                return _read_parquet_records_cached(str(trials_path), trials_path.stat().st_mtime_ns)
            except Exception:
                logger.debug("Failed parquet read for study trials: %s", trials_path)
        return None

    def get_experiment_studies(self, experiment_id: str) -> list[dict[str, Any]]:
        _ensure_safe_id(experiment_id, label="experiment_id")
        return self.list_studies(experiment_id=experiment_id, status=None, limit=500, offset=0)

    def _ensemble_components_map(self, ensemble_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
        if not ensemble_ids or not self._table_exists("ensemble_components"):
            return {}
        placeholders = ",".join("?" for _ in ensemble_ids)
        rows = self._query(
            "SELECT ensemble_id, run_id, weight, rank FROM ensemble_components "
            f"WHERE ensemble_id IN ({placeholders}) ORDER BY rank ASC",
            tuple(ensemble_ids),
        )
        payload: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            ensemble_id = row["ensemble_id"]
            if not isinstance(ensemble_id, str):
                continue
            payload.setdefault(ensemble_id, []).append(dict(row))
        return payload

    def _ensemble_metrics_map(self, ensemble_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not ensemble_ids or not self._table_exists("ensemble_metrics"):
            return {}
        placeholders = ",".join("?" for _ in ensemble_ids)
        rows = self._query(
            f"SELECT ensemble_id, name, value FROM ensemble_metrics WHERE ensemble_id IN ({placeholders})",
            tuple(ensemble_ids),
        )
        payload: dict[str, dict[str, Any]] = {}
        for row in rows:
            ensemble_id = row["ensemble_id"]
            name = row["name"]
            if not isinstance(ensemble_id, str) or not isinstance(name, str):
                continue
            payload.setdefault(ensemble_id, {})[name] = _sanitize_metric_value(row["value"])
        for ensemble_id, raw_metrics in list(payload.items()):
            payload[ensemble_id] = {
                **raw_metrics,
                **_normalize_round_metrics(raw_metrics),
            }
        return payload

    def list_ensembles(
        self,
        *,
        experiment_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        if not self._table_exists("ensembles"):
            return []
        if experiment_id:
            rows = self._query(
                "SELECT * FROM ensembles WHERE experiment_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (experiment_id, limit, offset),
            )
        else:
            rows = self._query(
                "SELECT * FROM ensembles ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )

        ensemble_ids = [row["ensemble_id"] for row in rows if isinstance(row["ensemble_id"], str)]
        components_map = self._ensemble_components_map(ensemble_ids)
        metrics_map = self._ensemble_metrics_map(ensemble_ids)

        payload: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            ensemble_id = item.get("ensemble_id")
            if not isinstance(ensemble_id, str):
                continue
            item["config"] = _parse_json_object(item.pop("config_json", None)) or None
            item["components"] = components_map.get(ensemble_id, [])
            item["metrics"] = metrics_map.get(ensemble_id, {})
            payload.append(item)
        return payload

    def count_ensembles(self, *, experiment_id: str | None = None) -> int:
        if not self._table_exists("ensembles"):
            return 0
        if experiment_id:
            row = self._query_one(
                "SELECT COUNT(*) AS cnt FROM ensembles WHERE experiment_id = ?",
                (experiment_id,),
            )
        else:
            row = self._query_one("SELECT COUNT(*) AS cnt FROM ensembles")
        return int(row["cnt"]) if row else 0

    def get_ensemble(self, ensemble_id: str) -> dict[str, Any] | None:
        _ensure_safe_id(ensemble_id, label="ensemble_id")
        if not self._table_exists("ensembles"):
            return None
        row = self._query_one("SELECT * FROM ensembles WHERE ensemble_id = ?", (ensemble_id,))
        if row is None:
            return None
        item = dict(row)
        item["config"] = _parse_json_object(item.pop("config_json", None)) or None
        item["components"] = self._ensemble_components_map([ensemble_id]).get(ensemble_id, [])
        item["metrics"] = self._ensemble_metrics_map([ensemble_id]).get(ensemble_id, {})
        return item

    def _resolve_ensemble_artifacts_dir(self, ensemble: dict[str, Any]) -> Path | None:
        artifacts_path = ensemble.get("artifacts_path")
        if not isinstance(artifacts_path, str) or not artifacts_path.strip():
            return None

        candidate = Path(artifacts_path).expanduser()
        try:
            resolved = candidate.resolve() if candidate.is_absolute() else (self.store_root / candidate).resolve()
        except OSError:
            return None

        if not self._in_store(resolved):
            logger.warning("Ignoring ensemble artifacts path outside store root: %s", resolved)
            return None
        if not resolved.exists() or not resolved.is_dir():
            return None
        return resolved

    def _read_ensemble_parquet_records(self, artifacts_dir: Path, filename: str) -> list[dict[str, Any]] | None:
        path = artifacts_dir / filename
        if not path.exists() or not path.is_file():
            return None
        return _read_parquet_records_cached(str(path), path.stat().st_mtime_ns)

    def _read_ensemble_json(self, artifacts_dir: Path, filename: str) -> dict[str, Any] | None:
        path = artifacts_dir / filename
        if not path.exists() or not path.is_file():
            return None
        payload = _read_json_dict(path)
        return payload if payload else None

    def get_ensemble_correlations(self, ensemble_id: str) -> dict[str, Any] | None:
        ensemble = self.get_ensemble(ensemble_id)
        if not ensemble:
            return None
        artifacts_dir = self._resolve_ensemble_artifacts_dir(ensemble)
        if artifacts_dir is None:
            return None
        corr_path = artifacts_dir / "correlation_matrix.parquet"
        if not corr_path.exists():
            return None
        labels, matrix = _read_parquet_matrix_cached(str(corr_path), corr_path.stat().st_mtime_ns)
        return {
            "labels": labels,
            "matrix": matrix,
        }

    def get_ensemble_artifacts(self, ensemble_id: str) -> dict[str, Any] | None:
        ensemble = self.get_ensemble(ensemble_id)
        if not ensemble:
            return None
        artifacts_dir = self._resolve_ensemble_artifacts_dir(ensemble)
        if artifacts_dir is None:
            return None

        component_predictions_path = artifacts_dir / "component_predictions.parquet"
        weights = self._read_ensemble_parquet_records(artifacts_dir, "weights.parquet")
        component_metrics = self._read_ensemble_parquet_records(artifacts_dir, "component_metrics.parquet")
        era_metrics = self._read_ensemble_parquet_records(artifacts_dir, "era_metrics.parquet")
        regime_metrics = self._read_ensemble_parquet_records(artifacts_dir, "regime_metrics.parquet")
        lineage = self._read_ensemble_json(artifacts_dir, "lineage.json")
        bootstrap_metrics = self._read_ensemble_json(artifacts_dir, "bootstrap_metrics.json")
        available_files = sorted(path.name for path in artifacts_dir.iterdir() if path.is_file())

        return {
            "weights": weights,
            "component_metrics": component_metrics,
            "era_metrics": era_metrics,
            "regime_metrics": regime_metrics,
            "lineage": lineage,
            "bootstrap_metrics": bootstrap_metrics,
            "heavy_component_predictions_available": component_predictions_path.is_file(),
            "available_files": available_files,
        }

    def get_experiment_ensembles(self, experiment_id: str) -> list[dict[str, Any]]:
        _ensure_safe_id(experiment_id, label="experiment_id")
        return self.list_ensembles(experiment_id=experiment_id, limit=500, offset=0)

    def get_runpod_pods(self) -> dict[str, Any]:
        # Phase 1 intentionally does not integrate cloud runtime controls.
        return {
            "items": [],
            "total": 0,
            "error": "RunPod client unavailable in this environment.",
        }

    def _doc_root_numerai(self) -> Path:
        return self.config.numerai_docs_root

    def _doc_root_numereng(self) -> Path:
        return self.config.numereng_docs_root

    def _docs_shared_assets_root(self) -> Path:
        return self.config.shared_docs_assets_root

    def _doc_root_for_domain(self, domain: str) -> Path:
        if domain == "numerai":
            return self._doc_root_numerai()
        if domain == "numereng":
            return self._doc_root_numereng()
        raise ValueError(f"Unknown docs domain: {domain}")

    def _resolve_within_root(self, root: Path, relative_path: str) -> Path:
        root_resolved = root.resolve()
        target = (root / relative_path).resolve()
        if not target.is_relative_to(root_resolved):
            raise ValueError("Path traversal not allowed")
        return target

    def _parse_summary_tree(self, summary_text: str, *, base_dir: str = "") -> list[dict[str, Any]]:
        sections: list[dict[str, Any]] = []
        current_heading: str | None = None
        current_items: list[dict[str, Any]] = []
        link_re = re.compile(r"^\s*\*\s+\[(.+?)\]\((.+?)\)\s*$")
        heading_re = re.compile(r"^##\s+(.+)$")

        for line in summary_text.splitlines():
            heading = heading_re.match(line)
            if heading:
                if current_heading and current_items:
                    sections.append({"heading": current_heading, "items": current_items})
                current_heading = heading.group(1).strip()
                current_items = []
                continue

            link = link_re.match(line)
            if not link:
                continue
            title = link.group(1).strip()
            href = link.group(2)
            resolved = _resolve_summary_href(base_dir, href)
            if resolved is None:
                continue
            if not resolved.lower().endswith(".md"):
                continue

            indent = len(line) - len(line.lstrip())
            node: dict[str, Any] = {"title": title, "path": resolved}
            if indent >= 2 and current_items:
                current_items[-1].setdefault("children", []).append(node)
            else:
                current_items.append(node)

        if current_heading and current_items:
            sections.append({"heading": current_heading, "items": current_items})
        return sections

    def _fallback_doc_tree(self, root: Path, heading: str) -> dict[str, Any]:
        if not root.exists():
            return {"sections": []}
        items: list[dict[str, Any]] = []
        for path in sorted(root.rglob("*.md")):
            rel = path.relative_to(root).as_posix()
            title = path.stem.replace("_", " ")
            items.append({"title": title, "path": rel})
        return {"sections": [{"heading": heading, "items": items}] if items else []}

    def _append_generated_numerai_forum_section(
        self,
        *,
        sections: list[dict[str, Any]],
        root: Path,
    ) -> list[dict[str, Any]]:
        forum_index_path = root / _FORUM_INDEX_RELATIVE_PATH
        if not forum_index_path.exists():
            return sections
        if any(section.get("heading") == "Forum Archive" for section in sections):
            return sections

        existing_paths: set[str] = set()
        for section in sections:
            items = section.get("items")
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                path = item.get("path")
                if isinstance(path, str):
                    existing_paths.add(path)

        forum_items = self._build_generated_numerai_forum_items(root)
        if not forum_items and _FORUM_INDEX_RELATIVE_PATH in existing_paths:
            return sections
        if not forum_items:
            forum_items = [
                {
                    "title": "Numerai Forum Archive Index",
                    "path": _FORUM_INDEX_RELATIVE_PATH,
                }
            ]

        return [
            *sections,
            {
                "heading": "Forum Archive",
                "items": forum_items,
            },
        ]

    def _build_generated_numerai_forum_items(self, root: Path) -> list[dict[str, Any]]:
        posts_root = root / _FORUM_POSTS_RELATIVE_PATH
        if not posts_root.exists() or not posts_root.is_dir():
            return []

        items: list[dict[str, Any]] = []
        year_dirs = sorted(
            [path for path in posts_root.iterdir() if path.is_dir() and re.fullmatch(r"\d{4}", path.name)],
            key=lambda path: path.name,
        )
        for year_dir in year_dirs:
            year = year_dir.name
            year_index_path = year_dir / "INDEX.md"
            year_node: dict[str, Any] = {
                "title": year,
                "path": (f"{_FORUM_POSTS_RELATIVE_PATH}/{year}/INDEX.md" if year_index_path.exists() else None),
            }
            months: list[dict[str, Any]] = []
            month_dirs = sorted(
                [path for path in year_dir.iterdir() if path.is_dir() and re.fullmatch(r"\d{2}", path.name)],
                key=lambda path: path.name,
            )
            for month_dir in month_dirs:
                month = month_dir.name
                month_index_path = month_dir / "INDEX.md"
                if not month_index_path.exists():
                    continue
                months.append(
                    {
                        "title": f"{year}/{month}",
                        "path": f"{_FORUM_POSTS_RELATIVE_PATH}/{year}/{month}/INDEX.md",
                    }
                )

            if months:
                year_node["children"] = months

            if year_node["path"] is not None or months:
                items.append(year_node)

        return items

    def get_doc_tree(self, domain: str) -> dict[str, Any]:
        root = self._doc_root_for_domain(domain)
        summary_path = root / "SUMMARY.md"
        if summary_path.exists():
            sections = self._parse_summary_tree(summary_path.read_text(), base_dir="")
        else:
            heading = "Numerai" if domain == "numerai" else "Numereng"
            sections = self._fallback_doc_tree(root, heading).get("sections", [])

        normalized_sections = list(sections) if isinstance(sections, list) else []
        if domain == "numerai":
            normalized_sections = self._append_generated_numerai_forum_section(
                sections=normalized_sections,
                root=root,
            )
        return {"sections": normalized_sections}

    def get_doc_content(self, domain: str, path: str) -> dict[str, Any]:
        normalized_path = _normalize_markdown_doc_path(path)
        root = self._doc_root_for_domain(domain)
        if domain == "numerai" and not root.exists():
            return {"content": "", "exists": False, "missing_reason": "docs_not_downloaded"}
        full_path = self._resolve_within_root(root, normalized_path)
        if not full_path.exists() or not full_path.is_file():
            return {"content": "", "exists": False, "missing_reason": "document_not_found"}
        return {"content": full_path.read_text(), "exists": True}

    def get_doc_asset_path(self, domain: str, path: str) -> Path:
        normalized_path = _normalize_relative_path(path, label="doc asset path")
        domain_root = self._doc_root_for_domain(domain)
        docs_root = self._docs_shared_assets_root().parent
        shared_assets_root = self._docs_shared_assets_root()

        candidates: list[Path] = [
            self._resolve_within_root(domain_root, normalized_path),
            self._resolve_within_root(docs_root, normalized_path),
        ]

        if normalized_path.startswith(".gitbook/assets/"):
            shared_rel = normalized_path.removeprefix(".gitbook/assets/")
            if shared_rel:
                candidates.append(self._resolve_within_root(shared_assets_root, shared_rel))
        elif normalized_path.startswith("assets/"):
            shared_rel = normalized_path.removeprefix("assets/")
            if shared_rel:
                candidates.append(self._resolve_within_root(shared_assets_root, shared_rel))

        seen: set[Path] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.is_file():
                return candidate

        raise FileNotFoundError(f"Doc asset not found: {path}")

    def experiment_doc_path(self, experiment_id: str, filename: str) -> Path:
        _ensure_safe_id(experiment_id, label="experiment_id")
        return self._resolve_experiment_dir(experiment_id) / filename

    def run_doc_path(self, run_id: str, filename: str) -> Path:
        _ensure_safe_id(run_id, label="run_id")
        return self._resolve_run_dir(run_id) / filename

    def get_experiment_doc(self, experiment_id: str, filename: str) -> dict[str, Any]:
        path = self.experiment_doc_path(experiment_id, filename)
        if not path.exists():
            return {"content": "", "exists": False}
        return {"content": path.read_text(), "exists": True}

    def get_run_doc(self, run_id: str, filename: str) -> dict[str, Any]:
        path = self.run_doc_path(run_id, filename)
        if not path.exists():
            return {"content": "", "exists": False}
        return {"content": path.read_text(), "exists": True}

    def notes_root(self) -> Path:
        return resolve_workspace_layout_from_store_root(self.store_root).notes_root

    def _scan_notes_tree(self, root: Path, rel: Path | None = None) -> list[dict[str, Any]]:
        base = root if rel is None else root / rel
        if not base.is_dir():
            return []
        entries: list[dict[str, Any]] = []
        for child in sorted(base.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            child_rel = child.relative_to(root)
            if child.is_dir():
                children = self._scan_notes_tree(root, child_rel)
                if children:
                    entries.append(
                        {
                            "title": child.name,
                            "path": str(child_rel),
                            "children": children,
                        }
                    )
            elif child.suffix == ".md":
                if child.stem.upper() in _HIDDEN_NOTE_STEMS:
                    continue
                entries.append({"title": child.stem, "path": str(child_rel)})
        return entries

    def get_notes_tree(self) -> dict[str, Any]:
        root = self.notes_root()
        if not root.exists() or not root.is_dir():
            return {"sections": []}
        sections: list[dict[str, Any]] = []

        root_files: list[dict[str, Any]] = []
        top_level_dirs: list[Path] = []
        try:
            children = list(root.iterdir())
        except OSError:
            return {"sections": []}

        for child in children:
            if child.is_dir():
                top_level_dirs.append(child)
            elif child.is_file() and child.suffix == ".md" and child.stem.upper() not in _HIDDEN_NOTE_STEMS:
                root_files.append({"title": child.stem, "path": str(child.relative_to(root))})

        if root_files:
            root_files.sort(key=lambda item: item["title"].lower())
            sections.append({"heading": "Notes", "items": root_files})

        for directory in sorted(top_level_dirs, key=lambda p: p.name.lower()):
            items = self._scan_notes_tree(root, directory.relative_to(root))
            if items:
                sections.append({"heading": directory.name, "items": items})

        return {"sections": sections}

    def get_note_content(self, path: str) -> dict[str, Any]:
        normalized_path = _normalize_markdown_doc_path(path)
        root = self.notes_root()
        full_path = self._resolve_within_root(root, normalized_path)
        if not full_path.exists() or not full_path.is_file():
            return {"content": "", "exists": False}
        return {"content": full_path.read_text(), "exists": True}
