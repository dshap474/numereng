"""Store adapter for the viz compatibility API.

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
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from numbers import Integral, Real
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_SAFE_ID = re.compile(r"^[\w\-.]+$")
_ROUND_CONFIG_STEM_RE = re.compile(r"^(r\d+)_\d+_(.+)$")
_ROUND_INDEX_RE = re.compile(r"^r(\d+)_")
_FORUM_INDEX_RELATIVE_PATH = "forum/INDEX.md"
_FORUM_POSTS_RELATIVE_PATH = "forum/posts"

_TERMINAL_JOB_STATUSES = {"completed", "failed", "canceled", "stale"}
_ACTIVE_JOB_STATUSES = {"queued", "starting", "running", "canceling"}
_HIDDEN_NOTE_STEMS = {"CLAUDE", "AGENTS"}
_CLASSIC_PAYOUT_TARGET = "target_ender_20"
_PAYOUT_CORR_WEIGHT = 0.75
_PAYOUT_MMC_WEIGHT = 2.25
_PAYOUT_CLIP = 0.05
_METRIC_QUERY_ALIASES: dict[str, tuple[str, ...]] = {
    "corr20v2_mean": ("corr20v2_mean", "corr.mean", "corr_mean"),
    "corr20v2_sharpe": ("corr20v2_sharpe", "corr.sharpe", "corr_sharpe", "sharpe"),
    "mmc_mean": ("mmc_mean", "mmc.mean"),
    "payout_estimate_mean": ("payout_estimate_mean", "payout_score"),
    "bmc_mean": ("bmc_mean", "bmc.mean"),
}
_DERIVED_METRIC_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "payout_estimate_mean": ("corr20v2_mean", "mmc_mean"),
}


def _is_classic_payout_target(target_col: str | None) -> bool:
    return target_col == _CLASSIC_PAYOUT_TARGET


@lru_cache(maxsize=512)
def _read_csv_records_cached(path_str: str, mtime_ns: int, top_n: int = 0) -> list[dict[str, Any]]:
    """Read CSV records with optional top-N filtering and mtime-keyed cache."""

    _ = mtime_ns
    frame = pd.read_csv(path_str)
    if top_n > 0:
        numeric_columns = frame.select_dtypes(include="number").columns
        if len(numeric_columns) > 0:
            frame = frame.nlargest(top_n, numeric_columns[-1])
    return _frame_to_records(frame)


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


@lru_cache(maxsize=32)
def _read_csv_frame_cached(
    path_str: str,
    mtime_ns: int,
    columns: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Read csv frame with optional column subset from mtime-keyed cache."""

    _ = mtime_ns
    return pd.read_csv(path_str, usecols=list(columns) if columns else None)


@lru_cache(maxsize=128)
def _read_csv_matrix_cached(path_str: str, mtime_ns: int) -> tuple[list[str], list[list[float | None]]]:
    """Read correlation matrix CSV into labels + matrix payload."""

    _ = mtime_ns
    frame = pd.read_csv(path_str, index_col=0)
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
    target_train = (
        _first_present_str(data_info, ("target_train", "target_col", "target"))
        or _first_present_str(manifest, ("target_col", "target_train", "target"))
    )
    target_payout = _first_present_str(data_info, ("target_payout",))
    target = target_payout or target_train
    return target_train, target_payout, target


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
    """Flatten nested metric payloads into dotted keys for lookup compatibility."""

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

    requested: list[str] = []
    requested_seen: set[str] = set()
    for name in metric_names:
        if name not in requested_seen:
            requested_seen.add(name)
            requested.append(name)
        for dependency in _DERIVED_METRIC_DEPENDENCIES.get(name, ()):
            if dependency in requested_seen:
                continue
            requested_seen.add(dependency)
            requested.append(dependency)

    expanded: list[str] = []
    seen: set[str] = set()
    for name in requested:
        aliases = _METRIC_QUERY_ALIASES.get(name, (name,))
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
    normalized: dict[str, Any] = _sanitize_metrics(metrics)
    flattened = _flatten_metrics(normalized)
    for key, value in flattened.items():
        normalized.setdefault(key, value)

    corr_mean = _extract_numeric_metric(normalized, "corr20v2_mean", "corr.mean", "corr_mean")
    if corr_mean is not None:
        normalized["corr20v2_mean"] = corr_mean

    corr_std = _extract_numeric_metric(normalized, "corr20v2_std", "corr.std", "corr_std")
    if corr_std is not None:
        normalized["corr20v2_std"] = corr_std

    corr_sharpe = _extract_numeric_metric(normalized, "corr20v2_sharpe", "corr.sharpe", "corr_sharpe", "sharpe")
    if corr_sharpe is not None:
        normalized["corr20v2_sharpe"] = corr_sharpe

    corr_n_eras = _extract_numeric_metric(normalized, "corr20v2_n_eras", "corr.n_eras", "n_eras")
    if corr_n_eras is not None:
        normalized["corr20v2_n_eras"] = int(corr_n_eras)

    mmc_mean = _extract_numeric_metric(normalized, "mmc_mean", "mmc.mean")
    if mmc_mean is not None:
        normalized["mmc_mean"] = mmc_mean

    mmc_std = _extract_numeric_metric(normalized, "mmc_std", "mmc.std")
    if mmc_std is not None:
        normalized["mmc_std"] = mmc_std

    mmc_sharpe = _extract_numeric_metric(normalized, "mmc_sharpe", "mmc.sharpe")
    if mmc_sharpe is not None:
        normalized["mmc_sharpe"] = mmc_sharpe

    mmc_n_eras = _extract_numeric_metric(normalized, "mmc_n_eras", "mmc.n_eras")
    if mmc_n_eras is not None:
        normalized["mmc_n_eras"] = int(mmc_n_eras)

    bmc_mean = _extract_numeric_metric(normalized, "bmc_mean", "bmc.mean")
    if bmc_mean is not None:
        normalized["bmc_mean"] = bmc_mean

    bmc_std = _extract_numeric_metric(normalized, "bmc_std", "bmc.std")
    if bmc_std is not None:
        normalized["bmc_std"] = bmc_std

    bmc_sharpe = _extract_numeric_metric(normalized, "bmc_sharpe", "bmc.sharpe")
    if bmc_sharpe is not None:
        normalized["bmc_sharpe"] = bmc_sharpe

    bmc_n_eras = _extract_numeric_metric(normalized, "bmc_n_eras", "bmc.n_eras")
    if bmc_n_eras is not None:
        normalized["bmc_n_eras"] = int(bmc_n_eras)

    bmc_last_200_eras_mean = _extract_numeric_metric(
        normalized,
        "bmc_last_200_eras_mean",
        "bmc_last_200_eras.mean",
    )
    if bmc_last_200_eras_mean is not None:
        normalized["bmc_last_200_eras_mean"] = bmc_last_200_eras_mean

    cwmm_mean = _extract_numeric_metric(normalized, "cwmm_mean", "cwmm.mean")
    if cwmm_mean is not None:
        normalized["cwmm_mean"] = cwmm_mean

    cwmm_std = _extract_numeric_metric(normalized, "cwmm_std", "cwmm.std")
    if cwmm_std is not None:
        normalized["cwmm_std"] = cwmm_std

    cwmm_sharpe = _extract_numeric_metric(normalized, "cwmm_sharpe", "cwmm.sharpe")
    if cwmm_sharpe is not None:
        normalized["cwmm_sharpe"] = cwmm_sharpe

    payout = _extract_numeric_metric(normalized, "payout_estimate_mean", "payout_score")
    payout_key_present = any(
        key in normalized for key in ("payout_estimate_mean", "payout_score", "payout_estimate", "payout_estimate.mean")
    )
    can_derive_payout = target_col is None or _is_classic_payout_target(target_col)
    if (
        payout is None
        and not payout_key_present
        and can_derive_payout
        and corr_mean is not None
        and mmc_mean is not None
    ):
        payout_raw = (_PAYOUT_CORR_WEIGHT * corr_mean) + (_PAYOUT_MMC_WEIGHT * mmc_mean)
        payout = max(-_PAYOUT_CLIP, min(_PAYOUT_CLIP, payout_raw))
    if payout is not None:
        normalized["payout_estimate_mean"] = payout
    elif payout_key_present:
        normalized.setdefault("payout_estimate_mean", None)

    max_drawdown = _extract_numeric_metric(normalized, "max_drawdown", "corr.max_drawdown")
    if max_drawdown is not None:
        normalized["max_drawdown"] = max_drawdown

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

    return Path(__file__).resolve().parents[4]


def resolve_store_root(explicit: str | Path | None = None) -> Path:
    """Resolve store root using explicit value, env, then repo default."""

    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    env_root = os.getenv("NUMERENG_STORE_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return (repository_root() / ".numereng").resolve()


@dataclass(frozen=True)
class VizStoreConfig:
    """Configuration for viz store adapter."""

    store_root: Path
    repo_root: Path

    @classmethod
    def from_env(
        cls,
        *,
        store_root: str | Path | None = None,
    ) -> VizStoreConfig:
        return cls(
            store_root=resolve_store_root(store_root),
            repo_root=repository_root(),
        )


class VizStoreAdapter:
    """Read-oriented adapter over `.numereng` SQLite + artifact filesystem."""

    def __init__(self, config: VizStoreConfig) -> None:
        self.config = config
        self.store_root = config.store_root
        self.repo_root = config.repo_root
        self.db_path = self.store_root / "numereng.db"
        self._lock = threading.RLock()
        self._conn = self._connect_read_only()

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
        row = self._query_one(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
            (name,),
        )
        return row is not None

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

    def _iter_experiment_ids(self) -> list[str]:
        experiment_dir = self.store_root / "experiments"
        ids: set[str] = set()
        if experiment_dir.exists():
            for child in experiment_dir.iterdir():
                if child.is_dir() and not child.name.startswith("."):
                    ids.add(child.name)
        if self._table_exists("experiments"):
            rows = self._query("SELECT experiment_id FROM experiments")
            for row in rows:
                value = row["experiment_id"]
                if isinstance(value, str) and value:
                    ids.add(value)
        return sorted(ids)

    def _iter_configs(self, experiment_id: str) -> list[Path]:
        root = self.store_root / "experiments" / experiment_id
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
        rel = path.relative_to(self.store_root).as_posix()
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
        exp_ids = [experiment_id] if experiment_id else self._iter_experiment_ids()
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
                run_count = counts.get(exp_id, 0) if isinstance(exp_id, str) else 0
                payload.append(self._format_experiment(row_dict, run_count=run_count))
            return payload

        # Filesystem fallback when DB is absent.
        items: list[dict[str, Any]] = []
        for exp_id in self._iter_experiment_ids():
            exp_dir = self.store_root / "experiments" / exp_id
            created_at = None
            updated_at = None
            if exp_dir.exists():
                stat = exp_dir.stat()
                created_at = datetime.fromtimestamp(stat.st_ctime, tz=UTC).isoformat()
                updated_at = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()
            items.append(
                {
                    "experiment_id": exp_id,
                    "name": exp_id,
                    "status": "draft",
                    "created_at": created_at,
                    "updated_at": updated_at,
                    "preset": None,
                    "hypothesis": None,
                    "tags": [],
                    "champion_run_id": None,
                    "run_count": 0,
                    "metadata": {},
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
                payload = self._format_experiment(dict(row), run_count=run_count)
                payload["study_count"] = self.count_studies(experiment_id=experiment_id)
                payload["ensemble_count"] = self.count_ensembles(experiment_id=experiment_id)
                return payload

        exp_dir = self.store_root / "experiments" / experiment_id
        if exp_dir.exists() and exp_dir.is_dir():
            created_at = datetime.fromtimestamp(exp_dir.stat().st_ctime, tz=UTC).isoformat()
            updated_at = datetime.fromtimestamp(exp_dir.stat().st_mtime, tz=UTC).isoformat()
            return {
                "experiment_id": experiment_id,
                "name": experiment_id,
                "status": "draft",
                "created_at": created_at,
                "updated_at": updated_at,
                "preset": None,
                "hypothesis": None,
                "tags": [],
                "champion_run_id": None,
                "run_count": 0,
                "metadata": {},
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
            manifest = _read_json_dict(self.store_root / "runs" / run_id / "run.json")
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
        model_raw = manifest.get("model")
        model_info: dict[str, Any] = model_raw if isinstance(model_raw, dict) else {}
        data_raw = manifest.get("data")
        data_info: dict[str, Any] = data_raw if isinstance(data_raw, dict) else {}
        lineage_raw = manifest.get("lineage")
        lineage: dict[str, Any] = lineage_raw if isinstance(lineage_raw, dict) else {}
        target_train, target_payout, target = _resolve_run_targets(manifest)
        run_id = row.get("run_id")
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
            payload: list[dict[str, Any]] = []
            for row in rows:
                row_dict = dict(row)
                run_id = row_dict.get("run_id")
                if not isinstance(run_id, str):
                    continue
                payload.append(
                    self._format_run(
                        row_dict,
                        champion_run_id=champion_run_id,
                        metrics=metrics_map.get(run_id, {}),
                    )
                )
            if payload:
                return payload

        # Filesystem fallback.
        payload = []
        runs_dir = self.store_root / "runs"
        if not runs_dir.exists():
            return payload
        for run_dir in sorted(runs_dir.iterdir(), key=lambda p: p.name):
            if not run_dir.is_dir():
                continue
            manifest_path = run_dir / "run.json"
            if not manifest_path.exists():
                continue
            try:
                payload_raw = json.loads(manifest_path.read_text())
            except json.JSONDecodeError:
                continue
            if not isinstance(payload_raw, dict):
                continue
            lineage_raw = payload_raw.get("lineage")
            lineage: dict[str, Any] = lineage_raw if isinstance(lineage_raw, dict) else {}
            exp_id = payload_raw.get("experiment_id") or lineage.get("experiment_id")
            if exp_id != experiment_id:
                continue
            run_id = payload_raw.get("run_id")
            if not isinstance(run_id, str):
                continue
            metrics = {}
            metrics_path = run_dir / "metrics.json"
            if metrics_path.exists():
                try:
                    metric_payload = json.loads(metrics_path.read_text())
                except json.JSONDecodeError:
                    metric_payload = {}
                if isinstance(metric_payload, dict):
                    metrics = metric_payload
            model_raw = payload_raw.get("model")
            model_info: dict[str, Any] = model_raw if isinstance(model_raw, dict) else {}
            data_raw = payload_raw.get("data")
            data_info: dict[str, Any] = data_raw if isinstance(data_raw, dict) else {}
            target_train, target_payout, target = _resolve_run_targets(payload_raw)
            payload.append(
                {
                    "run_id": run_id,
                    "experiment_id": experiment_id,
                    "created_at": payload_raw.get("created_at"),
                    "status": payload_raw.get("status"),
                    "is_champion": bool(champion_run_id and champion_run_id == run_id),
                    "metrics": _normalize_round_metrics(metrics, target_col=target_train),
                    "config_hash": payload_raw.get("config_hash"),
                    "notes": None,
                    "round_id": lineage.get("round_id"),
                    "round_index": lineage.get("round_index"),
                    "sweep_dimension": lineage.get("sweep_dimension"),
                    "run_name": payload_raw.get("run_name"),
                    "model_type": model_info.get("type") or payload_raw.get("model_type"),
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
        exp_dir = self.store_root / "experiments" / experiment_id
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
                source_file = metrics_path.relative_to(self.store_root).as_posix()
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
        if self._table_exists("runs"):
            row = self._query_one("SELECT run_path FROM runs WHERE run_id = ?", (run_id,))
            if row is not None:
                run_path = row["run_path"]
                if isinstance(run_path, str) and run_path:
                    path = Path(run_path)
                    if path.exists() and path.is_dir():
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
                        return path

        return self.store_root / "runs" / run_id

    def _read_artifact_records(self, *, parquet_path: Path, csv_path: Path) -> list[dict[str, Any]] | None:
        if parquet_path.exists():
            try:
                return _read_parquet_records_cached(str(parquet_path), parquet_path.stat().st_mtime_ns)
            except Exception:
                logger.debug("Failed parquet read for %s; falling back to CSV", parquet_path)
        if csv_path.exists():
            return _read_csv_records_cached(str(csv_path), csv_path.stat().st_mtime_ns, 0)
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
            if suffix == ".csv":
                return _read_csv_frame_cached(str(path), path.stat().st_mtime_ns, normalized_columns)
        except Exception:
            logger.debug("Failed table read for %s", path, exc_info=True)
            return None
        return None

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
        csv_files = sorted(artifacts_predictions.glob("*.csv"))
        if csv_files:
            return csv_files[0]
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
                    "corr20v2": score,
                }
            )
        return _sort_records_by_era(records) if records else None

    def get_per_era_corr(self, run_id: str) -> list[dict[str, Any]] | None:
        _ensure_safe_id(run_id, label="run_id")
        run_dir = self._resolve_run_dir(run_id)
        parquet_path = run_dir / "artifacts" / "predictions" / "val_per_era_corr20v2.parquet"
        csv_path = run_dir / "artifacts" / "predictions" / "val_per_era_corr20v2.csv"
        from_artifact = self._read_artifact_records(parquet_path=parquet_path, csv_path=csv_path)
        if from_artifact is not None:
            return from_artifact
        return self._derive_per_era_corr_fallback(run_id, run_dir)

    def _derive_per_era_payout_map_fallback(self, run_id: str, run_dir: Path) -> list[dict[str, Any]] | None:
        correlation_contribution, numerai_corr = _load_scoring_functions()
        if correlation_contribution is None or numerai_corr is None:
            return None

        loaded = self._scoring_context(run_id, run_dir)
        if loaded is None:
            return None
        frame, context = loaded
        era_col = str(context["era_col"])
        target_col = str(context["target_col"])
        if not _is_classic_payout_target(target_col):
            return None
        prediction_col = str(context["prediction_col"])
        id_col_raw = context.get("id_col")
        if not isinstance(id_col_raw, str) or id_col_raw not in frame.columns:
            return None
        id_col = id_col_raw

        score_provenance = context["score_provenance"]
        manifest = context["manifest"]
        meta_model_col = str(context["meta_model_col"])
        meta_model_path = self._resolve_meta_model_artifact_path(run_dir, manifest, score_provenance)
        if meta_model_path is None:
            return None

        meta_columns = tuple(dict.fromkeys([id_col, meta_model_col, era_col]))
        meta_frame = self._read_table_frame(meta_model_path, columns=meta_columns)
        if meta_frame is None:
            # Some meta model snapshots do not include era; retry with id + meta column only.
            meta_frame = self._read_table_frame(meta_model_path, columns=(id_col, meta_model_col))
        if meta_frame is None or meta_frame.empty:
            return None
        if meta_model_col not in meta_frame.columns:
            fallback = [str(col) for col in meta_frame.columns if str(col) not in {id_col, era_col, "data_type"}]
            if not fallback:
                return None
            meta_model_col = fallback[0]

        if id_col not in meta_frame.columns:
            if meta_frame.index.name == id_col:
                meta_frame = meta_frame.reset_index()
            elif meta_frame.index.name is None and not isinstance(meta_frame.index, pd.RangeIndex):
                meta_frame = meta_frame.reset_index().rename(columns={"index": id_col})
        if id_col not in meta_frame.columns:
            return None

        filtered_preds = frame.dropna(subset=[era_col, id_col, target_col, prediction_col])
        if filtered_preds.empty:
            return None

        merge_keys = [id_col]
        if era_col in meta_frame.columns:
            merge_keys.append(era_col)
        meta_keep = [id_col, meta_model_col]
        if era_col in meta_frame.columns:
            meta_keep.append(era_col)
        merged = filtered_preds.merge(meta_frame[meta_keep], on=merge_keys, how="inner")
        merged = merged.dropna(subset=[era_col, target_col, prediction_col, meta_model_col])
        if merged.empty:
            return None

        records: list[dict[str, Any]] = []
        for era, group in merged.groupby(era_col):
            corr_raw = numerai_corr(group[[prediction_col]], group[target_col])
            mmc_raw = correlation_contribution(group[[prediction_col]], group[meta_model_col], group[target_col])
            corr = _resolve_series_value(corr_raw, prediction_col)
            mmc = _resolve_series_value(mmc_raw, prediction_col)
            if corr is None or mmc is None:
                continue
            payout_raw = (_PAYOUT_CORR_WEIGHT * corr) + (_PAYOUT_MMC_WEIGHT * mmc)
            payout = max(-_PAYOUT_CLIP, min(_PAYOUT_CLIP, payout_raw))
            records.append(
                {
                    "era": _coerce_json_value(era),
                    "corr20v2": corr,
                    "mmc": mmc,
                    "payout_estimate": payout,
                }
            )
        return _sort_records_by_era(records) if records else None

    def get_per_era_payout_map(self, run_id: str) -> list[dict[str, Any]] | None:
        _ensure_safe_id(run_id, label="run_id")
        run_dir = self._resolve_run_dir(run_id)
        manifest = self.get_run_manifest(run_id) or _read_json_dict(run_dir / "run.json")
        target_train, _, _ = _resolve_run_targets(manifest)
        if not _is_classic_payout_target(target_train):
            return None
        parquet_path = run_dir / "artifacts" / "predictions" / "val_per_era_payout_map.parquet"
        csv_path = run_dir / "artifacts" / "predictions" / "val_per_era_payout_map.csv"
        from_artifact = self._read_artifact_records(parquet_path=parquet_path, csv_path=csv_path)
        if from_artifact is not None:
            return from_artifact
        return self._derive_per_era_payout_map_fallback(run_id, run_dir)

    def get_feature_importance(self, run_id: str, top_n: int) -> list[dict[str, Any]] | None:
        _ensure_safe_id(run_id, label="run_id")
        run_dir = self._resolve_run_dir(run_id)
        path = run_dir / "artifacts" / "eval" / "feature_importance.csv"
        if not path.exists():
            return None
        return _read_csv_records_cached(str(path), path.stat().st_mtime_ns, max(0, top_n))

    def get_trials(self, run_id: str) -> list[dict[str, Any]] | None:
        _ensure_safe_id(run_id, label="run_id")
        run_dir = self._resolve_run_dir(run_id)
        path = run_dir / "artifacts" / "reports" / "trials.csv"
        if not path.exists():
            return None
        return _read_csv_records_cached(str(path), path.stat().st_mtime_ns, 0)

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
            payload["columns"] = {
                str(key): _coerce_json_value(value)
                for key, value in columns_raw.items()
            }
        joins_raw = provenance.get("joins")
        if isinstance(joins_raw, dict):
            payload["joins"] = {
                str(key): _coerce_json_value(value)
                for key, value in joins_raw.items()
            }

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
            "SELECT * FROM logical_runs"
            + where_sql
            + " ORDER BY updated_at DESC LIMIT ? OFFSET ?",
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
        csv_path = Path(storage_path) / "trials_live.csv"
        if csv_path.exists():
            return _read_csv_records_cached(str(csv_path), csv_path.stat().st_mtime_ns, 0)
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

    def _read_ensemble_csv_records(self, artifacts_dir: Path, filename: str) -> list[dict[str, Any]] | None:
        path = artifacts_dir / filename
        if not path.exists() or not path.is_file():
            return None
        return _read_csv_records_cached(str(path), path.stat().st_mtime_ns, 0)

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
        corr_path = artifacts_dir / "correlation_matrix.csv"
        if not corr_path.exists():
            return None
        labels, matrix = _read_csv_matrix_cached(str(corr_path), corr_path.stat().st_mtime_ns)
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
        weights = self._read_ensemble_csv_records(artifacts_dir, "weights.csv")
        component_metrics = self._read_ensemble_csv_records(artifacts_dir, "component_metrics.csv")
        era_metrics = self._read_ensemble_csv_records(artifacts_dir, "era_metrics.csv")
        regime_metrics = self._read_ensemble_csv_records(artifacts_dir, "regime_metrics.csv")
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
        return self.repo_root / "docs" / "numerai"

    def _doc_root_numereng(self) -> Path:
        return self.repo_root / "docs" / "numereng"

    def _docs_shared_assets_root(self) -> Path:
        return self.repo_root / "docs" / "assets"

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
                "path": (
                    f"{_FORUM_POSTS_RELATIVE_PATH}/{year}/INDEX.md"
                    if year_index_path.exists()
                    else None
                ),
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
        full_path = self._resolve_within_root(root, normalized_path)
        if not full_path.exists() or not full_path.is_file():
            return {"content": "", "exists": False}
        return {"content": full_path.read_text(), "exists": True}

    def get_doc_asset_path(self, domain: str, path: str) -> Path:
        normalized_path = _normalize_relative_path(path, label="doc asset path")
        domain_root = self._doc_root_for_domain(domain)
        docs_root = self.repo_root / "docs"
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
        return self.store_root / "experiments" / experiment_id / filename

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
        return self.store_root / "notes"

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
