"""Parquet IO helpers for dataset tool scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_parquet(path: str | Path, *, columns: list[str] | None = None) -> pd.DataFrame:
    """Read one parquet file and raise explicit validation errors on failure."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise ValueError(f"input_path_not_found:{resolved}")
    try:
        return pd.read_parquet(resolved, columns=columns)
    except Exception as exc:  # pragma: no cover - thin IO wrapper
        raise ValueError(f"parquet_read_failed:{resolved}") from exc


def write_parquet(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    index: bool = False,
    compression: str = "zstd",
    compression_level: int | None = 3,
) -> Path:
    """Write one parquet file with deterministic compression defaults."""
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {
        "index": index,
        "compression": compression,
    }
    if compression_level is not None:
        kwargs["compression_level"] = int(compression_level)
    try:
        frame.to_parquet(resolved, **kwargs)
    except Exception as exc:  # pragma: no cover - thin IO wrapper
        raise ValueError(f"parquet_write_failed:{resolved}") from exc
    return resolved


def dump_json_stdout(payload: dict[str, Any]) -> None:
    """Print one JSON payload to stdout for script consumers."""
    print(json.dumps(payload, sort_keys=True))
