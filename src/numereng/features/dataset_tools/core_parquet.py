"""Parquet IO helpers for dataset tool scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from numereng.platform.parquet import (
    PARQUET_COMPRESSION,
    PARQUET_COMPRESSION_LEVEL,
)
from numereng.platform.parquet import (
    write_parquet as write_parquet_file,
)


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
    compression: str = PARQUET_COMPRESSION,
    compression_level: int | None = PARQUET_COMPRESSION_LEVEL,
) -> Path:
    """Write one parquet file with deterministic compression defaults."""
    if compression != PARQUET_COMPRESSION or compression_level != PARQUET_COMPRESSION_LEVEL:
        raise ValueError("parquet_write_policy_override_not_supported")
    try:
        return write_parquet_file(frame, path, index=index)
    except Exception as exc:  # pragma: no cover - thin IO wrapper
        raise ValueError(f"parquet_write_failed:{Path(path).expanduser().resolve()}") from exc


def dump_json_stdout(payload: dict[str, Any]) -> None:
    """Print one JSON payload to stdout for script consumers."""
    print(json.dumps(payload, sort_keys=True))
