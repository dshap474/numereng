"""Shared parquet IO policy for numereng-managed tabular artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

PARQUET_COMPRESSION = "zstd"
PARQUET_COMPRESSION_LEVEL = 3


def parquet_write_kwargs(*, index: bool) -> dict[str, Any]:
    """Return the canonical numereng parquet write policy."""

    return {
        "index": index,
        "compression": PARQUET_COMPRESSION,
        "compression_level": PARQUET_COMPRESSION_LEVEL,
    }


def write_parquet(frame: pd.DataFrame, path: str | Path, *, index: bool = False) -> Path:
    """Write one parquet file using the canonical numereng compression policy."""

    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(resolved, **parquet_write_kwargs(index=index))
    return resolved
