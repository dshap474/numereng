"""Filesystem artifact helpers for HPO studies."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from numereng.platform.parquet import write_parquet

_SAFE_ID = re.compile(r"^[\w\-.]+$")


def build_study_id(*, study_name: str, experiment_id: str | None) -> str:
    """Build a stable-safe study ID."""

    slug = _slug(study_name)
    scope = _slug(experiment_id) if experiment_id else "global"
    stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    study_id = f"{scope}-{slug}-{stamp}"
    if not _SAFE_ID.match(study_id):
        raise ValueError("hpo_study_id_invalid")
    return study_id


def resolve_study_storage_path(*, store_root: Path, experiment_id: str | None, study_id: str) -> Path:
    """Resolve canonical storage path for one study."""

    if experiment_id:
        root = store_root / "experiments" / experiment_id / "hpo" / study_id
    else:
        root = store_root / "hpo" / study_id
    root.mkdir(parents=True, exist_ok=True)
    (root / "configs").mkdir(parents=True, exist_ok=True)
    return root


def write_trial_config(*, storage_path: Path, trial_number: int, config: dict[str, Any]) -> Path:
    """Persist one materialized trial config to JSON."""

    config_path = storage_path / "configs" / f"trial_{trial_number:04d}.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    return config_path


def write_trials_table(*, storage_path: Path, trials: list[dict[str, Any]]) -> None:
    """Persist trial summary table in parquet."""

    frame = pd.DataFrame(trials)
    parquet_path = storage_path / "trials_live.parquet"
    write_parquet(frame, parquet_path, index=False)


def _slug(value: str | None) -> str:
    raw = (value or "").strip().lower()
    if not raw:
        return "study"
    chars = []
    for ch in raw:
        if ch.isalnum() or ch in {"-", "_", "."}:
            chars.append(ch)
        elif ch in {" ", "/", "\\"}:
            chars.append("-")
    text = "".join(chars).strip("-._")
    return text or "study"
