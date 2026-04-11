"""Filesystem artifact helpers for HPO studies."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from numereng.features.store import resolve_workspace_layout_from_store_root
from numereng.platform.parquet import write_parquet

_SAFE_ID = re.compile(r"^[\w\-.]+$")


def ensure_safe_study_id(study_id: str) -> str:
    """Validate and normalize one explicit study ID."""

    stripped = study_id.strip()
    if not stripped:
        raise ValueError("hpo_study_id_invalid")
    if not _SAFE_ID.match(stripped):
        raise ValueError("hpo_study_id_invalid")
    return stripped


def resolve_study_storage_path(*, store_root: Path, experiment_id: str | None, study_id: str) -> Path:
    """Resolve canonical storage path for one study."""

    safe_study_id = ensure_safe_study_id(study_id)
    if experiment_id:
        root = (
            resolve_workspace_layout_from_store_root(store_root).experiments_root
            / experiment_id
            / "hpo"
            / safe_study_id
        )
    else:
        root = store_root / "hpo" / safe_study_id
    (root / "configs").mkdir(parents=True, exist_ok=True)
    return root


def study_spec_path(*, storage_path: Path) -> Path:
    """Return the immutable study spec path."""

    return storage_path / "study_spec.json"


def study_summary_path(*, storage_path: Path) -> Path:
    """Return the mutable study summary path."""

    return storage_path / "study_summary.json"


def optuna_journal_path(*, storage_path: Path) -> Path:
    """Return the Optuna journal backend path."""

    return storage_path / "optuna_journal.log"


def write_study_spec(*, storage_path: Path, payload: dict[str, Any]) -> Path:
    """Persist one immutable study spec payload."""

    path = study_spec_path(storage_path=storage_path)
    _write_json(path=path, payload=payload)
    return path


def write_study_summary(*, storage_path: Path, payload: dict[str, Any]) -> Path:
    """Persist one mutable study summary payload."""

    path = study_summary_path(storage_path=storage_path)
    _write_json(path=path, payload=payload)
    return path


def read_study_spec(*, storage_path: Path) -> dict[str, Any] | None:
    """Load one immutable study spec payload if present."""

    return _read_json(path=study_spec_path(storage_path=storage_path))


def read_study_summary(*, storage_path: Path) -> dict[str, Any] | None:
    """Load one mutable study summary payload if present."""

    return _read_json(path=study_summary_path(storage_path=storage_path))


def write_trial_config(*, storage_path: Path, trial_number: int, config: dict[str, Any]) -> Path:
    """Persist one materialized trial config to JSON."""

    config_path = storage_path / "configs" / f"trial_{trial_number:04d}.json"
    _write_json(path=config_path, payload=config)
    return config_path


def write_trials_table(*, storage_path: Path, trials: list[dict[str, Any]]) -> None:
    """Persist the current trial summary table in parquet."""

    frame = pd.DataFrame(trials)
    parquet_path = storage_path / "trials_live.parquet"
    write_parquet(frame, parquet_path, index=False)


def _write_json(*, path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(*, path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return {str(key): value for key, value in payload.items()}
