"""IO helpers for feature-neutralization workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

_PREDICTIONS_CANDIDATES = (
    "live_predictions.csv",
    "live_predictions.parquet",
    "predictions.csv",
    "predictions.parquet",
    "val_predictions.parquet",
    "val_predictions.csv",
)


def resolve_predictions_path(predictions_path: str | Path) -> Path:
    """Resolve one explicit predictions path."""

    path = Path(predictions_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"neutralization_predictions_file_not_found:{path}")
    return path


def resolve_run_predictions_path(*, store_root: str | Path, run_id: str) -> Path:
    """Resolve one run-scoped predictions file from store artifacts."""

    root = Path(store_root).expanduser().resolve()
    run_dir = root / "runs" / run_id
    if not run_dir.is_dir():
        raise ValueError(f"neutralization_run_not_found:{run_id}")

    manifest_path = run_dir / "run.json"
    manifest_candidate = _resolve_manifest_predictions_path(run_dir=run_dir, manifest_path=manifest_path)
    if manifest_candidate is not None:
        return manifest_candidate

    predictions_dir = run_dir / "artifacts" / "predictions"
    if not predictions_dir.is_dir():
        raise ValueError(f"neutralization_run_predictions_not_found:{run_id}")

    for filename in _PREDICTIONS_CANDIDATES:
        candidate = predictions_dir / filename
        if candidate.is_file():
            return candidate.resolve()

    generic_candidates = sorted(
        path for path in predictions_dir.iterdir() if path.is_file() and path.suffix.lower() in {".parquet", ".csv"}
    )
    if len(generic_candidates) == 1:
        return generic_candidates[0].resolve()

    raise ValueError(f"neutralization_run_predictions_not_found:{run_id}")


def read_table(path: Path) -> pd.DataFrame:
    """Read parquet/csv into a dataframe."""

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"neutralization_predictions_format_unsupported:{path.suffix}")


def write_table(*, frame: pd.DataFrame, path: Path) -> None:
    """Write dataframe to parquet/csv based on file suffix."""

    suffix = path.suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".parquet":
        frame.to_parquet(path, index=False)
        return
    if suffix == ".csv":
        frame.to_csv(path, index=False)
        return
    raise ValueError(f"neutralization_output_format_unsupported:{path.suffix}")


def default_sidecar_output_path(source_path: Path) -> Path:
    """Build default sidecar output path beside source predictions."""

    suffix = source_path.suffix.lower()
    output_suffix = suffix if suffix in {".parquet", ".csv"} else ".parquet"
    return source_path.with_name(f"{source_path.stem}.neutralized{output_suffix}")


def _resolve_manifest_predictions_path(*, run_dir: Path, manifest_path: Path) -> Path | None:
    if not manifest_path.is_file():
        return None

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        return None

    predictions_ref = artifacts.get("predictions")
    if not isinstance(predictions_ref, str) or not predictions_ref.strip():
        return None

    candidate = Path(predictions_ref)
    if not candidate.is_absolute():
        candidate = run_dir / candidate
    resolved = candidate.resolve()
    if resolved.is_file():
        return resolved
    return None


def resolve_neutralizer_path(neutralizer_path: str | Path) -> Path:
    """Resolve and validate one neutralizer table path."""

    path = Path(neutralizer_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"neutralization_neutralizer_file_not_found:{path}")
    return path


def resolve_output_path(*, source_path: Path, output_path: str | Path | None) -> Path:
    """Resolve requested or default neutralization output path."""

    if output_path is None:
        return default_sidecar_output_path(source_path)
    return Path(output_path).expanduser().resolve()


def ensure_unique_join_keys(frame: pd.DataFrame, *, keys: tuple[str, str] = ("era", "id")) -> None:
    """Ensure key columns are unique in the provided dataframe."""

    if frame.duplicated(list(keys)).any():
        raise ValueError("neutralization_neutralizer_keys_duplicated")


def normalize_join_keys(frame: pd.DataFrame, *, era_col: str = "era", id_col: str = "id") -> pd.DataFrame:
    """Return a copy with normalized join key dtypes."""

    payload = frame.copy()
    payload[era_col] = payload[era_col].astype(str).map(_normalize_era_key)
    payload[id_col] = payload[id_col].astype(str)
    return payload


def _normalize_era_key(value: str) -> str:
    normalized = value.strip()
    if normalized.isdigit():
        return str(int(normalized))
    return normalized


def json_safe(value: Any) -> Any:
    """Serialize helper for config snapshots."""

    if isinstance(value, Path):
        return str(value)
    return value
