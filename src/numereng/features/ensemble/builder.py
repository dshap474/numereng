"""Run-prediction loading and blend construction for ensembles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class EnsembleBuildError(ValueError):
    """Raised when ensemble inputs cannot be loaded/aligned."""


def load_ranked_components(
    *,
    store_root: Path,
    run_ids: tuple[str, ...],
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series | None]:
    """Load/align run predictions and return per-era ranked component matrix."""

    frames: list[pd.DataFrame] = []
    resolved_target: pd.Series | None = None
    target_included = False
    join_cols = ["era", "id"]

    for run_id in run_ids:
        frame = _load_run_prediction_frame(store_root=store_root, run_id=run_id, target_col=target_col)
        pred_col = f"pred_{run_id}"
        frame = frame.rename(columns={"prediction": pred_col})
        keep_cols = join_cols + [pred_col]
        if target_col in frame.columns and not target_included:
            keep_cols.append(target_col)
            target_included = True
        frame = frame[keep_cols]
        frames.append(frame)

    if not frames:
        raise EnsembleBuildError("ensemble_components_empty")

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=join_cols, how="inner")

    if merged.empty:
        raise EnsembleBuildError("ensemble_component_alignment_empty")

    merged = merged.sort_values(join_cols).reset_index(drop=True)

    if target_col in merged.columns:
        resolved_target = merged[target_col]

    pred_cols = [f"pred_{run_id}" for run_id in run_ids]
    ranked = merged.groupby("era", group_keys=False)[pred_cols].rank(pct=True)

    return ranked, merged["era"], merged["id"], resolved_target


def build_blended_predictions(*, ranked_predictions: pd.DataFrame, weights: tuple[float, ...]) -> np.ndarray:
    """Build weighted blend from ranked component matrix."""

    if ranked_predictions.shape[1] != len(weights):
        raise EnsembleBuildError("ensemble_weights_length_mismatch")
    matrix = ranked_predictions.to_numpy(dtype=float)
    blended = np.asarray(matrix @ np.asarray(weights, dtype=float), dtype=float)
    return blended


def _load_run_prediction_frame(*, store_root: Path, run_id: str, target_col: str) -> pd.DataFrame:
    run_dir = store_root / "runs" / run_id
    manifest_path = run_dir / "run.json"
    if not manifest_path.exists():
        raise EnsembleBuildError(f"ensemble_run_manifest_not_found:{run_id}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EnsembleBuildError(f"ensemble_run_manifest_invalid:{run_id}") from exc

    predictions_path = _resolve_predictions_path(run_dir=run_dir, manifest=manifest)
    if predictions_path is None:
        raise EnsembleBuildError(f"ensemble_run_predictions_not_found:{run_id}")

    frame = _read_table(predictions_path)
    if "era" not in frame.columns or "id" not in frame.columns:
        raise EnsembleBuildError(f"ensemble_predictions_missing_keys:{run_id}")

    prediction_col = _resolve_prediction_col(frame)
    if prediction_col != "prediction":
        frame = frame.rename(columns={prediction_col: "prediction"})

    required = ["era", "id", "prediction"]
    if target_col in frame.columns:
        required.append(target_col)

    frame = frame[required].copy()
    frame["era"] = frame["era"].astype(str)
    frame["id"] = frame["id"].astype(str)
    frame["prediction"] = pd.to_numeric(frame["prediction"], errors="coerce")
    frame = frame.dropna(subset=["prediction"]).reset_index(drop=True)
    return frame


def _resolve_predictions_path(*, run_dir: Path, manifest: dict[str, Any]) -> Path | None:
    artifacts = manifest.get("artifacts")
    if isinstance(artifacts, dict):
        candidate = artifacts.get("predictions")
        if isinstance(candidate, str) and candidate.strip():
            path = (run_dir / candidate).resolve()
            if path.exists() and path.is_file():
                return path

    predictions_dir = run_dir / "artifacts" / "predictions"
    if predictions_dir.is_dir():
        allowed_suffixes = {".parquet", ".csv"}
        files = sorted(
            [
                item
                for item in predictions_dir.iterdir()
                if item.is_file() and item.suffix.lower() in allowed_suffixes
            ],
            key=lambda item: item.name,
        )
        if files:
            return files[0]

    for name in ("predictions.parquet", "predictions.csv"):
        path = run_dir / name
        if path.exists() and path.is_file():
            return path

    return None


def _resolve_prediction_col(frame: pd.DataFrame) -> str:
    if "prediction" in frame.columns:
        return "prediction"
    candidates = [
        column
        for column in frame.columns
        if column not in {"era", "id", "target", "target_ender_20", "target_cyrus_20"}
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not candidates:
        raise EnsembleBuildError("ensemble_prediction_column_missing")
    return candidates[0]


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise EnsembleBuildError(f"ensemble_predictions_format_unsupported:{path.suffix}")
