"""`comparison_surface_id` computation (META-PROGRAM §4a, full contract).

The surface id is the sha256 of a canonical JSON object over six fields that
must all match before two runs may be compared or combined. The payload target
is the *contribution* target from score provenance, never run.json's training
target. The panel hash reads the prediction parquet index and is cached in a
sidecar `surface.json` next to metrics.json, recomputed when the parquet mtime
changes.

USAGE:
    from numereng.features.research_portfolio.surface import compute_surface_id
    result = compute_surface_id(run_dir=store_root / "runs" / run_id)
    result.surface_id  # None when predictions/benchmark provenance are absent
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from numereng.features.scoring import SCORING_CONTRACT_VERSION

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

_SIDECAR_NAME = "surface.json"
_DEFAULT_ERA_COL = "era"
_DEFAULT_ID_COL = "id"
_DATA_SCOPE_FIELDS = (
    "data_version",
    "dataset_scope",
    "dataset_variant",
    "feature_set",
    "target_horizon",
)


@dataclass(frozen=True)
class SurfaceResult:
    """Resolved comparison-surface identity for one run."""

    surface_id: str | None
    unavailable_reason: str | None
    components: dict[str, object] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Public computation
# --------------------------------------------------------------------------- #


def compute_surface_id(*, run_dir: Path) -> SurfaceResult:
    """Compute the full comparison_surface_id for one run directory."""

    resolved = _load_json(run_dir / "resolved.json")
    provenance = _load_json(run_dir / "score_provenance.json")
    if resolved is None:
        return SurfaceResult(None, "missing_resolved_config")
    if provenance is None:
        return SurfaceResult(None, "missing_score_provenance")

    era_col, id_col = _panel_columns(provenance)
    benchmark_hash = _benchmark_hash(provenance)
    contribution_target = _contribution_target(provenance)
    panel_hash, panel_reason = _panel_hash(run_dir=run_dir, era_col=era_col, id_col=id_col)

    if panel_hash is None:
        return SurfaceResult(None, panel_reason or "panel_unavailable")
    if benchmark_hash is None:
        return SurfaceResult(None, "missing_benchmark_hash")
    if not contribution_target:
        return SurfaceResult(None, "missing_contribution_target")

    components: dict[str, object] = {
        "data_version_and_scope": _data_version_and_scope(resolved),
        "evaluator_profile": _evaluator_profile(resolved),
        "panel_hash": panel_hash,
        "contribution_target": contribution_target,
        "benchmark_hash": benchmark_hash,
        "scoring_contract_version": SCORING_CONTRACT_VERSION,
    }
    surface_id = _sha256_canonical(components)
    return SurfaceResult(surface_id, None, components)


# --------------------------------------------------------------------------- #
# Component readers
# --------------------------------------------------------------------------- #


def _data_version_and_scope(resolved: dict[str, object]) -> dict[str, object]:
    data = resolved.get("data")
    data = data if isinstance(data, dict) else {}
    return {field_name: data.get(field_name) for field_name in _DATA_SCOPE_FIELDS}


def _evaluator_profile(resolved: dict[str, object]) -> object:
    training = resolved.get("training")
    training = training if isinstance(training, dict) else {}
    return training.get("engine")


def _contribution_target(provenance: dict[str, object]) -> list[str]:
    columns = provenance.get("columns")
    columns = columns if isinstance(columns, dict) else {}
    targets = columns.get("contribution_target_cols")
    if not isinstance(targets, list):
        return []
    return sorted(str(item) for item in targets)


def _benchmark_hash(provenance: dict[str, object]) -> str | None:
    sources = provenance.get("sources")
    sources = sources if isinstance(sources, dict) else {}
    benchmark = sources.get("benchmark")
    benchmark = benchmark if isinstance(benchmark, dict) else {}
    value = benchmark.get("sha256")
    return str(value) if isinstance(value, str) and value else None


def _panel_columns(provenance: dict[str, object]) -> tuple[str, str]:
    columns = provenance.get("columns")
    columns = columns if isinstance(columns, dict) else {}
    era_col = columns.get("era_col")
    id_col = columns.get("id_col")
    era_col = era_col if isinstance(era_col, str) and era_col else _DEFAULT_ERA_COL
    id_col = id_col if isinstance(id_col, str) and id_col else _DEFAULT_ID_COL
    return era_col, id_col


# --------------------------------------------------------------------------- #
# Panel hash (cached sidecar)
# --------------------------------------------------------------------------- #


def _panel_hash(*, run_dir: Path, era_col: str, id_col: str) -> tuple[str | None, str | None]:
    predictions_path = _predictions_path(run_dir)
    if predictions_path is None:
        return None, "missing_predictions"
    mtime_ns = predictions_path.stat().st_mtime_ns
    cached = _read_sidecar(run_dir, predictions_mtime=mtime_ns)
    if cached is not None:
        return cached, None
    try:
        import pandas as pd

        frame = pd.read_parquet(predictions_path, columns=[era_col, id_col])
    except Exception:  # noqa: BLE001 - any read failure means the panel is unusable
        return None, "predictions_unreadable"
    ordered = list(zip((str(era) for era in frame[era_col]), (str(pid) for pid in frame[id_col]), strict=True))
    panel_hash = hashlib.sha256(json.dumps(ordered, separators=(",", ":")).encode("utf-8")).hexdigest()
    _write_sidecar(run_dir, predictions_mtime=mtime_ns, panel_hash=panel_hash)
    return panel_hash, None


def _predictions_path(run_dir: Path) -> Path | None:
    manifest = _load_json(run_dir / "run.json")
    if isinstance(manifest, dict):
        artifacts = manifest.get("artifacts")
        rel = artifacts.get("predictions") if isinstance(artifacts, dict) else None
        if isinstance(rel, str) and rel:
            candidate = run_dir / rel
            if candidate.is_file():
                return candidate
    matches = sorted((run_dir / "artifacts" / "predictions").glob("pred_*.parquet"))
    return matches[0] if matches else None


def _read_sidecar(run_dir: Path, *, predictions_mtime: int) -> str | None:
    payload = _load_json(run_dir / _SIDECAR_NAME)
    if not isinstance(payload, dict):
        return None
    if payload.get("predictions_mtime") != predictions_mtime:
        return None
    panel_hash = payload.get("panel_hash")
    return panel_hash if isinstance(panel_hash, str) and panel_hash else None


def _write_sidecar(run_dir: Path, *, predictions_mtime: int, panel_hash: str) -> None:
    payload = {"predictions_mtime": predictions_mtime, "panel_hash": panel_hash}
    try:
        (run_dir / _SIDECAR_NAME).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    except OSError:
        return


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _sha256_canonical(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, object] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


__all__ = ["SurfaceResult", "compute_surface_id"]
