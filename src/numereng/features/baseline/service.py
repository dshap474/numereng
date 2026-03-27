"""Named baseline build orchestration."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from numereng.features.baseline.contracts import BaselineBuildRequest, BaselineBuildResult
from numereng.features.ensemble.builder import build_blended_predictions, load_ranked_components
from numereng.features.ensemble.weights import EnsembleWeightsError, normalize_weights
from numereng.features.scoring.metrics import attach_scoring_targets, per_era_corr, resolve_fnc_source_paths
from numereng.features.store import resolve_store_root
from numereng.features.training.client import TrainingDataClient, create_training_data_client
from numereng.features.training.errors import TrainingDataError
from numereng.features.training.repo import load_fold_data_lazy, seed_active_benchmark
from numereng.platform.parquet import write_parquet

_SAFE_ID = re.compile(r"^[\w\-.]+$")


class BaselineError(Exception):
    """Base error for baseline workflows."""


class BaselineValidationError(BaselineError):
    """Raised when baseline inputs are invalid."""


class BaselineExecutionError(BaselineError):
    """Raised when baseline construction fails."""


def build_baseline(
    *,
    store_root: str | Path = ".numereng",
    request: BaselineBuildRequest,
    client: TrainingDataClient | None = None,
) -> BaselineBuildResult:
    """Build one named baseline from existing run predictions."""

    run_ids = _dedupe_run_ids(request.run_ids)
    if len(run_ids) < 2:
        raise BaselineValidationError("baseline_run_ids_insufficient")
    if not request.name.strip():
        raise BaselineValidationError("baseline_name_required")
    if not _SAFE_ID.match(request.name):
        raise BaselineValidationError(f"baseline_name_invalid:{request.name}")
    if not request.default_target.strip():
        raise BaselineValidationError("baseline_default_target_invalid")

    resolved_store_root = resolve_store_root(store_root)
    data_root = (resolved_store_root / "datasets").resolve()
    baseline_dir = (data_root / "baselines" / request.name).resolve()

    manifests = tuple(_load_run_context(resolved_store_root, run_id) for run_id in run_ids)
    available_targets = _resolve_available_targets(manifests)
    if request.default_target not in available_targets:
        raise BaselineValidationError(f"baseline_default_target_missing_from_runs:{request.default_target}")

    data_version = _require_consistent(manifests, "data_version")
    dataset_variant = _require_consistent(manifests, "dataset_variant")
    dataset_scope = _require_consistent(manifests, "dataset_scope")
    era_col = _require_consistent(manifests, "era_col")
    id_col = _require_consistent(manifests, "id_col")

    try:
        ranked_predictions, era_series, id_series, _ = load_ranked_components(
            store_root=resolved_store_root,
            run_ids=run_ids,
            target_col=request.default_target,
        )
        weights = normalize_weights(raw_weights=None, n_components=len(run_ids))
    except (ValueError, EnsembleWeightsError) as exc:
        raise BaselineExecutionError(str(exc)) from exc

    blended = build_blended_predictions(ranked_predictions=ranked_predictions, weights=weights)
    prediction_frame = pd.DataFrame(
        {
            era_col: era_series.astype(str).tolist(),
            id_col: id_series.astype(str).tolist(),
            "prediction": blended,
        }
    )

    baseline_dir.mkdir(parents=True, exist_ok=True)
    predictions_filename = f"pred_{request.name}.parquet"
    predictions_path = baseline_dir / predictions_filename
    write_parquet(prediction_frame, predictions_path, index=False)

    joined_targets = _attach_targets(
        prediction_frame=prediction_frame,
        target_cols=available_targets,
        data_version=data_version,
        dataset_variant=dataset_variant,
        dataset_scope=dataset_scope,
        era_col=era_col,
        id_col=id_col,
        data_root=data_root,
        client=client or create_training_data_client(),
    )
    artifacts = _write_target_corr_artifacts(
        baseline_dir=baseline_dir,
        predictions_filename=predictions_filename,
        prediction_frame=joined_targets,
        target_cols=available_targets,
        era_col=era_col,
        default_target=request.default_target,
    )

    created_at = datetime.now(UTC).isoformat()
    source_experiment_id = _single_or_none(item.experiment_id for item in manifests)
    metadata_path = baseline_dir / "baseline.json"
    metadata_payload = _build_metadata(
        name=request.name,
        description=request.description,
        baseline_dir=baseline_dir,
        predictions_filename=predictions_filename,
        artifacts=artifacts,
        default_target=request.default_target,
        available_targets=available_targets,
        run_ids=run_ids,
        manifests=manifests,
        source_experiment_id=source_experiment_id,
        prediction_frame=prediction_frame,
        era_col=era_col,
        created_at=created_at,
    )
    metadata_path.write_text(json.dumps(metadata_payload, indent=2, sort_keys=True), encoding="utf-8")

    active_predictions_path: Path | None = None
    active_metadata_path: Path | None = None
    if request.promote_active:
        active_predictions_path, active_metadata_path = seed_active_benchmark(
            source_dir=baseline_dir,
            predictions_filename=predictions_filename,
            data_root=data_root,
        )

    return BaselineBuildResult(
        name=request.name,
        baseline_dir=baseline_dir,
        predictions_path=predictions_path,
        metadata_path=metadata_path,
        available_targets=available_targets,
        default_target=request.default_target,
        source_run_ids=run_ids,
        source_experiment_id=source_experiment_id,
        active_predictions_path=active_predictions_path,
        active_metadata_path=active_metadata_path,
        created_at=created_at,
    )


def _attach_targets(
    *,
    prediction_frame: pd.DataFrame,
    target_cols: tuple[str, ...],
    data_version: str,
    dataset_variant: str,
    dataset_scope: str,
    era_col: str,
    id_col: str,
    data_root: Path,
    client: TrainingDataClient,
) -> pd.DataFrame:
    try:
        source_paths, include_validation_only = resolve_fnc_source_paths(
            client=client,
            data_version=data_version,
            dataset_variant=dataset_variant,
            feature_source_paths=None,
            dataset_scope=dataset_scope,
            data_root=data_root,
        )
        eras = sorted({str(value) for value in prediction_frame[era_col].tolist()})
        target_frame = load_fold_data_lazy(
            source_paths,
            eras=eras,
            columns=[era_col, id_col, *target_cols],
            era_col=era_col,
            id_col=id_col,
            include_validation_only=include_validation_only,
        )
        return attach_scoring_targets(
            prediction_frame,
            target_frame,
            target_cols=target_cols,
            era_col=era_col,
            id_col=id_col,
        )
    except (TrainingDataError, ValueError) as exc:
        raise BaselineExecutionError(str(exc)) from exc


def _write_target_corr_artifacts(
    *,
    baseline_dir: Path,
    predictions_filename: str,
    prediction_frame: pd.DataFrame,
    target_cols: tuple[str, ...],
    era_col: str,
    default_target: str,
) -> dict[str, str]:
    artifacts: dict[str, str] = {
        "predictions": str((baseline_dir / predictions_filename).resolve()),
    }
    for target_col in target_cols:
        corr_frame = per_era_corr(prediction_frame, ["prediction"], target_col, era_col=era_col).reset_index()
        if "prediction" in corr_frame.columns:
            corr_frame = corr_frame.rename(columns={"prediction": "corr"})
        filename = f"val_per_era_corr20v2_{target_col}.parquet"
        artifact_path = baseline_dir / filename
        write_parquet(corr_frame, artifact_path, index=False)
        artifacts[f"per_era_corr_{target_col}"] = str(artifact_path.resolve())
        if target_col == default_target:
            artifacts["per_era_corr_default"] = str(artifact_path.resolve())
    return artifacts


def _build_metadata(
    *,
    name: str,
    description: str | None,
    baseline_dir: Path,
    predictions_filename: str,
    artifacts: dict[str, str],
    default_target: str,
    available_targets: tuple[str, ...],
    run_ids: tuple[str, ...],
    manifests: tuple[_RunContext, ...],
    source_experiment_id: str | None,
    prediction_frame: pd.DataFrame,
    era_col: str,
    created_at: str,
) -> dict[str, object]:
    seeds = sorted({seed for seed in (item.seed for item in manifests) if seed is not None})
    eras = sorted({str(value) for value in prediction_frame[era_col].tolist()})
    return {
        "artifacts": artifacts,
        "available_targets": list(available_targets),
        "created_at": created_at,
        "default_target": default_target,
        "description": description or _default_description(available_targets=available_targets, run_ids=run_ids),
        "era_count": len(eras),
        "era_max": eras[-1] if eras else None,
        "era_min": eras[0] if eras else None,
        "kind": "internal_research_baseline",
        "metric_basis": "corr20v2",
        "name": name,
        "pred_col": "prediction",
        "predictions_file": predictions_filename,
        "row_count": int(len(prediction_frame)),
        "source_experiment_id": source_experiment_id,
        "source_families": _source_families(available_targets=available_targets, manifests=manifests),
        "source_path": str(baseline_dir.resolve()),
        "source_run_ids": list(run_ids),
        "source_seeds": seeds,
        "split": "oof_full",
    }


def _default_description(*, available_targets: tuple[str, ...], run_ids: tuple[str, ...]) -> str:
    target_list = ", ".join(available_targets)
    return f"Equal-weight rank-average blend of {len(run_ids)} runs across {target_list}."


def _source_families(
    *,
    available_targets: tuple[str, ...],
    manifests: tuple[_RunContext, ...],
) -> dict[str, list[str]]:
    families: dict[str, list[str]] = {_target_family_key(target): [] for target in available_targets}
    for item in manifests:
        families.setdefault(_target_family_key(item.target_col), []).append(item.run_id)
    return families


def _target_family_key(target_col: str) -> str:
    if target_col.startswith("target_"):
        return target_col[len("target_") :].replace("_", "")
    return target_col.replace("_", "")


def _resolve_available_targets(manifests: tuple[_RunContext, ...]) -> tuple[str, ...]:
    values = sorted({item.target_col for item in manifests})
    if not values:
        raise BaselineValidationError("baseline_run_targets_missing")
    return tuple(values)


def _require_consistent(manifests: tuple[_RunContext, ...], field_name: str) -> str:
    values = {getattr(item, field_name) for item in manifests}
    if len(values) != 1:
        raise BaselineValidationError(f"baseline_inconsistent_{field_name}")
    value = values.pop()
    if not value:
        raise BaselineValidationError(f"baseline_missing_{field_name}")
    return str(value)


def _dedupe_run_ids(run_ids: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    deduped: list[str] = []
    for run_id in run_ids:
        value = str(run_id).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return tuple(deduped)


def _single_or_none(values: Any) -> str | None:
    resolved = {value for value in values if value}
    if len(resolved) == 1:
        return str(next(iter(resolved)))
    return None


class _RunContext:
    def __init__(
        self,
        *,
        run_id: str,
        experiment_id: str | None,
        target_col: str,
        data_version: str,
        dataset_variant: str,
        dataset_scope: str,
        era_col: str,
        id_col: str,
        seed: int | None,
    ) -> None:
        self.run_id = run_id
        self.experiment_id = experiment_id
        self.target_col = target_col
        self.data_version = data_version
        self.dataset_variant = dataset_variant
        self.dataset_scope = dataset_scope
        self.era_col = era_col
        self.id_col = id_col
        self.seed = seed


def _load_run_context(store_root: Path, run_id: str) -> _RunContext:
    if not _SAFE_ID.match(run_id):
        raise BaselineValidationError(f"baseline_run_id_invalid:{run_id}")
    run_dir = store_root / "runs" / run_id
    manifest = _load_required_json(run_dir / "run.json", f"baseline_run_manifest_invalid:{run_id}")
    resolved = _load_optional_json(run_dir / "resolved.json")
    data = _as_mapping(manifest.get("data"))
    training = _as_mapping(manifest.get("training"))
    training_data = _as_mapping(training.get("data"))
    resolved_data = _as_mapping(resolved.get("data"))
    resolved_model = _as_mapping(resolved.get("model"))
    resolved_params = _as_mapping(resolved_model.get("params"))

    target_col = _coerce_required_str(
        data.get("target_col") or resolved_data.get("target_col"),
        error_code=f"baseline_run_target_missing:{run_id}",
    )
    data_version = _coerce_required_str(
        data.get("version") or resolved_data.get("data_version"),
        error_code=f"baseline_run_data_version_missing:{run_id}",
    )
    return _RunContext(
        run_id=run_id,
        experiment_id=_coerce_optional_str(manifest.get("experiment_id")),
        target_col=target_col,
        data_version=data_version,
        dataset_variant=_coerce_optional_str(training_data.get("dataset_variant")) or "non_downsampled",
        dataset_scope=_coerce_optional_str(training_data.get("dataset_scope")) or "train_plus_validation",
        era_col=_coerce_optional_str(resolved_data.get("era_col")) or "era",
        id_col=_coerce_optional_str(resolved_data.get("id_col")) or "id",
        seed=_coerce_optional_int(resolved_params.get("random_state")),
    )


def _load_required_json(path: Path, error_code: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineExecutionError(error_code) from exc
    if not isinstance(payload, dict):
        raise BaselineExecutionError(error_code)
    return payload


def _load_optional_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _as_mapping(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _coerce_required_str(value: object, *, error_code: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise BaselineExecutionError(error_code)


def _coerce_optional_str(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _coerce_optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


__all__ = [
    "BaselineError",
    "BaselineExecutionError",
    "BaselineValidationError",
    "build_baseline",
]
