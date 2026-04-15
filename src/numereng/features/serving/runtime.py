"""Runtime helpers for live serving, blending, and pickle assembly."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypedDict

import cloudpickle
import numpy as np
import pandas as pd

from numereng.features.feature_neutralization import neutralize_prediction_frame
from numereng.features.scoring.metrics import attach_benchmark_predictions, load_custom_benchmark_predictions
from numereng.features.serving.contracts import (
    RankMethod,
    ServingBlendRule,
    ServingComponentSpec,
    ServingNeutralizationSpec,
)
from numereng.features.serving.repo import run_dir
from numereng.features.store import resolve_workspace_layout
from numereng.features.training.errors import TrainingModelError
from numereng.features.training.model_artifacts import ModelArtifactError, load_model_artifact
from numereng.features.training.model_factory import build_model
from numereng.features.training.models import build_x_cols, normalize_x_groups
from numereng.features.training.repo import (
    DEFAULT_BASELINES_DIR,
    apply_missing_all_twos_as_nan,
    load_config,
    load_features,
    load_full_data,
)
from numereng.features.training.service import resolve_model_config
from numereng.platform.parquet import write_parquet


class ServingRuntimeError(Exception):
    """Raised when live serving inference fails."""


class ServingUnsupportedConfigError(Exception):
    """Raised when a component config cannot be served in v1."""


class ServingDataClient(Protocol):
    """Client surface required for live-serving data access."""

    def download_dataset(
        self,
        filename: str,
        *,
        dest_path: str | None = None,
        round_num: int | None = None,
    ) -> str:
        """Download one Numerai dataset."""


@dataclass(frozen=True)
class ServingPredictionMember:
    """Minimal metadata needed after a component prediction is materialized."""

    component_id: str
    weight: float


@dataclass(frozen=True)
class ServingDataContextKey:
    """Key for reusing one prepared historical dataset across compatible components."""

    data_version: str
    dataset_variant: str
    feature_set: str
    target_col: str
    era_col: str
    id_col: str
    dataset_scope: str
    nan_missing_all_twos: bool
    missing_value: float


@dataclass(frozen=True)
class PreparedServingData:
    """Prepared historical dataset and feature metadata for one compatible component group."""

    key: ServingDataContextKey
    features: tuple[str, ...]
    full: pd.DataFrame


@dataclass(frozen=True)
class ServingComponentPlan:
    """Resolved serving plan extracted from one component config."""

    component: ServingComponentSpec
    config_path: Path
    data_key: ServingDataContextKey
    target_col: str
    model_type: str
    model_params: dict[str, object]
    model_config: dict[str, object]
    x_groups: tuple[str, ...]
    baseline_name: str | None
    baseline_predictions_path: str | None
    baseline_pred_col: str
    uses_custom_module: bool


@dataclass(frozen=True)
class FittedComponent:
    """Fitted component plus runtime metadata needed for live inference."""

    component_id: str
    weight: float
    model: Any
    id_col: str
    era_col: str
    feature_cols: tuple[str, ...]
    baseline_col: str | None
    baseline_name: str | None
    baseline_predictions_path: str | None
    baseline_pred_col: str = "prediction"


@dataclass(frozen=True)
class LoadedServingArtifact:
    """One persisted run-backed component ready for live inference."""

    component: FittedComponent
    model_type: str
    model_upload_compatible: bool
    uses_custom_module: bool


class HostedComponentPayload(TypedDict):
    """Hosted-safe serialized payload for one LightGBM component."""

    component_id: str
    weight: float
    model_str: str
    id_col: str
    era_col: str
    feature_cols: tuple[str, ...]


class HostedBlendPayload(TypedDict):
    """Hosted-safe serialized blend contract."""

    per_era_rank: bool
    rank_method: RankMethod
    rank_pct: bool
    final_rerank: bool


def prepare_component_plan(
    *,
    workspace_root: str | Path,
    component: ServingComponentSpec,
    config_path: Path,
) -> ServingComponentPlan:
    """Resolve one component config into a serving plan."""

    _ = workspace_root
    config = load_config(config_path)
    data_config = _mapping_dict(config.get("data"))
    model_config = _mapping_dict(config.get("model"))
    preprocessing_config = _mapping_dict(config.get("preprocessing"))
    output_config = _mapping_dict(config.get("output"))

    data_key = ServingDataContextKey(
        data_version=str(data_config.get("data_version", "v5.2")),
        dataset_variant=str(data_config.get("dataset_variant", "non_downsampled")),
        feature_set=str(data_config.get("feature_set", "small")),
        target_col=str(data_config.get("target_col", "target")),
        era_col=str(data_config.get("era_col", "era")),
        id_col=str(data_config.get("id_col", "id")),
        dataset_scope=str(data_config.get("dataset_scope", "train_only")),
        nan_missing_all_twos=bool(preprocessing_config.get("nan_missing_all_twos", False)),
        missing_value=_coerce_float(preprocessing_config.get("missing_value"), default=2.0),
    )
    x_groups_input = model_config.get("x_groups") or model_config.get("data_needed")
    if x_groups_input is not None and not isinstance(x_groups_input, list):
        raise ServingUnsupportedConfigError("serving_component_x_groups_invalid")
    x_groups = tuple(normalize_x_groups(None if x_groups_input is None else [str(item) for item in x_groups_input]))

    baseline_name: str | None = None
    baseline_predictions_path: str | None = None
    baseline_pred_col = "prediction"
    if "baseline" in x_groups:
        baseline_spec = _mapping_dict(model_config.get("baseline"))
        if not baseline_spec:
            raise ServingUnsupportedConfigError("serving_component_baseline_config_missing")
        raw_name = baseline_spec.get("name")
        raw_path = baseline_spec.get("predictions_path")
        if raw_name is None or raw_path is None:
            raise ServingUnsupportedConfigError("serving_component_baseline_config_missing")
        baseline_name = str(raw_name)
        baseline_predictions_path = _resolve_baseline_path(
            workspace_root=workspace_root,
            baselines_dir_override=output_config.get("baselines_dir") or data_config.get("baselines_dir"),
            baseline_path=str(raw_path),
        )
        baseline_pred_col = str(baseline_spec.get("pred_col", "prediction"))

    model_type, model_params = resolve_model_config(model_config)
    module_path = model_config.get("module_path")
    uses_custom_module = bool(module_path) or model_type != "LGBMRegressor"
    return ServingComponentPlan(
        component=component,
        config_path=config_path,
        data_key=data_key,
        target_col=data_key.target_col,
        model_type=model_type,
        model_params=dict(model_params),
        model_config=model_config,
        x_groups=x_groups,
        baseline_name=baseline_name,
        baseline_predictions_path=baseline_predictions_path,
        baseline_pred_col=baseline_pred_col,
        uses_custom_module=uses_custom_module,
    )


def prepare_training_context(
    *,
    workspace_root: str | Path,
    client: ServingDataClient,
    plan: ServingComponentPlan,
    cache: dict[ServingDataContextKey, PreparedServingData] | None = None,
) -> PreparedServingData:
    """Load and cache one reusable historical dataset context."""

    key = plan.data_key
    if cache is not None and key in cache:
        return cache[key]

    data_root = resolve_workspace_layout(workspace_root).store_root / "datasets"
    features = tuple(
        load_features(
            client,
            key.data_version,
            key.feature_set,
            dataset_variant=key.dataset_variant,
            data_root=data_root,
        )
    )
    full = load_full_data(
        client,
        key.data_version,
        key.dataset_variant,
        list(features),
        key.era_col,
        key.target_col,
        key.id_col,
        dataset_scope=key.dataset_scope,
        data_root=data_root,
    )
    if key.nan_missing_all_twos:
        full = apply_missing_all_twos_as_nan(full, list(features), key.era_col, key.missing_value)
    prepared = PreparedServingData(key=key, features=features, full=full)
    if cache is not None:
        cache[key] = prepared
    return prepared


def fit_component(
    *,
    workspace_root: str | Path,
    client: ServingDataClient,
    component: ServingComponentSpec,
    config_path: Path,
    plan: ServingComponentPlan | None = None,
    prepared_data: PreparedServingData | None = None,
) -> FittedComponent:
    """Fit one component on historical full data for live inference."""

    resolved_plan = (
        prepare_component_plan(workspace_root=workspace_root, component=component, config_path=config_path)
        if plan is None
        else plan
    )
    context = (
        prepare_training_context(workspace_root=workspace_root, client=client, plan=resolved_plan)
        if prepared_data is None
        else prepared_data
    )
    key = resolved_plan.data_key
    full = context.full

    baseline_col: str | None = None
    baseline_name = resolved_plan.baseline_name
    baseline_predictions_path = resolved_plan.baseline_predictions_path
    baseline_pred_col = resolved_plan.baseline_pred_col
    if baseline_name and baseline_predictions_path:
        baseline_frame, baseline_col = load_custom_benchmark_predictions(
            baseline_predictions_path,
            baseline_name,
            pred_col=baseline_pred_col,
            era_col=key.era_col,
            id_col=key.id_col,
        )
        full = attach_benchmark_predictions(
            full.copy(),
            baseline_frame,
            baseline_col,
            era_col=key.era_col,
            id_col=key.id_col,
        )

    feature_cols = tuple(
        build_x_cols(
            x_groups=list(resolved_plan.x_groups),
            features=list(context.features),
            era_col=key.era_col,
            id_col=key.id_col,
            baseline_col=baseline_col,
        )
    )
    model_params = _apply_serving_resource_controls(
        model_type=resolved_plan.model_type,
        model_params=resolved_plan.model_params,
    )
    try:
        model = build_model(
            resolved_plan.model_type,
            model_params,
            resolved_plan.model_config,
            feature_cols=list(feature_cols),
            store_root=resolve_workspace_layout(workspace_root).store_root,
        )
    except (ImportError, ModuleNotFoundError, TrainingModelError) as exc:
        raise ServingUnsupportedConfigError("serving_component_dependency_missing") from exc

    labeled = full[full[resolved_plan.target_col].notna()]
    if labeled.empty:
        raise ServingRuntimeError("serving_component_no_labeled_rows")
    try:
        model.fit(labeled[list(feature_cols)], labeled[resolved_plan.target_col])
    except MemoryError as exc:
        raise ServingRuntimeError("serving_resource_exhausted") from exc
    except OSError as exc:
        raise ServingRuntimeError("serving_runtime_os_error") from exc
    return FittedComponent(
        component_id=component.component_id,
        weight=component.weight,
        model=model,
        id_col=key.id_col,
        era_col=key.era_col,
        feature_cols=feature_cols,
        baseline_col=baseline_col,
        baseline_name=baseline_name,
        baseline_predictions_path=baseline_predictions_path,
        baseline_pred_col=baseline_pred_col,
    )


def load_run_backed_component(
    *,
    workspace_root: str | Path,
    component: ServingComponentSpec,
) -> LoadedServingArtifact:
    """Load one persisted fitted component from a run-backed artifact."""

    if component.run_id is None:
        raise ServingRuntimeError("serving_model_artifact_requires_run_id")
    try:
        loaded = load_model_artifact(run_dir=run_dir(workspace_root=workspace_root, run_id=component.run_id))
    except ModelArtifactError as exc:
        raise ServingRuntimeError(str(exc)) from exc
    manifest = loaded.manifest
    fitted = FittedComponent(
        component_id=component.component_id,
        weight=component.weight,
        model=loaded.model,
        id_col=manifest.id_col,
        era_col=manifest.era_col,
        feature_cols=manifest.feature_cols,
        baseline_col=manifest.baseline_col,
        baseline_name=manifest.baseline_name,
        baseline_predictions_path=manifest.baseline_predictions_path,
        baseline_pred_col=manifest.baseline_pred_col,
    )
    return LoadedServingArtifact(
        component=fitted,
        model_type=manifest.model_type,
        model_upload_compatible=manifest.model_upload_compatible,
        uses_custom_module=manifest.uses_custom_module,
    )


def prediction_member_from_fitted(component: FittedComponent) -> ServingPredictionMember:
    """Strip one fitted component down to blend-time metadata only."""

    return ServingPredictionMember(component_id=component.component_id, weight=component.weight)


def predict_component_live(*, component: FittedComponent, live_features: pd.DataFrame) -> pd.DataFrame:
    """Run one fitted component on a live feature frame."""

    live = _ensure_id_and_era_columns(live_features.copy(), id_col=component.id_col, era_col=component.era_col)
    if component.baseline_col and component.baseline_name and component.baseline_predictions_path:
        baseline_frame, baseline_col = load_custom_benchmark_predictions(
            component.baseline_predictions_path,
            component.baseline_name,
            pred_col=component.baseline_pred_col,
            era_col=component.era_col,
            id_col=component.id_col,
        )
        live = attach_benchmark_predictions(
            live,
            baseline_frame,
            baseline_col,
            era_col=component.era_col,
            id_col=component.id_col,
        )
    missing = [col for col in component.feature_cols if col not in live.columns]
    if missing:
        raise ServingRuntimeError("serving_live_feature_columns_missing:" + ",".join(missing[:5]))
    try:
        values = component.model.predict(live[list(component.feature_cols)])
    except MemoryError as exc:
        raise ServingRuntimeError("serving_resource_exhausted") from exc
    return pd.DataFrame(
        {
            "era": live[component.era_col].astype(str).to_numpy(),
            "id": live[component.id_col].astype(str).to_numpy(),
            "prediction": np.asarray(values, dtype=float),
        }
    )


def blend_component_predictions(
    *,
    component_predictions: list[tuple[ServingPredictionMember, pd.DataFrame]],
    live_features: pd.DataFrame,
    blend_rule: ServingBlendRule,
    neutralization: ServingNeutralizationSpec | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Blend fitted component predictions into internal and submit-ready frames."""

    if not component_predictions:
        raise ServingRuntimeError("serving_component_predictions_empty")
    anchor = component_predictions[0][1][["era", "id"]].copy()
    blended = np.zeros(len(anchor), dtype=float)
    for member, frame in component_predictions:
        if not frame[["era", "id"]].equals(anchor[["era", "id"]]):
            raise ServingRuntimeError("serving_component_predictions_misaligned")
        ranked = _rank_prediction_frame(frame=frame, blend_rule=blend_rule)
        blended += ranked["prediction"].to_numpy(dtype=float) * float(member.weight)

    internal = anchor.copy()
    internal["prediction"] = blended
    if blend_rule.final_rerank:
        internal = _rank_prediction_frame(frame=internal, blend_rule=blend_rule)
    if neutralization and neutralization.enabled:
        neutralizer_frame = _build_neutralizer_frame(
            live_features=live_features,
            neutralizer_cols=neutralization.neutralizer_cols,
        )
        internal = neutralize_prediction_frame(
            predictions=internal,
            neutralizers=neutralizer_frame,
            neutralizer_cols=tuple(col for col in neutralizer_frame.columns if col not in {"era", "id"}),
            proportion=neutralization.proportion,
            mode=neutralization.mode,
            rank_output=neutralization.rank_output,
        )
    submission = internal[["id", "prediction"]].copy()
    return internal, submission


def write_component_predictions(
    *,
    component_predictions: list[tuple[ServingPredictionMember, pd.DataFrame]],
    component_dir: Path,
) -> tuple[Path, ...]:
    """Persist per-component live prediction artifacts."""

    paths: list[Path] = []
    for member, frame in component_predictions:
        paths.append(write_parquet(frame, component_dir / f"{member.component_id}.parquet", index=False))
    return tuple(paths)


def write_blended_predictions(*, internal: pd.DataFrame, submission: pd.DataFrame, live_dir: Path) -> tuple[Path, Path]:
    """Persist final blended live predictions."""

    internal_path = write_parquet(internal, live_dir / "blended_internal.parquet", index=False)
    submission_path = write_parquet(submission, live_dir / "submission.parquet", index=False)
    return internal_path, submission_path


def build_pickled_predictor(
    *,
    fitted_components: tuple[FittedComponent, ...],
    blend_rule: ServingBlendRule,
    neutralization: ServingNeutralizationSpec | None,
    pickle_path: Path,
) -> Path:
    """Serialize one Numerai-compatible predict callable."""
    if neutralization is not None and neutralization.enabled:
        raise ServingUnsupportedConfigError("serving_model_upload_neutralization_not_supported")

    hosted_components = tuple(_hosted_component_payload(item) for item in fitted_components)
    blend_payload: HostedBlendPayload = {
        "per_era_rank": blend_rule.per_era_rank,
        "rank_method": blend_rule.rank_method,
        "rank_pct": blend_rule.rank_pct,
        "final_rerank": blend_rule.final_rerank,
    }

    class HostedPredictor:
        def __init__(
            self,
            *,
            components: tuple[HostedComponentPayload, ...],
            blend: HostedBlendPayload,
        ) -> None:
            self._components = components
            self._blend = blend

        def __call__(self, live_features, live_benchmark_models=None):
            _ = live_benchmark_models
            import lightgbm as lgb
            import numpy as np
            import pandas as pd

            def _ensure_id_and_era(frame, *, id_col, era_col):
                if id_col not in frame.columns:
                    if frame.index.name == id_col:
                        frame = frame.reset_index()
                    else:
                        raise ValueError(f"serving_live_missing_id_col:{id_col}")
                if era_col not in frame.columns:
                    frame[era_col] = "live"
                return frame

            def _rank_prediction_frame(frame):
                ranked = frame.copy()
                if self._blend["per_era_rank"] and "era" in ranked.columns:
                    ranked["prediction"] = ranked.groupby("era", sort=False)["prediction"].rank(
                        method=self._blend["rank_method"],
                        pct=self._blend["rank_pct"],
                    )
                else:
                    ranked["prediction"] = ranked["prediction"].rank(
                        method=self._blend["rank_method"],
                        pct=self._blend["rank_pct"],
                    )
                return ranked

            component_predictions = []
            for item in self._components:
                live = _ensure_id_and_era(
                    live_features.copy(),
                    id_col=item["id_col"],
                    era_col=item["era_col"],
                )
                missing = [col for col in item["feature_cols"] if col not in live.columns]
                if missing:
                    raise ValueError("serving_live_feature_columns_missing:" + ",".join(missing[:5]))
                model = lgb.Booster(model_str=item["model_str"])
                values = model.predict(live[list(item["feature_cols"])])
                frame = pd.DataFrame(
                    {
                        "era": live[item["era_col"]].astype(str).to_numpy(),
                        "id": live[item["id_col"]].astype(str).to_numpy(),
                        "prediction": np.asarray(values, dtype=float),
                    }
                )
                component_predictions.append((item["component_id"], float(item["weight"]), frame))

            if not component_predictions:
                raise ValueError("serving_component_predictions_empty")
            anchor = component_predictions[0][2][["era", "id"]].copy()
            blended = np.zeros(len(anchor), dtype=float)
            for _, weight, frame in component_predictions:
                if not frame[["era", "id"]].equals(anchor[["era", "id"]]):
                    raise ValueError("serving_component_predictions_misaligned")
                ranked = _rank_prediction_frame(frame)
                blended += ranked["prediction"].to_numpy(dtype=float) * weight

            internal = anchor.copy()
            internal["prediction"] = blended
            if self._blend["final_rerank"]:
                internal = _rank_prediction_frame(internal)
            return pd.DataFrame({"prediction": internal["prediction"].to_numpy(dtype=float)}, index=live_features.index)

    predictor = HostedPredictor(components=hosted_components, blend=blend_payload)

    resolved = Path(pickle_path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("wb") as handle:
        cloudpickle.dump(predictor, handle)
    return resolved


def _apply_serving_resource_controls(*, model_type: str, model_params: dict[str, object]) -> dict[str, object]:
    normalized = dict(model_params)
    if model_type == "LGBMRegressor":
        normalized["num_threads"] = min(_coerce_int(normalized.get("num_threads"), default=1), 1)
        if "force_col_wise" not in normalized and "force_row_wise" not in normalized:
            normalized["force_col_wise"] = True
    for key in ("n_jobs",):
        if key in normalized:
            normalized[key] = min(_coerce_int(normalized.get(key), default=1), 1)
    return normalized


def _hosted_component_payload(component: FittedComponent) -> HostedComponentPayload:
    if component.baseline_predictions_path is not None:
        raise ServingUnsupportedConfigError("serving_model_upload_baseline_inputs_not_supported")
    model = component.model
    if type(model).__module__ == "numereng.features.models.lgbm":
        raw_model = getattr(model, "_model", None)
        if raw_model is None:
            raise ServingUnsupportedConfigError("serving_model_upload_model_unwrap_failed")
        model = raw_model
    elif type(model).__module__.startswith("lightgbm"):
        model = model
    else:
        raise ServingUnsupportedConfigError("serving_model_upload_model_type_not_supported")
    booster = getattr(model, "booster_", None)
    if booster is None:
        raise ServingUnsupportedConfigError("serving_model_upload_model_unwrap_failed")
    return {
        "component_id": component.component_id,
        "weight": float(component.weight),
        "model_str": booster.model_to_string(),
        "id_col": component.id_col,
        "era_col": component.era_col,
        "feature_cols": tuple(component.feature_cols),
    }


def _ensure_id_and_era_columns(frame: pd.DataFrame, *, id_col: str, era_col: str) -> pd.DataFrame:
    if id_col not in frame.columns:
        if frame.index.name == id_col:
            frame = frame.reset_index()
        else:
            raise ServingRuntimeError(f"serving_live_missing_id_col:{id_col}")
    if era_col not in frame.columns:
        frame[era_col] = "live"
    return frame


def _rank_prediction_frame(*, frame: pd.DataFrame, blend_rule: ServingBlendRule) -> pd.DataFrame:
    ranked = frame.copy()
    if blend_rule.per_era_rank and "era" in ranked.columns:
        ranked["prediction"] = ranked.groupby("era", sort=False)["prediction"].rank(
            method=blend_rule.rank_method,
            pct=blend_rule.rank_pct,
        )
    else:
        ranked["prediction"] = ranked["prediction"].rank(method=blend_rule.rank_method, pct=blend_rule.rank_pct)
    return ranked


def _mapping_dict(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return {}


def _coerce_int(value: object, *, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        return int(stripped)
    return default


def _coerce_float(value: object, *, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        return float(stripped)
    return default


def _build_neutralizer_frame(
    *,
    live_features: pd.DataFrame,
    neutralizer_cols: tuple[str, ...] | None,
) -> pd.DataFrame:
    frame = _ensure_id_and_era_columns(live_features.copy(), id_col="id", era_col="era")
    if neutralizer_cols is None:
        selected = tuple(
            str(col)
            for col in frame.columns
            if col not in {"era", "id", "data_type"} and str(col).startswith("feature_")
        )
        if not selected:
            selected = tuple(str(col) for col in frame.columns if col not in {"era", "id", "data_type"})
    else:
        selected = tuple(neutralizer_cols)
    missing = [col for col in selected if col not in frame.columns]
    if missing:
        raise ServingRuntimeError("serving_neutralizer_columns_missing:" + ",".join(missing[:5]))
    payload = frame[["era", "id", *selected]].copy()
    for col in selected:
        payload[col] = pd.to_numeric(payload[col], errors="coerce").fillna(0.0)
    return payload


def _resolve_baseline_path(
    *,
    workspace_root: str | Path,
    baselines_dir_override: object,
    baseline_path: str,
) -> str:
    candidate = Path(baseline_path).expanduser()
    if candidate.is_absolute():
        return str(candidate.resolve())
    layout = resolve_workspace_layout(workspace_root)
    if isinstance(baselines_dir_override, str) and baselines_dir_override.strip():
        base_root = Path(baselines_dir_override).expanduser()
        if not base_root.is_absolute():
            base_root = layout.workspace_root / base_root
    else:
        base_root = layout.workspace_root / DEFAULT_BASELINES_DIR
    return str((base_root / candidate).resolve())


__all__ = [
    "FittedComponent",
    "LoadedServingArtifact",
    "PreparedServingData",
    "ServingComponentPlan",
    "ServingDataClient",
    "ServingDataContextKey",
    "ServingPredictionMember",
    "ServingRuntimeError",
    "ServingUnsupportedConfigError",
    "blend_component_predictions",
    "build_pickled_predictor",
    "fit_component",
    "load_run_backed_component",
    "prediction_member_from_fitted",
    "predict_component_live",
    "prepare_component_plan",
    "prepare_training_context",
    "write_blended_predictions",
    "write_component_predictions",
]
