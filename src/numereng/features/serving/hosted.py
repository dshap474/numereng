"""Numerai-hosted (model upload) payload assembly, dependency proof, and pickle build.

USAGE:
    from numereng.features.serving.hosted import build_pickled_predictor, resolve_live_benchmark_column

    column = resolve_live_benchmark_column(
        baseline_predictions_path=component.baseline_predictions_path,
        data_version="v5.3",
    )
    build = build_pickled_predictor(
        fitted_components=components,
        blend_rule=package.blend_rule,
        neutralization=package.neutralization,
        pickle_path=package_path / "artifacts" / "pickle" / "model.pkl",
    )

Numerai executes the uploaded pickle in an environment that has torch, lightgbm,
pandas, numpy, and scikit-learn but NOT `numereng`. Custom-module components are
therefore carried by value with `cloudpickle.register_pickle_by_value` (plus a reducer
for `@lru_cache` module functions, which that registration does not reach), and every
build proves it by reloading the payload with a recording unpickler: any surviving
`numereng.*` reference, or any distribution missing from the hosted runtime, fails
the build with a specific error code instead of failing later inside Numerai.
"""

from __future__ import annotations

import functools
import io
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, TypedDict

import cloudpickle

from numereng.features.serving.contracts import RankMethod, ServingBlendRule, ServingNeutralizationSpec
from numereng.features.serving.live_benchmark import attach_live_benchmark
from numereng.features.serving.runtime import FittedComponent, ServingUnsupportedConfigError
from numereng.features.training.errors import TrainingModelError
from numereng.features.training.repo import default_benchmark_model
from numereng.features.training.target_transforms import TargetTransformWrapper, prediction_inversion_kind

# --------------------------------------------------------------------------- #
# Hosted runtime contract
# --------------------------------------------------------------------------- #

# Distributions available inside numerai-predict, pinned to that environment.
# See docs/numerai/numerai-tournament/submissions/model-uploads.md.
HOSTED_RUNTIME_DISTRIBUTIONS: dict[str, str] = {
    "cloudpickle": "cloudpickle==3.1.1",
    "joblib": "joblib",
    "lightgbm": "lightgbm==4.5.0",
    "lightning": "pytorch-lightning==2.5.2",
    "numpy": "numpy==2.1.3",
    "pandas": "pandas==2.3.1",
    "pytorch_lightning": "pytorch-lightning==2.5.2",
    "scipy": "scipy",
    "sklearn": "scikit-learn==1.6.1",
    "threadpoolctl": "threadpoolctl",
    "torch": "torch==2.7.1",
}

_ALWAYS_REQUIRED_ROOTS = ("cloudpickle", "numpy", "pandas")
_OFFICIAL_BENCHMARK_KIND = "official_numerai_benchmark"
_BENCHMARK_METADATA_FILENAME = "benchmark.json"
_LGBM_WRAPPER_MODULE = "numereng.features.models.lgbm"
_WRAPPER_CHAIN_DEPTH = 4
_TORCH_SCAN_DEPTH = 6

_DRIFT_HOSTED_RANK = "hosted_blend_rank_reimplemented"
_DRIFT_HOSTED_DEPS = "hosted_dependency_set_from_observed_unpickle"

# `cloudpickle.register_pickle_by_value` only intercepts plain functions and classes, so an
# `@lru_cache` module function still pickles as a global reference into its own module.
_LRU_CACHE_WRAPPER: type = type(functools.lru_cache(maxsize=1)(len))


class HostedComponentPayload(TypedDict):
    """Hosted-safe serialized payload for one package component."""

    component_id: str
    weight: float
    id_col: str
    era_col: str
    feature_cols: tuple[str, ...]
    serialization: str
    model_str: str | None
    model: Any
    baseline_col: str | None
    benchmark_model_col: str | None
    accepts_era: bool
    prediction_inversion: str


class HostedBlendPayload(TypedDict):
    """Hosted-safe serialized blend contract."""

    per_era_rank: bool
    rank_method: RankMethod
    rank_pct: bool
    final_rerank: bool


@dataclass(frozen=True)
class HostedPickleBuild:
    """Result of one hosted pickle build plus the metadata needed to verify it."""

    pickle_path: Path
    python_requirements: tuple[str, ...]
    drift_risks: tuple[str, ...]
    benchmark_model_cols: tuple[tuple[str, str], ...]
    serialization_kinds: tuple[str, ...]

    @property
    def uses_baseline_inputs(self) -> bool:
        """True when at least one component consumes a live benchmark column."""

        return bool(self.benchmark_model_cols)


# --------------------------------------------------------------------------- #
# Benchmark column resolution
# --------------------------------------------------------------------------- #


def resolve_live_benchmark_column(*, baseline_predictions_path: str | None, data_version: str) -> str:
    """Resolve the Numerai `live_benchmark_models` column backing one baseline input.

    Identity comes from the benchmark provenance recorded next to the historical
    baseline parquet (`benchmark.json`), never from a guess: an unresolvable or
    non-official baseline is rejected rather than replaced by the default benchmark.
    """
    if not baseline_predictions_path:
        raise ServingUnsupportedConfigError("serving_model_upload_benchmark_not_resolvable:baseline_path_missing")
    metadata_path = Path(baseline_predictions_path).expanduser().parent / _BENCHMARK_METADATA_FILENAME
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ServingUnsupportedConfigError(
            f"serving_model_upload_benchmark_provenance_missing:{metadata_path}"
        ) from exc
    if not isinstance(metadata, dict):
        raise ServingUnsupportedConfigError(f"serving_model_upload_benchmark_provenance_invalid:{metadata_path}")
    kind = str(metadata.get("kind") or "")
    if kind != _OFFICIAL_BENCHMARK_KIND:
        raise ServingUnsupportedConfigError(f"serving_model_upload_benchmark_not_official:{kind or 'unknown'}")
    name = str(metadata.get("name") or "")
    if not name:
        raise ServingUnsupportedConfigError("serving_model_upload_benchmark_provenance_incomplete:name")
    resolved = name.removeprefix("official_")
    version_prefix = default_benchmark_model(data_version).split("_", 1)[0]
    if not resolved.startswith(f"{version_prefix}_"):
        raise ServingUnsupportedConfigError(
            f"serving_model_upload_benchmark_data_version_mismatch:{resolved}:{version_prefix}"
        )
    return resolved


# --------------------------------------------------------------------------- #
# Component payloads
# --------------------------------------------------------------------------- #


def build_hosted_component_payload(
    *,
    component: FittedComponent,
    data_version: str,
) -> tuple[HostedComponentPayload, tuple[str, ...]]:
    """Build one hosted component payload plus the runtime roots it needs."""

    benchmark_model_col: str | None = None
    if component.baseline_col:
        benchmark_model_col = resolve_live_benchmark_column(
            baseline_predictions_path=component.baseline_predictions_path,
            data_version=data_version,
        )
    inversion = _prediction_inversion(component.model)
    payload: HostedComponentPayload = {
        "component_id": component.component_id,
        "weight": float(component.weight),
        "id_col": component.id_col,
        "era_col": component.era_col,
        "feature_cols": tuple(component.feature_cols),
        "serialization": "cloudpickle_model",
        "model_str": None,
        "model": None,
        "baseline_col": component.baseline_col,
        "benchmark_model_col": benchmark_model_col,
        "accepts_era": bool(getattr(component.model, "accepts_era", False)),
        "prediction_inversion": inversion,
    }
    booster = _maybe_lightgbm_booster(component.model)
    if booster is not None:
        payload["serialization"] = "lightgbm_booster"
        payload["model_str"] = booster.model_to_string()
        return payload, ("lightgbm",)
    if not callable(getattr(component.model, "predict", None)):
        raise ServingUnsupportedConfigError(
            f"serving_model_upload_model_type_not_supported:{type(component.model).__name__}"
        )
    _move_torch_state_to_cpu(component.model)
    payload["model"] = component.model
    return payload, ()


def hosted_component_blockers(*, component: FittedComponent, data_version: str) -> tuple[str, ...]:
    """List the hosted-compat blockers for one fitted component; empty means compatible.

    This is the cheap read-only half of `build_hosted_component_payload` and calls the
    same helpers, so preflight verdicts cannot drift from what the build accepts. It
    cannot prove by-value serializability, which needs a real dump: that stays the
    build's job, and `pickle_upload_ready` still requires the isolated smoke run.
    """

    blockers: list[str] = []
    if component.baseline_col:
        try:
            resolve_live_benchmark_column(
                baseline_predictions_path=component.baseline_predictions_path,
                data_version=data_version,
            )
        except ServingUnsupportedConfigError as exc:
            blockers.append(str(exc))
    try:
        _prediction_inversion(component.model)
    except ServingUnsupportedConfigError as exc:
        blockers.append(str(exc))
    try:
        booster = _maybe_lightgbm_booster(component.model)
    except ServingUnsupportedConfigError as exc:
        blockers.append(str(exc))
    else:
        if booster is None and not callable(getattr(component.model, "predict", None)):
            blockers.append(f"serving_model_upload_model_type_not_supported:{type(component.model).__name__}")
    return tuple(blockers)


def hosted_by_value_modules(model: Any) -> tuple[str, ...]:
    """Name every in-repo module that must be carried by value for one fitted model."""

    names = ["numereng.features.serving.live_benchmark", "numereng.features.training.errors"]
    current = model
    for _ in range(_WRAPPER_CHAIN_DEPTH):
        module_name = type(current).__module__
        if module_name.startswith("numereng"):
            names.append(module_name)
        inner = current.__dict__.get("_model")
        if inner is None or inner is current:
            break
        current = inner
    return tuple(name for name in dict.fromkeys(names) if name in sys.modules)


# --------------------------------------------------------------------------- #
# Pickle build
# --------------------------------------------------------------------------- #


def build_pickled_predictor(
    *,
    fitted_components: tuple[FittedComponent, ...],
    blend_rule: ServingBlendRule,
    neutralization: ServingNeutralizationSpec | None,
    pickle_path: Path,
    data_version: str,
) -> HostedPickleBuild:
    """Serialize one Numerai-compatible `predict(live_features, live_benchmark_models)`."""

    if neutralization is not None and neutralization.enabled:
        raise ServingUnsupportedConfigError("serving_model_upload_neutralization_not_supported")

    hosted_components: list[HostedComponentPayload] = []
    required_roots: set[str] = set(_ALWAYS_REQUIRED_ROOTS)
    # The hosted closure always calls `attach_live_benchmark`, so its module must
    # travel by value even when every component is a plain LightGBM booster.
    by_value: list[str] = ["numereng.features.serving.live_benchmark"]
    for item in fitted_components:
        payload, roots = build_hosted_component_payload(component=item, data_version=data_version)
        hosted_components.append(payload)
        required_roots.update(roots)
        if payload["serialization"] == "cloudpickle_model":
            by_value.extend(hosted_by_value_modules(item.model))
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
                if item["prediction_inversion"] != "identity":
                    raise ValueError(
                        "serving_hosted_prediction_inversion_not_supported:" + str(item["prediction_inversion"])
                    )
                live = _ensure_id_and_era(
                    live_features.copy(),
                    id_col=item["id_col"],
                    era_col=item["era_col"],
                )
                baseline_col = item["baseline_col"]
                benchmark_model_col = item["benchmark_model_col"]
                if baseline_col:
                    if not benchmark_model_col:
                        raise ValueError("serving_live_benchmark_column_unresolved:" + str(item["component_id"]))
                    live = attach_live_benchmark(
                        live,
                        benchmark=live_benchmark_models,
                        id_col=item["id_col"],
                        baseline_col=baseline_col,
                        benchmark_model_col=benchmark_model_col,
                    )
                missing = [col for col in item["feature_cols"] if col not in live.columns]
                if missing:
                    raise ValueError("serving_live_feature_columns_missing:" + ",".join(missing[:5]))
                inputs = live[list(item["feature_cols"])]
                if item["serialization"] == "lightgbm_booster":
                    import lightgbm as lgb

                    values = lgb.Booster(model_str=item["model_str"]).predict(inputs)
                elif item["accepts_era"]:
                    values = item["model"].predict(inputs, era=live[item["era_col"]])
                else:
                    values = item["model"].predict(inputs)
                frame = pd.DataFrame(
                    {
                        "era": live[item["era_col"]].astype(str).to_numpy(),
                        "id": live[item["id_col"]].astype(str).to_numpy(),
                        "prediction": np.asarray(values, dtype=float).ravel(),
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

    predictor = HostedPredictor(components=tuple(hosted_components), blend=blend_payload)
    blob = _dump_by_value(predictor, module_names=tuple(dict.fromkeys(by_value)))
    required_roots.update(verify_hosted_payload(blob))

    resolved = Path(pickle_path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_bytes(blob)

    serialization_kinds = tuple(sorted({item["serialization"] for item in hosted_components}))
    drift_risks = [_DRIFT_HOSTED_RANK]
    if "cloudpickle_model" in serialization_kinds:
        drift_risks.append(_DRIFT_HOSTED_DEPS)
    return HostedPickleBuild(
        pickle_path=resolved,
        python_requirements=tuple(sorted({HOSTED_RUNTIME_DISTRIBUTIONS[root] for root in required_roots})),
        drift_risks=tuple(drift_risks),
        benchmark_model_cols=tuple(
            (item["component_id"], str(item["benchmark_model_col"]))
            for item in hosted_components
            if item["benchmark_model_col"]
        ),
        serialization_kinds=serialization_kinds,
    )


def verify_hosted_payload(blob: bytes) -> tuple[str, ...]:
    """Reload one payload with a recording unpickler and return the runtime roots it needs."""

    observed: set[str] = set()

    class _RecordingUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> Any:
            observed.add(module)
            return super().find_class(module, name)

    try:
        _RecordingUnpickler(io.BytesIO(blob)).load()
    except Exception as exc:
        raise ServingUnsupportedConfigError(
            f"serving_model_upload_payload_load_failed:{type(exc).__name__}:{exc}"
        ) from exc

    in_repo = sorted(item for item in observed if item.split(".", 1)[0].startswith("numereng"))
    if in_repo:
        raise ServingUnsupportedConfigError("serving_model_upload_payload_requires_numereng:" + ",".join(in_repo[:5]))
    roots = sorted({item.split(".", 1)[0] for item in observed})
    unavailable = [
        root for root in roots if root not in sys.stdlib_module_names and root not in HOSTED_RUNTIME_DISTRIBUTIONS
    ]
    if unavailable:
        raise ServingUnsupportedConfigError(
            "serving_model_upload_payload_dependency_unavailable:" + ",".join(unavailable[:5])
        )
    return tuple(root for root in roots if root in HOSTED_RUNTIME_DISTRIBUTIONS)


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _dump_by_value(predictor: Any, *, module_names: tuple[str, ...]) -> bytes:
    modules: list[ModuleType] = []
    registered = cloudpickle.list_registry_pickle_by_value()
    for name in module_names:
        module = sys.modules.get(name)
        if module is None or name in registered:
            continue
        modules.append(module)

    by_value_names = frozenset(module_names)

    def rebuild_lru_cache(func: Any, maxsize: Any, typed: bool) -> Any:
        import functools as _functools

        return _functools.lru_cache(maxsize=maxsize, typed=typed)(func)

    def reduce_lru_cache(wrapper: Any) -> Any:
        if getattr(wrapper, "__module__", None) not in by_value_names:
            return wrapper.__reduce__()
        parameters = wrapper.cache_parameters()
        return rebuild_lru_cache, (wrapper.__wrapped__, parameters["maxsize"], parameters["typed"])

    overrides: dict[Any, Any] = dict(cloudpickle.Pickler.dispatch_table)
    overrides[_LRU_CACHE_WRAPPER] = reduce_lru_cache

    class _ByValuePickler(cloudpickle.Pickler):
        dispatch_table = overrides

    for module in modules:
        cloudpickle.register_pickle_by_value(module)
    try:
        buffer = io.BytesIO()
        _ByValuePickler(buffer).dump(predictor)
        return buffer.getvalue()
    except Exception as exc:
        raise ServingUnsupportedConfigError(
            f"serving_model_upload_model_serialization_failed:{type(exc).__name__}:{exc}"
        ) from exc
    finally:
        for module in modules:
            cloudpickle.unregister_pickle_by_value(module)


def _move_torch_state_to_cpu(model: Any) -> None:
    """Strip CUDA device state from one model before it is carried into a hosted payload.

    Numerai's hosted runtime is CPU-only, and a model fitted on a GPU pickles both its
    tensors and any recorded `torch.device` with CUDA locations, so the payload either
    fails to deserialize (tensors) or pushes its inputs to a missing GPU (device
    attributes) there. Torch is read from `sys.modules` instead of imported, so this
    module stays import-light and a package with no torch component does nothing.
    Mutating in place is safe because `build_submission_pickle` loads its own components
    and never predicts with them locally afterwards; local and remote GPU inference load
    their own copies and are untouched.
    """

    torch = sys.modules.get("torch")
    if torch is None:
        return
    seen: set[int] = set()
    pending: list[tuple[Any, int]] = [(model, 0)]
    while pending:
        current, depth = pending.pop()
        attributes = getattr(current, "__dict__", None)
        if depth > _TORCH_SCAN_DEPTH or id(current) in seen or not isinstance(attributes, dict):
            continue
        seen.add(id(current))
        for key, value in list(attributes.items()):
            if isinstance(value, torch.nn.Module):
                # `Module.to` already recurses into submodules, parameters, and buffers.
                value.to("cpu")
                value.eval()
            elif isinstance(value, torch.Tensor):
                attributes[key] = value.cpu()
                continue
            elif isinstance(value, torch.device):
                attributes[key] = torch.device("cpu")
                continue
            elif isinstance(value, ModuleType):
                # Skip: an imported module pickles by reference and its namespace is huge.
                continue
            pending.append((value, depth + 1))


def _prediction_inversion(model: Any) -> str:
    transform = model.__dict__.get("_target_transform") if isinstance(model, TargetTransformWrapper) else None
    try:
        return prediction_inversion_kind(transform)
    except TrainingModelError as exc:
        raise ServingUnsupportedConfigError(f"serving_model_upload_target_transform_not_invertible:{exc}") from exc


def _maybe_lightgbm_booster(model: Any) -> Any:
    inner = model
    if isinstance(inner, TargetTransformWrapper):
        # The wrapper only reshapes labels at fit time, so the booster underneath is
        # what predicts. Unwrapping keeps transformed LGBM models on the lean path.
        inner = inner.__dict__.get("_model")
        if inner is None:
            raise ServingUnsupportedConfigError("serving_model_upload_model_unwrap_failed")
    if type(inner).__module__ == _LGBM_WRAPPER_MODULE:
        inner = inner.__dict__.get("_model")
        if inner is None:
            raise ServingUnsupportedConfigError("serving_model_upload_model_unwrap_failed")
    elif not type(inner).__module__.startswith("lightgbm"):
        return None
    booster = getattr(inner, "booster_", None)
    if booster is None:
        raise ServingUnsupportedConfigError("serving_model_upload_model_unwrap_failed")
    return booster


__all__ = [
    "HOSTED_RUNTIME_DISTRIBUTIONS",
    "HostedBlendPayload",
    "HostedComponentPayload",
    "HostedPickleBuild",
    "build_hosted_component_payload",
    "build_pickled_predictor",
    "hosted_by_value_modules",
    "hosted_component_blockers",
    "resolve_live_benchmark_column",
    "verify_hosted_payload",
]
