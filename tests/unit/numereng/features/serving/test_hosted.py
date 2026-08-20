"""Unit tests for the Numerai-hosted payload build, dependency proof, and benchmark resolution.

USAGE:
    uv run pytest tests/unit/numereng/features/serving/test_hosted.py -q

Fixtures here are deliberately synthetic: a tiny in-memory custom module stands in for a
real custom-model file, so no run under `.numereng/runs/` is read and no model is trained.
"""

from __future__ import annotations

import io
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from numereng.features.serving.contracts import ServingBlendRule, ServingNeutralizationSpec
from numereng.features.serving.hosted import (
    HOSTED_RUNTIME_DISTRIBUTIONS,
    _move_torch_state_to_cpu,
    build_pickled_predictor,
    hosted_component_blockers,
    resolve_live_benchmark_column,
    verify_hosted_payload,
)
from numereng.features.serving.runtime import FittedComponent, ServingUnsupportedConfigError
from numereng.features.training.model_factory import import_custom_model_module
from numereng.features.training.target_transforms import TargetTransformWrapper

# --------------------------------------------------------------------------- #
# Synthetic fixtures
# --------------------------------------------------------------------------- #

_ERA_AWARE_MODEL = """
import numpy as np


class EraAwareModel:
    accepts_era = True

    def __init__(self, scale=2.0):
        self.scale = scale

    def fit(self, X, y, era=None, **kwargs):
        if era is None:
            raise ValueError("model_requires_era")
        return self

    def predict(self, X, era=None, **kwargs):
        if era is None:
            raise ValueError("model_requires_era")
        return np.asarray(X["bench_raw"], dtype=float) * self.scale


MODEL_REGISTRY = {"EraAwareModel": EraAwareModel}
"""

_LRU_CACHE_MODEL = """
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def _scale():
    return 2.0


class CachedModel:
    def predict(self, X, **kwargs):
        return np.asarray(X["feature_a"], dtype=float) * _scale()


MODEL_REGISTRY = {"CachedModel": CachedModel}
"""

_NUMERENG_LEAK_MODEL = """
import numpy as np
from numereng.features.training.repo import default_benchmark_model


class LeakModel:
    def predict(self, X, **kwargs):
        _ = default_benchmark_model("v5.3")
        return np.asarray(X["feature_a"], dtype=float)


MODEL_REGISTRY = {"LeakModel": LeakModel}
"""

_FOREIGN_DEPENDENCY_MODEL = """
import numpy as np
import pyarrow


class ForeignModel:
    def __init__(self):
        self.schema = pyarrow.schema([("x", pyarrow.float64())])

    def predict(self, X, **kwargs):
        return np.asarray(X["feature_a"], dtype=float)


MODEL_REGISTRY = {"ForeignModel": ForeignModel}
"""


def _load_custom_model(tmp_path: Path, *, source: str, name: str) -> Any:
    """Load one synthetic custom module through the real model-factory importer.

    Going through the factory keeps the fixture on the production naming scheme, so the
    module lands in `sys.modules` under the same path-derived synthetic name a real
    plugin gets.
    """

    module_path = tmp_path / f"{name}.py"
    module_path.write_text(source, encoding="utf-8")
    module = import_custom_model_module(module_path)
    return next(iter(module.MODEL_REGISTRY.values()))()


def _write_benchmark_provenance(
    tmp_path: Path,
    *,
    name: str = "official_v53_lgbm_ender20",
    kind: str = "official_numerai_benchmark",
) -> str:
    baseline_dir = tmp_path / "baselines" / "active_benchmark"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    (baseline_dir / "benchmark.json").write_text(json.dumps({"name": name, "kind": kind}), encoding="utf-8")
    return str(baseline_dir / "predictions.parquet")


def _component(
    *,
    model: Any,
    baseline_predictions_path: str | None = None,
    feature_cols: tuple[str, ...] = ("feature_a", "bench_raw"),
) -> FittedComponent:
    return FittedComponent(
        component_id="nn",
        weight=1.0,
        model=model,
        id_col="id",
        era_col="era",
        feature_cols=feature_cols,
        baseline_col=None if baseline_predictions_path is None else "bench_raw",
        baseline_name=None if baseline_predictions_path is None else "bench_raw",
        baseline_predictions_path=baseline_predictions_path,
        baseline_pred_col="prediction",
    )


class _BlockNumereng:
    """Import blocker proving the payload never needs `numereng` at unpickle time.

    The prefix covers the synthetic `numereng_custom_<md5(path)>` plugin modules too:
    that name hashes the training machine's absolute path, so the hosted container can
    never resolve it and the payload must not reference it.
    """

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
        if fullname.startswith("numereng"):
            raise ImportError(f"numereng_import_blocked:{fullname}")
        return None


def _load_without_numereng(blob: bytes) -> Any:
    saved = {name: module for name, module in sys.modules.items() if name.startswith("numereng")}
    blocker = _BlockNumereng()
    for name in saved:
        del sys.modules[name]
    sys.meta_path.insert(0, blocker)
    try:
        return pickle.loads(blob)
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(saved)


class _RecordingUnpickler(pickle.Unpickler):
    """Unpickler that records every module the payload asks for."""

    def __init__(self, blob: bytes) -> None:
        super().__init__(io.BytesIO(blob))
        self.observed: set[str] = set()

    def find_class(self, module: str, name: str) -> Any:
        self.observed.add(module)
        return super().find_class(module, name)


def _observed_modules(blob: bytes) -> set[str]:
    unpickler = _RecordingUnpickler(blob)
    unpickler.load()
    return unpickler.observed


# --------------------------------------------------------------------------- #
# Benchmark column resolution
# --------------------------------------------------------------------------- #


def test_resolve_live_benchmark_column_reads_recorded_provenance(tmp_path: Path) -> None:
    path = _write_benchmark_provenance(tmp_path)

    assert resolve_live_benchmark_column(baseline_predictions_path=path, data_version="v5.3") == "v53_lgbm_ender20"


def test_resolve_live_benchmark_column_requires_a_baseline_path() -> None:
    with pytest.raises(ServingUnsupportedConfigError, match="serving_model_upload_benchmark_not_resolvable"):
        resolve_live_benchmark_column(baseline_predictions_path=None, data_version="v5.3")


def test_resolve_live_benchmark_column_requires_recorded_provenance(tmp_path: Path) -> None:
    with pytest.raises(ServingUnsupportedConfigError, match="serving_model_upload_benchmark_provenance_missing"):
        resolve_live_benchmark_column(
            baseline_predictions_path=str(tmp_path / "predictions.parquet"),
            data_version="v5.3",
        )


def test_resolve_live_benchmark_column_rejects_non_official_baselines(tmp_path: Path) -> None:
    path = _write_benchmark_provenance(tmp_path, name="blend_v1", kind="local_blend")

    with pytest.raises(ServingUnsupportedConfigError, match="serving_model_upload_benchmark_not_official"):
        resolve_live_benchmark_column(baseline_predictions_path=path, data_version="v5.3")


def test_resolve_live_benchmark_column_rejects_data_version_mismatch(tmp_path: Path) -> None:
    path = _write_benchmark_provenance(tmp_path, name="official_v50_lgbm_ct_blend")

    with pytest.raises(ServingUnsupportedConfigError, match="serving_model_upload_benchmark_data_version_mismatch"):
        resolve_live_benchmark_column(baseline_predictions_path=path, data_version="v5.3")


# --------------------------------------------------------------------------- #
# Payload build
# --------------------------------------------------------------------------- #


def test_build_pickled_predictor_serves_custom_module_with_live_benchmark(tmp_path: Path) -> None:
    model = _load_custom_model(tmp_path, source=_ERA_AWARE_MODEL, name="era_aware")
    component = _component(model=model, baseline_predictions_path=_write_benchmark_provenance(tmp_path))

    build = build_pickled_predictor(
        fitted_components=(component,),
        blend_rule=ServingBlendRule(),
        neutralization=None,
        pickle_path=tmp_path / "model.pkl",
        data_version="v5.3",
    )

    assert build.serialization_kinds == ("cloudpickle_model",)
    assert build.benchmark_model_cols == (("nn", "v53_lgbm_ender20"),)
    assert build.uses_baseline_inputs is True
    assert "hosted_blend_rank_reimplemented" in build.drift_risks
    assert "hosted_dependency_set_from_observed_unpickle" in build.drift_risks

    predictor = _load_without_numereng(build.pickle_path.read_bytes())
    live = pd.DataFrame({"id": ["live_1", "live_2"], "feature_a": [0.1, 0.4]})
    benchmark = pd.DataFrame({"id": ["live_1", "live_2"], "v53_lgbm_ender20": [0.25, 0.75]})

    submission = predictor(live, benchmark)

    assert list(submission.columns) == ["prediction"]
    assert submission.index.equals(live.index)
    assert submission["prediction"].tolist() == pytest.approx([0.5, 1.0])


def test_build_pickled_predictor_never_references_the_synthetic_custom_module(tmp_path: Path) -> None:
    """The hosted payload must carry the custom model class by value.

    The training-time module name hashes an absolute path on the training machine, so a
    by-reference class would make the payload unloadable inside Numerai's container.
    """

    model = _load_custom_model(tmp_path, source=_ERA_AWARE_MODEL, name="era_aware")
    synthetic_module = type(model).__module__
    assert synthetic_module.startswith("numereng_custom_")
    assert synthetic_module in sys.modules

    build = build_pickled_predictor(
        fitted_components=(_component(model=model, baseline_predictions_path=_write_benchmark_provenance(tmp_path)),),
        blend_rule=ServingBlendRule(),
        neutralization=None,
        pickle_path=tmp_path / "model.pkl",
        data_version="v5.3",
    )
    blob = build.pickle_path.read_bytes()

    # The class travels by value, so no `numereng*` module is ever looked up or imported.
    assert not [name for name in _observed_modules(blob) if name.startswith("numereng")]
    assert not [root for root in verify_hosted_payload(blob) if root.startswith("numereng")]
    predictor = _load_without_numereng(blob)
    live = pd.DataFrame({"id": ["live_1", "live_2"], "feature_a": [0.1, 0.4]})
    benchmark = pd.DataFrame({"id": ["live_1", "live_2"], "v53_lgbm_ender20": [0.25, 0.75]})

    assert predictor(live, benchmark)["prediction"].tolist() == pytest.approx([0.5, 1.0])


def test_build_pickled_predictor_carries_lru_cached_plugin_helpers_by_value(tmp_path: Path) -> None:
    """`register_pickle_by_value` does not reach `@lru_cache` module functions.

    Those wrappers pickle as a plain global reference into the synthetic plugin module,
    so the payload would fail to unpickle inside Numerai even though the model class
    itself travelled by value.
    """

    model = _load_custom_model(tmp_path, source=_LRU_CACHE_MODEL, name="cached")

    build = build_pickled_predictor(
        fitted_components=(_component(model=model, feature_cols=("feature_a",)),),
        blend_rule=ServingBlendRule(),
        neutralization=None,
        pickle_path=tmp_path / "model.pkl",
        data_version="v5.3",
    )
    blob = build.pickle_path.read_bytes()

    assert not [name for name in _observed_modules(blob) if name.startswith("numereng")]
    predictor = _load_without_numereng(blob)
    live = pd.DataFrame({"id": ["live_1", "live_2"], "feature_a": [0.1, 0.4]})

    assert predictor(live, None)["prediction"].tolist() == pytest.approx([0.5, 1.0])


def test_build_pickled_predictor_rejects_missing_live_benchmark_at_predict_time(tmp_path: Path) -> None:
    model = _load_custom_model(tmp_path, source=_ERA_AWARE_MODEL, name="era_aware")
    component = _component(model=model, baseline_predictions_path=_write_benchmark_provenance(tmp_path))
    build = build_pickled_predictor(
        fitted_components=(component,),
        blend_rule=ServingBlendRule(),
        neutralization=None,
        pickle_path=tmp_path / "model.pkl",
        data_version="v5.3",
    )
    predictor = pd.read_pickle(build.pickle_path)
    live = pd.DataFrame({"id": ["live_1", "live_2"], "feature_a": [0.1, 0.4]})

    with pytest.raises(ValueError, match="serving_live_benchmark_models_missing:v53_lgbm_ender20"):
        predictor(live, None)

    with pytest.raises(ValueError, match="serving_live_benchmark_column_missing:v53_lgbm_ender20"):
        predictor(live, pd.DataFrame({"id": ["live_1", "live_2"], "v50_lgbm_ct_blend": [0.25, 0.75]}))


def test_build_pickled_predictor_rejects_neutralized_packages(tmp_path: Path) -> None:
    model = _load_custom_model(tmp_path, source=_ERA_AWARE_MODEL, name="era_aware")
    component = _component(model=model, baseline_predictions_path=_write_benchmark_provenance(tmp_path))

    with pytest.raises(ServingUnsupportedConfigError, match="serving_model_upload_neutralization_not_supported"):
        build_pickled_predictor(
            fitted_components=(component,),
            blend_rule=ServingBlendRule(),
            neutralization=ServingNeutralizationSpec(enabled=True),
            pickle_path=tmp_path / "model.pkl",
            data_version="v5.3",
        )


def test_build_pickled_predictor_rejects_payloads_that_need_numereng(tmp_path: Path) -> None:
    model = _load_custom_model(tmp_path, source=_NUMERENG_LEAK_MODEL, name="leak")

    with pytest.raises(ServingUnsupportedConfigError, match="serving_model_upload_payload_requires_numereng"):
        build_pickled_predictor(
            fitted_components=(_component(model=model, feature_cols=("feature_a",)),),
            blend_rule=ServingBlendRule(),
            neutralization=None,
            pickle_path=tmp_path / "model.pkl",
            data_version="v5.3",
        )


def test_build_pickled_predictor_rejects_dependencies_absent_from_the_hosted_runtime(tmp_path: Path) -> None:
    model = _load_custom_model(tmp_path, source=_FOREIGN_DEPENDENCY_MODEL, name="foreign")

    with pytest.raises(ServingUnsupportedConfigError, match="serving_model_upload_payload_dependency_unavailable"):
        build_pickled_predictor(
            fitted_components=(_component(model=model, feature_cols=("feature_a",)),),
            blend_rule=ServingBlendRule(),
            neutralization=None,
            pickle_path=tmp_path / "model.pkl",
            data_version="v5.3",
        )


def test_build_pickled_predictor_declares_torch_when_the_payload_needs_it(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    class _TorchModel:
        def __init__(self) -> None:
            self.layer = torch.nn.Linear(1, 1)

        def predict(self, X, **kwargs):
            with torch.no_grad():
                values = torch.as_tensor(X[["feature_a"]].to_numpy(dtype="float32"))
                return self.layer(values).numpy().ravel()

    module_name = "numereng_custom_torch_probe"
    module = type(sys)(module_name)
    module.__dict__["_TorchModel"] = _TorchModel
    _TorchModel.__module__ = module_name
    sys.modules[module_name] = module

    build = build_pickled_predictor(
        fitted_components=(_component(model=_TorchModel(), feature_cols=("feature_a",)),),
        blend_rule=ServingBlendRule(),
        neutralization=None,
        pickle_path=tmp_path / "model.pkl",
        data_version="v5.3",
    )

    assert "torch==2.7.1" in build.python_requirements
    assert HOSTED_RUNTIME_DISTRIBUTIONS["torch"] == "torch==2.7.1"


def test_build_pickled_predictor_moves_gpu_fitted_torch_state_to_cpu(tmp_path: Path) -> None:
    """A GPU-fitted payload must carry no CUDA device state into Numerai's CPU-only runtime.

    CUDA cannot be allocated here, so the net records the device it was moved to and the
    wrapper holds a `torch.device("cuda")` the way a GPU-fitted custom model does.
    """

    torch = pytest.importorskip("torch")

    class _RecordingNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = torch.nn.Linear(1, 1)
            self.moved_to: list[str] = []

        def to(self, *args: Any, **kwargs: Any) -> Any:
            self.moved_to.append(str(args[0]) if args else str(kwargs.get("device")))
            return super().to(*args, **kwargs)

        def forward(self, values: Any) -> Any:
            return self.layer(values)

    class _GpuFittedModel:
        def __init__(self) -> None:
            self._torch = torch
            self._net = _RecordingNet()
            self._device = torch.device("cuda")
            self._scale = torch.ones(1)

        def predict(self, X: Any, **kwargs: Any) -> Any:
            with torch.no_grad():
                values = torch.as_tensor(X[["feature_a"]].to_numpy(dtype="float32")).to(self._device)
                return (self._net(values).ravel() * self._scale).numpy()

    module_name = "numereng_custom_gpu_probe"
    module = type(sys)(module_name)
    module.__dict__["_GpuFittedModel"] = _GpuFittedModel
    module.__dict__["_RecordingNet"] = _RecordingNet
    _GpuFittedModel.__module__ = module_name
    _RecordingNet.__module__ = module_name
    sys.modules[module_name] = module
    model = _GpuFittedModel()
    model._net.train()

    build = build_pickled_predictor(
        fitted_components=(_component(model=model, feature_cols=("feature_a",)),),
        blend_rule=ServingBlendRule(),
        neutralization=None,
        pickle_path=tmp_path / "model.pkl",
        data_version="v5.3",
    )

    assert model._net.moved_to == ["cpu"]
    assert model._net.training is False
    assert model._device == torch.device("cpu")
    assert model._scale.device.type == "cpu"

    predictor = _load_without_numereng(build.pickle_path.read_bytes())
    reloaded = predictor._components[0]["model"]
    assert reloaded._device == torch.device("cpu")
    assert all(parameter.device.type == "cpu" for parameter in reloaded._net.parameters())
    frame = pd.DataFrame({"feature_a": [0.1, 0.9], "id": ["a", "b"], "era": ["live", "live"]}).set_index("id")
    assert list(predictor(frame).columns) == ["prediction"]


def test_move_torch_state_to_cpu_leaves_a_package_without_torch_state_alone() -> None:
    """The device sweep must be inert for the components that carry no torch state."""

    class _Plain:
        def __init__(self) -> None:
            self.value = 1.0
            self.frame = pd.DataFrame({"a": [1.0]})

    model = _Plain()
    _move_torch_state_to_cpu(model)

    assert model.value == 1.0
    assert model.frame.equals(pd.DataFrame({"a": [1.0]}))


def test_verify_hosted_payload_reports_hosted_runtime_roots() -> None:
    import cloudpickle

    roots = verify_hosted_payload(cloudpickle.dumps(pd.DataFrame({"a": [1.0]})))

    assert "pandas" in roots


# --------------------------------------------------------------------------- #
# Capability probe
# --------------------------------------------------------------------------- #


def test_hosted_component_blockers_accepts_a_torch_style_baseline_component(tmp_path: Path) -> None:
    model = _load_custom_model(tmp_path, source=_ERA_AWARE_MODEL, name="era_aware")
    component = _component(model=model, baseline_predictions_path=_write_benchmark_provenance(tmp_path))

    assert hosted_component_blockers(component=component, data_version="v5.3") == ()


def test_hosted_component_blockers_reports_unresolvable_benchmarks(tmp_path: Path) -> None:
    model = _load_custom_model(tmp_path, source=_ERA_AWARE_MODEL, name="era_aware")
    component = _component(model=model, baseline_predictions_path=str(tmp_path / "predictions.parquet"))

    blockers = hosted_component_blockers(component=component, data_version="v5.3")

    assert any(item.startswith("serving_model_upload_benchmark_provenance_missing") for item in blockers)


def test_hosted_component_blockers_reports_unknown_target_transforms(tmp_path: Path) -> None:
    model = _load_custom_model(tmp_path, source=_ERA_AWARE_MODEL, name="era_aware")
    wrapped = TargetTransformWrapper(model, {"type": "invented_transform"})
    component = _component(model=wrapped, baseline_predictions_path=_write_benchmark_provenance(tmp_path))

    blockers = hosted_component_blockers(component=component, data_version="v5.3")

    assert any(item.startswith("serving_model_upload_target_transform_not_invertible") for item in blockers)


def test_hosted_component_blockers_accepts_the_residual_to_benchmark_transform(tmp_path: Path) -> None:
    model = _load_custom_model(tmp_path, source=_ERA_AWARE_MODEL, name="era_aware")
    wrapped = TargetTransformWrapper(model, {"type": "residual_to_benchmark", "benchmark_col": "bench_raw"})
    component = _component(model=wrapped, baseline_predictions_path=_write_benchmark_provenance(tmp_path))

    assert hosted_component_blockers(component=component, data_version="v5.3") == ()


def test_hosted_component_blockers_rejects_models_without_predict(tmp_path: Path) -> None:
    component = _component(model=object(), baseline_predictions_path=_write_benchmark_provenance(tmp_path))

    blockers = hosted_component_blockers(component=component, data_version="v5.3")

    assert any(item.startswith("serving_model_upload_model_type_not_supported") for item in blockers)
