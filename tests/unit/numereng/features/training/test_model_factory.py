from __future__ import annotations

import sys
import types
from collections.abc import Sized
from pathlib import Path
from typing import Any, cast

import pytest

from numereng.features.training import model_factory
from numereng.features.training.errors import TrainingModelError


class _DummyModel:
    def __init__(self, feature_cols: list[str] | None = None, **_: object) -> None:
        self.feature_cols = feature_cols

    def fit(self, *_: object, **__: object) -> _DummyModel:
        return self

    def predict(self, *_: object, **__: object) -> list[float]:
        return []


def _write_model_file(path: Path, class_name: str = "PluginModel") -> None:
    path.write_text(
        f"""
class {class_name}:
    def __init__(self, feature_cols=None, **kwargs):
        self.feature_cols = feature_cols
        self.kwargs = kwargs

    def fit(self, *args, **kwargs):
        return self

    def predict(self, X):
        return []

MODEL_REGISTRY = {{{class_name!r}: {class_name}}}
"""
    )


def _write_fake_backend_module(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    class_name: str,
    with_impl: bool,
) -> None:
    module = types.ModuleType(module_name)
    if with_impl:

        class _BackendModel:
            def __init__(self, **_: object) -> None:
                pass

            def fit(self, *_: object, **__: object) -> _BackendModel:
                return self

            def predict(self, X: object, **__: object) -> list[float]:
                return [0.0] * len(cast(Sized, X))

        setattr(module, class_name, _BackendModel)
    monkeypatch.setitem(sys.modules, module_name, module)


def _custom_model_path(model_name: str) -> Path:
    return Path(model_factory.__file__).resolve().parents[1] / "models" / "custom_models" / f"{model_name}.py"


def _repo_custom_models_root() -> Path:
    return Path(model_factory.__file__).resolve().parents[1] / "models" / "custom_models"


def test_build_model_uses_builtin_model_registry() -> None:
    original = model_factory._BUILTIN_MODELS["LGBMRegressor"]
    model_factory._BUILTIN_MODELS["LGBMRegressor"] = cast(Any, _DummyModel)
    try:
        model = model_factory.build_model("LGBMRegressor", {"alpha": 1}, {})
        assert isinstance(model, _DummyModel)
        assert model.feature_cols is None
    finally:
        model_factory._BUILTIN_MODELS["LGBMRegressor"] = cast(Any, original)


def test_build_model_with_explicit_module_path(tmp_path: Path) -> None:
    module_path = tmp_path / "custom_model.py"
    _write_model_file(module_path, "ExplicitModel")

    model = model_factory.build_model(
        "ExplicitModel",
        {"alpha": 1},
        {"module_path": str(module_path)},
    )
    assert model.__class__.__name__ == "ExplicitModel"
    assert hasattr(model, "predict")


def test_normalize_model_params_resolves_bare_tabpfn_checkpoint_into_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache" / "tabpfn"
    monkeypatch.delenv("TABPFN_MODEL_CACHE_DIR", raising=False)
    monkeypatch.setattr(model_factory, "_resolve_store_root", lambda: tmp_path)

    params = model_factory._normalize_model_params(
        "TabPFNRegressor",
        {"model_path": "tabpfn-v2-regressor.ckpt"},
    )

    assert params["model_path"] == str((cache_dir / "tabpfn-v2-regressor.ckpt").resolve())
    assert cache_dir.is_dir()
    assert model_factory.os.environ["TABPFN_MODEL_CACHE_DIR"] == str(cache_dir.resolve())


def test_build_model_passes_normalized_tabpfn_model_path_to_plugin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache" / "tabpfn"
    monkeypatch.delenv("TABPFN_MODEL_CACHE_DIR", raising=False)
    monkeypatch.setattr(model_factory, "_resolve_store_root", lambda: tmp_path)
    plugin_path = tmp_path / "tabpfn_plugin.py"
    _write_model_file(plugin_path, "TabPFNRegressor")

    model = model_factory.build_model(
        "TabPFNRegressor",
        {"model_path": "tabpfn-v2-regressor.ckpt"},
        {"module_path": str(plugin_path)},
    )

    assert model.kwargs["model_path"] == str((cache_dir / "tabpfn-v2-regressor.ckpt").resolve())


def test_build_model_uses_explicit_store_root_for_tabpfn_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_root = tmp_path / "custom-store"
    cache_dir = store_root / "cache" / "tabpfn"
    plugin_path = tmp_path / "tabpfn_plugin.py"
    monkeypatch.delenv("TABPFN_MODEL_CACHE_DIR", raising=False)
    monkeypatch.setattr(model_factory, "_resolve_store_root", lambda: tmp_path / "repo-default")
    _write_model_file(plugin_path, "TabPFNRegressor")

    model = model_factory.build_model(
        "TabPFNRegressor",
        {"model_path": "tabpfn-v2-regressor.ckpt"},
        {"module_path": str(plugin_path)},
        store_root=store_root,
    )

    assert model.kwargs["model_path"] == str((cache_dir / "tabpfn-v2-regressor.ckpt").resolve())
    assert model_factory.os.environ["TABPFN_MODEL_CACHE_DIR"] == str(cache_dir.resolve())


def test_normalize_model_params_preserves_explicit_relative_tabpfn_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    explicit_relative = Path("artifacts") / "tabpfn" / "custom.ckpt"
    monkeypatch.delenv("TABPFN_MODEL_CACHE_DIR", raising=False)
    monkeypatch.setattr(model_factory, "_resolve_store_root", lambda: tmp_path)

    params = model_factory._normalize_model_params(
        "TabPFNRegressor",
        {"model_path": str(explicit_relative)},
    )

    assert params["model_path"] == str((tmp_path / explicit_relative).resolve())


def test_normalize_model_params_binds_tabpfn_auto_to_project_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache" / "tabpfn"
    monkeypatch.delenv("TABPFN_MODEL_CACHE_DIR", raising=False)
    monkeypatch.setattr(model_factory, "_resolve_store_root", lambda: tmp_path)

    params = model_factory._normalize_model_params(
        "TabPFNRegressor",
        {"model_path": "auto"},
    )

    assert params["model_path"] == "auto"
    assert cache_dir.is_dir()
    assert model_factory.os.environ["TABPFN_MODEL_CACHE_DIR"] == str(cache_dir.resolve())


def test_normalize_model_params_respects_explicit_tabpfn_cache_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configured_cache = tmp_path / "external-tabpfn-cache"
    monkeypatch.setenv("TABPFN_MODEL_CACHE_DIR", str(configured_cache))
    monkeypatch.setattr(model_factory, "_resolve_store_root", lambda: tmp_path / "ignored-store-root")

    params = model_factory._normalize_model_params(
        "TabPFNRegressor",
        {"model_path": "tabpfn-v2-regressor.ckpt"},
    )

    assert params["model_path"] == str((configured_cache / "tabpfn-v2-regressor.ckpt").resolve())
    assert configured_cache.is_dir()


def test_build_model_discover_module_from_custom_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "custom_models"
    root.mkdir()
    plugin_path = root / "discovered.py"
    _write_model_file(plugin_path, "DiscoveredModel")
    monkeypatch.setattr(model_factory, "_resolve_custom_models_root", lambda: root)

    model = model_factory.build_model("DiscoveredModel", {"alpha": 2}, {})
    assert model.__class__.__name__ == "DiscoveredModel"
    assert hasattr(model, "fit")


def test_build_model_xgboost_from_explicit_module_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _write_fake_backend_module(monkeypatch, "xgboost", "XGBRegressor", True)

    model = model_factory.build_model(
        "XGBoostRegressor",
        {"n_estimators": 100},
        {"module_path": str(_custom_model_path("xgboost_model"))},
    )
    assert model.__class__.__name__ == "XGBoostRegressor"


def test_build_model_xgboost_from_custom_root(monkeypatch: pytest.MonkeyPatch) -> None:
    _write_fake_backend_module(monkeypatch, "xgboost", "XGBRegressor", True)
    monkeypatch.setattr(model_factory, "_resolve_custom_models_root", _repo_custom_models_root)

    model = model_factory.build_model("XGBoostRegressor", {"n_estimators": 200}, {})
    assert model.__class__.__name__ == "XGBoostRegressor"


def test_build_model_xgboost_missing_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    _write_fake_backend_module(monkeypatch, "xgboost", "XGBRegressor", False)

    with pytest.raises(
        TrainingModelError,
        match="training_model_backend_missing_xgboost",
    ):
        model_factory.build_model(
            "XGBoostRegressor",
            {"n_estimators": 100},
            {"module_path": str(_custom_model_path("xgboost_model"))},
        )


def test_build_model_catboost_from_explicit_module_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _write_fake_backend_module(monkeypatch, "catboost", "CatBoostRegressor", True)

    model = model_factory.build_model(
        "CatBoostRegressor",
        {"iterations": 200},
        {"module_path": str(_custom_model_path("catboost_model"))},
    )
    assert model.__class__.__name__ == "CatBoostRegressor"


def test_build_model_catboost_from_custom_root(monkeypatch: pytest.MonkeyPatch) -> None:
    _write_fake_backend_module(monkeypatch, "catboost", "CatBoostRegressor", True)
    monkeypatch.setattr(model_factory, "_resolve_custom_models_root", _repo_custom_models_root)

    model = model_factory.build_model("CatBoostRegressor", {"iterations": 200}, {})
    assert model.__class__.__name__ == "CatBoostRegressor"


def test_build_model_does_not_fall_back_to_packaged_repo_custom_models(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    empty_root = tmp_path / "custom_models"
    empty_root.mkdir()
    _write_fake_backend_module(monkeypatch, "xgboost", "XGBRegressor", True)
    monkeypatch.setattr(model_factory, "_resolve_custom_models_root", lambda: empty_root)

    with pytest.raises(TrainingModelError, match="training_model_type_not_supported"):
        model_factory.build_model("XGBoostRegressor", {"n_estimators": 200}, {})


def test_build_model_catboost_missing_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    _write_fake_backend_module(monkeypatch, "catboost", "CatBoostRegressor", False)

    with pytest.raises(
        TrainingModelError,
        match="training_model_backend_missing_catboost",
    ):
        model_factory.build_model(
            "CatBoostRegressor",
            {"iterations": 200},
            {"module_path": str(_custom_model_path("catboost_model"))},
        )


def test_build_model_missing_explicit_module_path_is_error(tmp_path: Path) -> None:
    missing_module = tmp_path / "does_not_exist.py"
    with pytest.raises(
        TrainingModelError,
        match="training_model_custom_module_not_found",
    ):
        model_factory.build_model(
            "MissingModel",
            {},
            {"module_path": str(missing_module)},
        )


def test_build_model_invalid_module_path_type_is_error() -> None:
    with pytest.raises(
        TrainingModelError,
        match="training_model_invalid_module_path",
    ):
        model_factory.build_model(
            "MissingModel",
            {},
            {"module_path": 123},
        )


def test_build_model_invalid_module_file_is_error(tmp_path: Path) -> None:
    invalid_module = tmp_path / "not_a_python_module"
    invalid_module.mkdir()

    with pytest.raises(
        TrainingModelError,
        match="training_model_custom_module_invalid",
    ):
        model_factory.build_model(
            "MissingModel",
            {},
            {"module_path": str(invalid_module)},
        )


def test_build_model_module_load_failure_is_error(tmp_path: Path) -> None:
    broken_module = tmp_path / "broken_model.py"
    broken_module.write_text("raise RuntimeError('boom')\n", encoding="utf-8")

    with pytest.raises(
        TrainingModelError,
        match="training_model_custom_module_load_failed",
    ):
        model_factory.build_model(
            "BrokenModel",
            {},
            {"module_path": str(broken_module)},
        )


def test_build_model_invalid_registry_entry_is_error(tmp_path: Path) -> None:
    invalid_registry_module = tmp_path / "invalid_registry.py"
    invalid_registry_module.write_text(
        'MODEL_REGISTRY = {"BadModel": 123}\n',
        encoding="utf-8",
    )

    with pytest.raises(
        TrainingModelError,
        match="training_model_invalid_registry_entry",
    ):
        model_factory.build_model(
            "BadModel",
            {},
            {"module_path": str(invalid_registry_module)},
        )


def test_build_model_unsupported_type_is_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(model_factory, "_resolve_custom_models_root", lambda: tmp_path)
    with pytest.raises(TrainingModelError, match="training_model_type_not_supported"):
        model_factory.build_model("Unknown", {}, {})
