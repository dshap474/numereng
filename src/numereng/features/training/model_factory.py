"""Model factory for training pipelines."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

from numereng.features.models.lgbm import LGBMRegressor
from numereng.features.training.errors import TrainingModelError
from numereng.features.training.target_transforms import TargetTransformWrapper

_BUILTIN_MODELS = {"LGBMRegressor": LGBMRegressor}


def _resolve_custom_models_root() -> Path:
    """Return the repository-local custom model directory."""
    return (Path(__file__).resolve().parents[1] / "models" / "custom_models").resolve()


def _load_model_registry(module_path: Path) -> dict[str, Any]:
    """Load a module and return its MODEL_REGISTRY mapping."""
    module_name = "numereng_custom_" + hashlib.md5(str(module_path.resolve()).encode("utf-8")).hexdigest()[:12]

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise TrainingModelError(f"training_model_custom_module_invalid:{module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001
        raise TrainingModelError(f"training_model_custom_module_load_failed:{module_path}") from exc

    registry = getattr(module, "MODEL_REGISTRY", None)
    if not isinstance(registry, dict):
        raise TrainingModelError(f"training_model_registry_missing_or_invalid:{module_path}")
    return registry


def _iter_module_paths(module_path: str | None) -> list[Path]:
    """Return candidate plugin module paths."""
    root = _resolve_custom_models_root()
    if module_path is None:
        if not root.exists():
            return []
        return [p for p in root.rglob("*.py") if p.name != "__init__.py"]

    explicit = Path(module_path)
    candidates: list[Path] = []

    if explicit.is_absolute():
        candidates.append(explicit)
    else:
        for base in (root, Path.cwd()):
            resolved = (base / explicit).resolve()
            candidates.append(resolved)
            if resolved.suffix != ".py":
                candidates.append((resolved.with_suffix(".py")).resolve())

    normalized: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate.exists():
            key = candidate.resolve()
            if key not in seen:
                normalized.append(key)
                seen.add(key)

    if normalized:
        return normalized

    raise TrainingModelError(f"training_model_custom_module_not_found:{module_path}")


def _resolve_model_class(
    model_type: str,
    model_config: dict[str, Any] | None,
) -> Any:
    """Resolve a model class from built-ins or plugin registries."""
    if model_type in _BUILTIN_MODELS:
        return _BUILTIN_MODELS[model_type]

    plugin_model_config = model_config or {}
    module_path = plugin_model_config.get("module_path")
    if module_path is not None and not isinstance(module_path, str):
        raise TrainingModelError("training_model_invalid_module_path")

    for module_file in _iter_module_paths(module_path):
        registry = _load_model_registry(module_file)
        model_cls = registry.get(model_type)
        if model_cls is None:
            continue
        return model_cls

    if module_path is not None:
        raise TrainingModelError(f"training_model_type_not_supported:{model_type}")

    raise TrainingModelError(
        f"training_model_type_not_supported:{model_type}. No matching model found in custom models."
    )


def _validate_model_cls(model_type: str, model_cls: Any) -> None:
    if not callable(model_cls):
        raise TrainingModelError(f"training_model_invalid_registry_entry:{model_type}")
    if not hasattr(model_cls, "fit") or not hasattr(model_cls, "predict"):
        raise TrainingModelError(f"training_model_invalid_model_class:{model_type}")


def build_model(
    model_type: str,
    model_params: dict[str, Any],
    model_config: dict[str, Any] | None = None,
    *,
    feature_cols: list[str] | None = None,
) -> Any:
    """Create configured model instance from typed model config."""
    resolved_model_config = model_config or {}
    model_cls = _resolve_model_class(model_type, resolved_model_config)
    _validate_model_cls(model_type, model_cls)

    model: Any = model_cls(feature_cols=feature_cols, **model_params)

    target_transform = resolved_model_config.get("target_transform")
    if target_transform:
        model = TargetTransformWrapper(model, target_transform)

    return model
