"""Search-space parsing and config override helpers for HPO."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from numereng.features.hpo.contracts import HpoParameterSpec


class HpoSearchSpaceError(ValueError):
    """Raised when HPO search-space payloads are invalid."""


def resolve_search_space(
    *,
    base_config: dict[str, Any],
    raw_search_space: dict[str, dict[str, Any]] | None,
) -> tuple[HpoParameterSpec, ...]:
    """Resolve explicit or inferred search-space parameter specs."""

    if raw_search_space:
        specs = _parse_explicit_search_space(raw_search_space)
    else:
        specs = _infer_search_space(base_config)

    if not specs:
        raise HpoSearchSpaceError("hpo_search_space_empty")
    return tuple(specs)


def apply_param_overrides(base_config: dict[str, Any], *, params: dict[str, Any]) -> dict[str, Any]:
    """Apply path-based parameter overrides to a config clone."""

    updated: dict[str, Any] = deepcopy(base_config)
    for path, value in params.items():
        _set_path_value(updated, path=path, value=value)
    return updated


def _parse_explicit_search_space(raw_search_space: dict[str, dict[str, Any]]) -> list[HpoParameterSpec]:
    specs: list[HpoParameterSpec] = []

    for path, spec in raw_search_space.items():
        if not isinstance(path, str) or not path.strip():
            raise HpoSearchSpaceError("hpo_search_space_path_invalid")
        if not isinstance(spec, dict):
            raise HpoSearchSpaceError(f"hpo_search_space_spec_invalid:{path}")

        kind = str(spec.get("type", "")).strip().lower()
        if kind not in {"float", "int", "categorical"}:
            raise HpoSearchSpaceError(f"hpo_search_space_type_invalid:{path}")

        if kind == "categorical":
            raw_choices = spec.get("choices")
            if not isinstance(raw_choices, list) or not raw_choices:
                raise HpoSearchSpaceError(f"hpo_search_space_choices_invalid:{path}")
            choices = tuple(raw_choices)
            specs.append(
                HpoParameterSpec(
                    path=path,
                    kind="categorical",
                    choices=choices,
                )
            )
            continue

        low = spec.get("low")
        high = spec.get("high")
        if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
            raise HpoSearchSpaceError(f"hpo_search_space_bounds_invalid:{path}")

        step = spec.get("step")
        if step is not None and not isinstance(step, (int, float)):
            raise HpoSearchSpaceError(f"hpo_search_space_step_invalid:{path}")

        log = bool(spec.get("log", False))
        specs.append(
            HpoParameterSpec(
                path=path,
                kind="int" if kind == "int" else "float",
                low=int(low) if kind == "int" else float(low),
                high=int(high) if kind == "int" else float(high),
                step=int(step) if kind == "int" and isinstance(step, (int, float)) else step,
                log=log,
            )
        )

    return specs


def _infer_search_space(base_config: dict[str, Any]) -> list[HpoParameterSpec]:
    model = base_config.get("model")
    if not isinstance(model, dict):
        return []
    params = model.get("params")
    if not isinstance(params, dict):
        return []

    specs: list[HpoParameterSpec] = []
    for key, value in sorted(params.items()):
        path = f"model.params.{key}"
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            low_int = max(1, value // 2)
            high_int = max(low_int + 1, value * 2)
            specs.append(HpoParameterSpec(path=path, kind="int", low=low_int, high=high_int, step=1, log=False))
            continue
        if isinstance(value, float):
            if value == 0.0:
                low_float = -1.0
                high_float = 1.0
            else:
                magnitude = abs(value)
                low_float = value - magnitude
                high_float = value + magnitude
                if low_float == high_float:
                    high_float = low_float + 1.0
            specs.append(
                HpoParameterSpec(
                    path=path,
                    kind="float",
                    low=low_float,
                    high=high_float,
                    step=None,
                    log=bool(value > 0 and key in {"learning_rate", "reg_alpha", "reg_lambda"}),
                )
            )
    return specs


def _set_path_value(payload: dict[str, Any], *, path: str, value: Any) -> None:
    tokens = [token for token in path.split(".") if token]
    if not tokens:
        raise HpoSearchSpaceError("hpo_search_space_path_invalid")

    cursor: dict[str, Any] = payload
    for token in tokens[:-1]:
        existing = cursor.get(token)
        if existing is None:
            cursor[token] = {}
            existing = cursor[token]
        if not isinstance(existing, dict):
            raise HpoSearchSpaceError(f"hpo_search_space_path_conflict:{path}")
        cursor = existing
    cursor[tokens[-1]] = value
