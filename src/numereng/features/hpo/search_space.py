"""Search-space parsing and config override helpers for HPO."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from numereng.features.hpo.contracts import HpoParameterSpec


class HpoSearchSpaceError(ValueError):
    """Raised when HPO search-space payloads are invalid."""


def resolve_search_space(
    *,
    raw_search_space: dict[str, dict[str, Any]] | None,
) -> tuple[HpoParameterSpec, ...]:
    """Resolve one explicit search-space parameter mapping."""

    if not raw_search_space:
        raise HpoSearchSpaceError("hpo_search_space_required")
    specs = _parse_explicit_search_space(raw_search_space)
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
            specs.append(HpoParameterSpec(path=path, kind="categorical", choices=tuple(raw_choices)))
            continue

        low = spec.get("low")
        high = spec.get("high")
        if not isinstance(low, (int, float)) or isinstance(low, bool):
            raise HpoSearchSpaceError(f"hpo_search_space_bounds_invalid:{path}")
        if not isinstance(high, (int, float)) or isinstance(high, bool):
            raise HpoSearchSpaceError(f"hpo_search_space_bounds_invalid:{path}")

        step = spec.get("step")
        if step is not None and (not isinstance(step, (int, float)) or isinstance(step, bool)):
            raise HpoSearchSpaceError(f"hpo_search_space_step_invalid:{path}")

        log = bool(spec.get("log", False))
        if kind == "int":
            specs.append(
                HpoParameterSpec(
                    path=path,
                    kind="int",
                    low=int(low),
                    high=int(high),
                    step=int(step) if step is not None else 1,
                    log=log,
                )
            )
            continue

        specs.append(
            HpoParameterSpec(
                path=path,
                kind="float",
                low=float(low),
                high=float(high),
                step=float(step) if step is not None else None,
                log=log,
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
