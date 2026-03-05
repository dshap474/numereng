"""Config helpers for unified training engine resolution."""

from __future__ import annotations

from typing import Literal

from numereng.features.training.errors import TrainingConfigError
from numereng.features.training.strategies.core.protocol import TrainingProfile

OFFICIAL_WINDOW_SIZE_ERAS = 156


def as_config_mapping(value: object, *, field: str) -> dict[str, object]:
    """Validate one config section as a mapping."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    raise TrainingConfigError(f"training_engine_field_not_mapping:{field}")


def infer_target_horizon(data_config: dict[str, object]) -> Literal["20d", "60d"]:
    """Resolve target horizon from explicit config or target name."""
    explicit_horizon = data_config.get("target_horizon")
    if explicit_horizon is not None:
        return _coerce_target_horizon(explicit_horizon)

    target_col = str(data_config.get("target_col", "target"))
    if "60" in target_col:
        return "60d"
    if "20" in target_col or target_col == "target":
        return "20d"
    raise TrainingConfigError("training_engine_target_horizon_ambiguous")


def default_embargo_for_horizon(horizon: Literal["20d", "60d"]) -> int:
    """Return default official-style embargo per target horizon."""
    if horizon == "60d":
        return 16
    return 8


def parse_training_profile(value: object) -> TrainingProfile:
    """Coerce one training profile value."""
    profile = str(value)
    if profile not in {"simple", "purged_walk_forward", "submission"}:
        raise TrainingConfigError(f"training_profile_unknown:{profile}")
    return profile  # type: ignore[return-value]


def parse_legacy_training_engine_mode(value: object) -> TrainingProfile:
    """Map legacy training engine mode values into canonical training profiles."""
    mode = str(value)
    if mode == "official":
        return "purged_walk_forward"
    if mode == "full_history":
        return "submission"
    if mode == "custom":
        raise TrainingConfigError("training_profile_legacy_custom_not_supported")
    raise TrainingConfigError(f"training_engine_unknown:{mode}")


def parse_training_engine_mode(value: object) -> TrainingProfile:
    """Backward-compatible alias for legacy callers."""
    return parse_legacy_training_engine_mode(value)


def coerce_optional_positive_int(value: object, *, field: str) -> int | None:
    """Coerce optional positive integer config values."""
    if value is None:
        return None

    if isinstance(value, bool):
        raise TrainingConfigError(f"training_engine_invalid_integer:{field}")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError as exc:
            raise TrainingConfigError(f"training_engine_invalid_integer:{field}") from exc
    else:
        raise TrainingConfigError(f"training_engine_invalid_integer:{field}")

    if parsed < 1:
        raise TrainingConfigError(f"training_engine_invalid_integer:{field}")
    return parsed


def _coerce_target_horizon(value: object) -> Literal["20d", "60d"]:
    horizon = str(value).lower()
    if horizon not in {"20d", "60d"}:
        raise TrainingConfigError("training_engine_target_horizon_invalid")
    return horizon  # type: ignore[return-value]
