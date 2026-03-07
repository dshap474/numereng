"""Resolution logic for unified training engine modes."""

from __future__ import annotations

from numereng.features.training.errors import TrainingConfigError
from numereng.features.training.strategies.core.config import (
    OFFICIAL_WINDOW_SIZE_ERAS,
    as_config_mapping,
    default_embargo_for_horizon,
    infer_target_horizon,
    parse_legacy_training_engine_mode,
    parse_training_profile,
)
from numereng.features.training.strategies.core.protocol import (
    TrainingEnginePlan,
    TrainingProfile,
)

_DEFAULT_PROFILE: TrainingProfile = "purged_walk_forward"


def resolve_training_engine(
    *,
    training_config: dict[str, object],
    data_config: dict[str, object],
    profile: TrainingProfile | None = None,
    engine_mode: str | None = None,
    window_size_eras: int | None = None,
    embargo_eras: int | None = None,
) -> TrainingEnginePlan:
    """Resolve the single training profile plan from config and runtime overrides."""
    _raise_if_legacy_method_config(training_config)

    engine_config = as_config_mapping(training_config.get("engine"), field="training.engine")
    config_profile: TrainingProfile | None = None
    legacy_config_profile: TrainingProfile | None = None
    legacy_runtime_profile: TrainingProfile | None = None

    if "profile" in engine_config:
        config_profile = parse_training_profile(engine_config["profile"])
    if "mode" in engine_config:
        legacy_config_profile = parse_legacy_training_engine_mode(engine_config["mode"])
    if engine_mode is not None:
        legacy_runtime_profile = parse_legacy_training_engine_mode(engine_mode)

    override_sources: list[str] = []
    if config_profile is not None:
        override_sources.append("config_profile")
    if legacy_config_profile is not None:
        override_sources.append("config_engine_mode_legacy")
    if profile is not None:
        override_sources.append("api_or_cli_profile")
    if engine_mode is not None:
        override_sources.append("api_or_cli_engine_mode_legacy")
    if window_size_eras is not None or embargo_eras is not None:
        override_sources.append("api_or_cli_engine_parameters_legacy")
    if not override_sources:
        override_sources.append("default")

    resolved_profile = profile or legacy_runtime_profile or config_profile or legacy_config_profile or _DEFAULT_PROFILE
    _raise_if_profile_conflicts(
        explicit_profile=profile,
        explicit_legacy_profile=legacy_runtime_profile,
        config_profile=config_profile,
        config_legacy_profile=legacy_config_profile,
    )

    _raise_if_cv_configured(training_config, mode=resolved_profile)
    _raise_if_legacy_custom_knobs_present(
        engine_config=engine_config,
        window_size_eras=window_size_eras,
        embargo_eras=embargo_eras,
    )

    if resolved_profile == "purged_walk_forward":
        target_horizon = infer_target_horizon(data_config)
        purge_eras = default_embargo_for_horizon(target_horizon)
        cv_config: dict[str, object] = {
            "enabled": True,
            "mode": "official_walkforward",
            "chunk_size": OFFICIAL_WINDOW_SIZE_ERAS,
            "n_splits": 0,
            "embargo": purge_eras,
            "min_train_size": 1,
        }
        return TrainingEnginePlan(
            mode=resolved_profile,
            cv_config=cv_config,
            resolved_config={
                "profile": resolved_profile,
                "target_horizon": target_horizon,
                "window_size_eras": OFFICIAL_WINDOW_SIZE_ERAS,
                "embargo_eras": purge_eras,
            },
            override_sources=override_sources,
        )

    if resolved_profile == "simple":
        cv_config = {
            "enabled": True,
            "mode": "train_validation_holdout",
            "n_splits": 1,
            "embargo": 0,
            "min_train_size": 1,
        }
        return TrainingEnginePlan(
            mode=resolved_profile,
            cv_config=cv_config,
            resolved_config={
                "profile": resolved_profile,
            },
            override_sources=override_sources,
        )

    if resolved_profile == "full_history_refit":
        return TrainingEnginePlan(
            mode=resolved_profile,
            cv_config={
                "enabled": False,
                "mode": "full_history_refit",
                "n_splits": 0,
                "embargo": 0,
                "min_train_size": 0,
            },
            resolved_config={"profile": resolved_profile},
            override_sources=override_sources,
        )

    raise TrainingConfigError(f"training_profile_unknown:{resolved_profile}")


def _raise_if_legacy_method_config(training_config: dict[str, object]) -> None:
    legacy_fields = [field for field in ("method", "method_overrides", "strategy") if field in training_config]
    if legacy_fields:
        joined = ",".join(legacy_fields)
        raise TrainingConfigError(f"training_engine_legacy_config_not_supported:{joined}")


def _raise_if_cv_configured(training_config: dict[str, object], *, mode: TrainingProfile) -> None:
    if training_config.get("cv") is not None:
        raise TrainingConfigError(f"training_engine_disallows_training_cv:{mode}")


def _raise_if_legacy_custom_knobs_present(
    *,
    engine_config: dict[str, object],
    window_size_eras: int | None,
    embargo_eras: int | None,
) -> None:
    has_legacy_config_knobs = any(
        key in engine_config and engine_config.get(key) is not None
        for key in ("window_size_eras", "embargo_eras")
    )
    if has_legacy_config_knobs or window_size_eras is not None or embargo_eras is not None:
        raise TrainingConfigError("training_profile_disallows_custom_parameters")


def _raise_if_profile_conflicts(
    *,
    explicit_profile: TrainingProfile | None,
    explicit_legacy_profile: TrainingProfile | None,
    config_profile: TrainingProfile | None,
    config_legacy_profile: TrainingProfile | None,
) -> None:
    winners = {
        profile
        for profile in (explicit_profile, explicit_legacy_profile, config_profile, config_legacy_profile)
        if profile is not None
    }
    if len(winners) > 1:
        raise TrainingConfigError("training_profile_conflict")
