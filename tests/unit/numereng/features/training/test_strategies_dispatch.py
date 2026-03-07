from __future__ import annotations

import pytest

from numereng.features.training.errors import TrainingConfigError
from numereng.features.training.strategies.core.dispatch import resolve_training_engine


def test_resolve_training_engine_defaults_to_purged_walk_forward() -> None:
    plan = resolve_training_engine(training_config={}, data_config={})
    assert plan.mode == "purged_walk_forward"
    assert plan.cv_config["mode"] == "official_walkforward"
    assert plan.cv_config["chunk_size"] == 156
    assert plan.cv_config["embargo"] == 8
    assert plan.override_sources == ["default"]


def test_resolve_training_engine_purged_walk_forward_horizon_60_sets_embargo_16() -> None:
    plan = resolve_training_engine(
        training_config={},
        data_config={"target_col": "target_jerome_60"},
    )
    assert plan.mode == "purged_walk_forward"
    assert plan.cv_config["embargo"] == 16


def test_resolve_training_engine_simple_profile() -> None:
    plan = resolve_training_engine(
        training_config={"engine": {"profile": "simple"}},
        data_config={},
    )
    assert plan.mode == "simple"
    assert plan.cv_config["enabled"] is True
    assert plan.cv_config["mode"] == "train_validation_holdout"
    assert plan.cv_config["embargo"] == 0


def test_resolve_training_engine_rejects_submission_profile_rename() -> None:
    with pytest.raises(TrainingConfigError, match="training_profile_renamed:submission->full_history_refit"):
        resolve_training_engine(
            training_config={"engine": {"profile": "submission"}},
            data_config={},
        )


def test_resolve_training_engine_legacy_official_mode_maps_to_purged_walk_forward() -> None:
    plan = resolve_training_engine(
        training_config={"engine": {"mode": "official"}},
        data_config={},
    )
    assert plan.mode == "purged_walk_forward"
    assert "config_engine_mode_legacy" in plan.override_sources


def test_resolve_training_engine_legacy_full_history_mode_maps_to_full_history_refit() -> None:
    plan = resolve_training_engine(
        training_config={"engine": {"mode": "full_history"}},
        data_config={},
    )
    assert plan.mode == "full_history_refit"
    assert "config_engine_mode_legacy" in plan.override_sources


def test_resolve_training_engine_rejects_legacy_custom_mode() -> None:
    with pytest.raises(TrainingConfigError, match="training_profile_legacy_custom_not_supported"):
        resolve_training_engine(
            training_config={"engine": {"mode": "custom"}},
            data_config={},
        )


def test_resolve_training_engine_rejects_legacy_custom_knobs() -> None:
    with pytest.raises(TrainingConfigError, match="training_profile_disallows_custom_parameters"):
        resolve_training_engine(
            training_config={"engine": {"profile": "purged_walk_forward", "window_size_eras": 100}},
            data_config={},
        )


def test_resolve_training_engine_rejects_profile_conflicts() -> None:
    with pytest.raises(TrainingConfigError, match="training_profile_conflict"):
        resolve_training_engine(
            training_config={"engine": {"profile": "simple"}},
            data_config={},
            profile="full_history_refit",
        )


def test_resolve_training_engine_rejects_legacy_method_config() -> None:
    with pytest.raises(TrainingConfigError, match="training_engine_legacy_config_not_supported"):
        resolve_training_engine(
            training_config={"method": "official_walkforward"},
            data_config={},
        )
