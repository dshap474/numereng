"""Strict registry-contract tests (config/research_portfolio, spec §2.1)."""

from __future__ import annotations

import pytest

from numereng.config.research_portfolio import (
    REGISTRY_SCHEMA_VERSION,
    RegistryConfig,
    RegistryConfigError,
    load_registry_config,
)


def _policy() -> dict:
    return {"policy_revision": 1, "policy_decision_record_id": "DR-1"}


def _lane(**overrides) -> dict:
    lane = {
        "lane_id": "medium_ender20",
        "axis": "feature_scope",
        "structural": True,
        "research_stage": "seed-confirmed",
        "deployment_stage": "unbound",
        "combination_stage": "not-ready",
        "constitution_revision": 1,
        "experiments": {"scale": "exp-1", "superseded": []},
        "candidates": [{"candidate_id": "c1", "role": "believed_best", "anchor_config": "config_010_s42.json"}],
    }
    lane.update(overrides)
    return lane


def test_minimal_valid_registry_loads() -> None:
    config = load_registry_config({"schema_version": 1, "policy": _policy(), "lanes": [_lane()]})
    assert isinstance(config, RegistryConfig)
    assert config.schema_version == REGISTRY_SCHEMA_VERSION
    # The 8 gated policy params default to None until the human sets them.
    assert config.policy.scout_tranche_cap is None
    assert config.policy.cross_lane_weight_cap is None


def test_wrong_schema_version_rejected() -> None:
    with pytest.raises(RegistryConfigError, match="schema_version"):
        load_registry_config({"schema_version": 2, "policy": _policy()})


def test_unknown_key_rejected() -> None:
    with pytest.raises(RegistryConfigError):
        load_registry_config({"schema_version": 1, "policy": _policy(), "surprise": True})


def test_anchor_config_must_be_bare_json_filename() -> None:
    for bad in ("configs/config_010.json", "config_010.yaml", "  "):
        lane = _lane(candidates=[{"candidate_id": "c1", "role": "believed_best", "anchor_config": bad}])
        with pytest.raises(RegistryConfigError):
            load_registry_config({"schema_version": 1, "policy": _policy(), "lanes": [lane]})


def test_three_stage_fields_and_envelope_wall_hours() -> None:
    lane = _lane(
        envelope={
            "max_rounds": 50,
            "approved_tranche_rounds": 20,
            "max_wall_hours": {"cpu": 100.0, "gpu": 40.0},
            "approved_wall_hours": {"cpu": 50.0, "gpu": 20.0},
        }
    )
    config = load_registry_config({"schema_version": 1, "policy": _policy(), "lanes": [lane]})
    resolved = config.lanes[0]
    assert resolved.research_stage == "seed-confirmed"
    assert resolved.deployment_stage == "unbound"
    assert resolved.combination_stage == "not-ready"
    assert resolved.envelope.max_wall_hours.gpu == 40.0


def test_non_mapping_payload_rejected() -> None:
    with pytest.raises(RegistryConfigError, match="registry_not_object"):
        load_registry_config([])  # type: ignore[arg-type]
