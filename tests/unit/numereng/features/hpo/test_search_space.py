from __future__ import annotations

from typing import Any

import pytest

import numereng.features.hpo.search_space as search_space_module


def test_resolve_search_space_from_explicit_specs() -> None:
    specs = search_space_module.resolve_search_space(
        raw_search_space={
            "model.params.learning_rate": {
                "type": "float",
                "low": 0.001,
                "high": 0.1,
                "log": True,
            },
            "model.params.num_leaves": {
                "type": "int",
                "low": 16,
                "high": 128,
                "step": 8,
            },
            "model.params.feature_fraction": {
                "type": "categorical",
                "choices": [0.5, 0.75, 1.0],
            },
        }
    )

    by_path = {item.path: item for item in specs}
    assert by_path["model.params.learning_rate"].kind == "float"
    assert by_path["model.params.learning_rate"].log is True
    assert by_path["model.params.num_leaves"].kind == "int"
    assert by_path["model.params.num_leaves"].step == 8
    assert by_path["model.params.feature_fraction"].kind == "categorical"
    assert by_path["model.params.feature_fraction"].choices == (0.5, 0.75, 1.0)


def test_resolve_search_space_requires_explicit_mapping() -> None:
    with pytest.raises(search_space_module.HpoSearchSpaceError, match="hpo_search_space_required"):
        search_space_module.resolve_search_space(raw_search_space=None)


def test_resolve_search_space_rejects_invalid_spec_type() -> None:
    with pytest.raises(search_space_module.HpoSearchSpaceError, match="hpo_search_space_type_invalid"):
        search_space_module.resolve_search_space(
            raw_search_space={"model.params.learning_rate": {"type": "unsupported"}},
        )


def test_apply_param_overrides_updates_copy_and_rejects_conflicting_path() -> None:
    base_config: dict[str, Any] = {
        "model": {
            "params": {
                "learning_rate": 0.01,
                "num_leaves": 64,
            }
        },
        "training": {"engine": {"window_size_eras": 156}},
    }

    updated = search_space_module.apply_param_overrides(
        base_config,
        params={
            "model.params.learning_rate": 0.02,
            "training.engine.embargo_eras": 8,
        },
    )
    assert updated["model"]["params"]["learning_rate"] == 0.02
    assert updated["training"]["engine"]["embargo_eras"] == 8
    assert base_config["model"]["params"]["learning_rate"] == 0.01

    with pytest.raises(search_space_module.HpoSearchSpaceError, match="hpo_search_space_path_conflict"):
        search_space_module.apply_param_overrides(
            base_config,
            params={"model.params.learning_rate.inner": 1},
        )
