"""Config-contract tests for the combination-study schemas (freeze + trials, v1)."""

from __future__ import annotations

import pytest

from numereng.config.research_portfolio import (
    STUDY_SCHEMA_VERSION,
    StudyConfigError,
    load_freeze_config,
    load_trials_config,
)


def _freeze_dict(**overrides: object) -> dict:
    payload: dict = {
        "schema_version": STUDY_SCHEMA_VERSION,
        "study_id": "S1",
        "experiment_id": "exp-1",
        "decision_record_id": "DR-1",
        "baseline_candidate_id": "cand_alpha",
        "members": [{"candidate_id": "cand_alpha", "lane_id": "lane_alpha", "anchor_config": "a.json"}],
        "split": {"mode": "chronological_suffix", "holdout_n_eras": 6, "era_gap": 2},
        "meta_validation": {
            "mode": "expanding",
            "min_history_eras": 2,
            "validation_width_eras": 4,
            "step_eras": 4,
            "gap_eras": 1,
        },
        "inference": {"block_length_eras": 3, "n_resamples": 200, "rng_seed": 7},
        "study_trial_cap": 4,
    }
    payload.update(overrides)
    return payload


def test_freeze_loads_valid_payload() -> None:
    freeze = load_freeze_config(_freeze_dict())
    assert freeze.study_id == "S1"
    assert freeze.decision_record_id == "DR-1"
    assert freeze.exploratory is False


def test_freeze_defaults_blank_decision_record() -> None:
    # A missing DR is a *preflight* domain rejection, not a schema error.
    payload = _freeze_dict()
    del payload["decision_record_id"]
    assert load_freeze_config(payload).decision_record_id == ""


def test_freeze_rejects_unknown_key() -> None:
    with pytest.raises(StudyConfigError, match="freeze_schema_invalid"):
        load_freeze_config(_freeze_dict(surprise=True))


def test_freeze_rejects_wrong_schema_version() -> None:
    with pytest.raises(StudyConfigError, match="schema_version"):
        load_freeze_config(_freeze_dict(schema_version=2))


def test_freeze_rejects_bad_split_mode() -> None:
    with pytest.raises(StudyConfigError, match="split.mode"):
        load_freeze_config(_freeze_dict(split={"mode": "random", "holdout_n_eras": 6, "era_gap": 2}))


def test_freeze_rejects_bad_meta_mode() -> None:
    payload = _freeze_dict()
    payload["meta_validation"]["mode"] = "sliding"
    with pytest.raises(StudyConfigError, match="meta_validation.mode"):
        load_freeze_config(payload)


def test_trials_loads_and_rejects_unknown_key() -> None:
    trials = load_trials_config({"study_id": "S1", "trials": [{"trial_id": "t1"}]})
    assert trials.study_id == "S1"
    assert trials.trials[0].neutralization_p == 0.0
    with pytest.raises(StudyConfigError, match="trials_schema_invalid"):
        load_trials_config({"study_id": "S1", "trials": [{"trial_id": "t1", "boom": 1}]})


def test_loaders_reject_non_object() -> None:
    with pytest.raises(StudyConfigError, match="freeze_not_object"):
        load_freeze_config([])  # type: ignore[arg-type]
    with pytest.raises(StudyConfigError, match="trials_not_object"):
        load_trials_config([])  # type: ignore[arg-type]
