"""Strict schemas for the bounded combination study (freeze + trials, v1).

Two immutable configs drive the two-phase study (spec §4). `freeze.json` names
the members, split, meta-validation folds, optional neutralization, and the
study trial cap; it is validated then materialized into a frozen manifest.
`trials.json` is accepted only against a frozen, unsealed study and lists the
candidate-selecting trials to score.

`decision_record_id` defaults to empty so a missing DR is a freeze *preflight*
rejection (a domain error the human can read), never an opaque schema error.

USAGE:
    from numereng.config.research_portfolio import load_freeze_config, load_trials_config
    freeze = load_freeze_config(json.loads(text))   # raises StudyConfigError on bad input
    trials = load_trials_config(json.loads(text))
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

STUDY_SCHEMA_VERSION = 1
_SPLIT_MODE = "chronological_suffix"
_META_VALIDATION_MODES = ("expanding", "rolling")


class StudyConfigError(ValueError):
    """Raised when one study config payload fails strict validation."""


# --------------------------------------------------------------------------- #
# Contract models
# --------------------------------------------------------------------------- #


class _StrictModel(BaseModel):
    """Base model that rejects unknown study-config keys."""

    model_config = ConfigDict(extra="forbid")


class FreezeMember(_StrictModel):
    """One study member: a candidate anchored to its resolved seed runs."""

    candidate_id: str
    lane_id: str
    anchor_config: str
    run_ids: list[str] = []
    prediction_sha256: list[str] = []


class FreezeSplit(_StrictModel):
    """Chronological-suffix holdout carve-out with a purge gap."""

    mode: str = _SPLIT_MODE
    holdout_n_eras: int
    era_gap: int

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, value: str) -> str:
        if value != _SPLIT_MODE:
            raise ValueError(f"split.mode must be {_SPLIT_MODE}")
        return value


class FreezeMetaValidation(_StrictModel):
    """Expanding/rolling evaluation folds over the search region."""

    mode: str
    min_history_eras: int
    validation_width_eras: int
    step_eras: int
    gap_eras: int

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, value: str) -> str:
        if value not in _META_VALIDATION_MODES:
            raise ValueError(f"meta_validation.mode must be one of {list(_META_VALIDATION_MODES)}")
        return value


class FreezeNeutralization(_StrictModel):
    """Study-level neutralization block (nullable at the top level)."""

    source_path: str
    content_sha256: str
    columns: list[str] = []
    mode: str = "era"
    rank_output: bool = True
    justification: str = ""


class FreezeInference(_StrictModel):
    """Paired block-bootstrap inference parameters recorded into the manifest."""

    block_length_eras: int
    n_resamples: int
    rng_seed: int


class FreezeConfig(_StrictModel):
    """Top-level freeze.json contract (schema v1)."""

    schema_version: int
    study_id: str
    experiment_id: str
    decision_record_id: str = ""
    members: list[FreezeMember] = []
    baseline_candidate_id: str
    split: FreezeSplit
    meta_validation: FreezeMetaValidation
    neutralization: FreezeNeutralization | None = None
    inference: FreezeInference
    study_trial_cap: int
    exploratory: bool = False

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: int) -> int:
        if value != STUDY_SCHEMA_VERSION:
            raise ValueError(f"study schema_version must be {STUDY_SCHEMA_VERSION}")
        return value


class Trial(_StrictModel):
    """One trial: candidate membership per lane, lane weights, neutralization strength."""

    trial_id: str
    selection: dict[str, list[str]] = {}
    lane_weights: dict[str, float] = {}
    neutralization_p: float = 0.0


class TrialsConfig(_StrictModel):
    """Top-level trials.json contract (accepted only against a frozen study)."""

    study_id: str
    trials: list[Trial] = []


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #


def load_freeze_config(payload: Mapping[str, object]) -> FreezeConfig:
    """Validate one freeze payload into a `FreezeConfig` or raise a domain error."""

    if not isinstance(payload, Mapping):
        raise StudyConfigError("freeze_not_object")
    try:
        return FreezeConfig.model_validate(dict(payload))
    except ValidationError as exc:
        raise StudyConfigError(f"freeze_schema_invalid:{_first_error(exc)}") from exc


def load_trials_config(payload: Mapping[str, object]) -> TrialsConfig:
    """Validate one trials payload into a `TrialsConfig` or raise a domain error."""

    if not isinstance(payload, Mapping):
        raise StudyConfigError("trials_not_object")
    try:
        return TrialsConfig.model_validate(dict(payload))
    except ValidationError as exc:
        raise StudyConfigError(f"trials_schema_invalid:{_first_error(exc)}") from exc


def _first_error(exc: ValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "invalid payload"
    first = errors[0]
    location = ".".join(str(part) for part in first.get("loc", ()))
    message = str(first.get("msg", "invalid value"))
    return f"{location}:{message}" if location else message


__all__ = [
    "STUDY_SCHEMA_VERSION",
    "FreezeConfig",
    "FreezeInference",
    "FreezeMember",
    "FreezeMetaValidation",
    "FreezeNeutralization",
    "FreezeSplit",
    "StudyConfigError",
    "Trial",
    "TrialsConfig",
    "load_freeze_config",
    "load_trials_config",
]
