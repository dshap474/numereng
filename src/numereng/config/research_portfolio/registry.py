"""Strict schema for the human-maintained portfolio registry (`registry.json`, v1).

The registry holds ONLY intent and policy. Everything computable — metrics, run
ids, artifact modes, tranche usage, surface ids — is resolved live and never
stored here (see `features/research_portfolio/resolve.py`).

USAGE:
    from numereng.config.research_portfolio import RegistryConfig, load_registry_config
    config = load_registry_config(json.loads(text))   # raises RegistryConfigError on bad input
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

REGISTRY_SCHEMA_VERSION = 1
_JSON_SUFFIX = ".json"


class RegistryConfigError(ValueError):
    """Raised when one registry.json payload fails strict validation."""


# --------------------------------------------------------------------------- #
# Contract models
# --------------------------------------------------------------------------- #


class _StrictModel(BaseModel):
    """Base model that rejects unknown registry keys."""

    model_config = ConfigDict(extra="forbid")


class RegistryPolicy(_StrictModel):
    """Current policy values; META-PROGRAM §6a stays normative for semantics."""

    policy_revision: int
    policy_decision_record_id: str
    scout_tranche_cap: int | None = None
    scout_quality_floor: float | None = None
    coverage_reserve: int | None = None
    diversity_bmc_tolerance: float | None = None
    capacity_class_rule: str | None = None
    live_review_min_resolved_rounds: int | None = None
    combination_trial_cap: int | None = None
    cross_lane_weight_cap: float | None = None


class RegistryWallHours(_StrictModel):
    """Wall-hour budget for one substrate pair (nullable until the human sets it)."""

    cpu: float | None = None
    gpu: float | None = None


class RegistryEnvelope(_StrictModel):
    """Round + wall-hour envelope for one lane."""

    max_rounds: int | None = None
    approved_tranche_rounds: int | None = None
    max_wall_hours: RegistryWallHours = RegistryWallHours()
    approved_wall_hours: RegistryWallHours = RegistryWallHours()


class RegistrySupersededExperiment(_StrictModel):
    """One replaced experiment; its evidence is excluded from lane facts."""

    experiment_id: str
    superseded_by: str
    decision_record_id: str


class RegistryExperiments(_StrictModel):
    """Scout/scale experiment ids plus the supersession ledger."""

    scout: str | None = None
    scale: str | None = None
    superseded: list[RegistrySupersededExperiment] = []


class RegistryCandidate(_StrictModel):
    """One registry candidate anchored by an exact seed-config filename."""

    candidate_id: str
    role: str
    anchor_config: str

    @field_validator("anchor_config")
    @classmethod
    def _validate_anchor_config(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("anchor_config must not be empty")
        if "/" in stripped or "\\" in stripped:
            raise ValueError("anchor_config must be a bare filename, not a path")
        if not stripped.endswith(_JSON_SUFFIX):
            raise ValueError("anchor_config must reference a .json file")
        return stripped


class RegistryLive(_StrictModel):
    """Live-slot binding for one lane."""

    slot: str


class RegistryLane(_StrictModel):
    """One portfolio lane: intent, lifecycle stages, envelope, and candidates."""

    lane_id: str
    axis: str
    structural: bool
    research_stage: str
    deployment_stage: str
    combination_stage: str
    constitution_revision: int
    experiments: RegistryExperiments
    envelope: RegistryEnvelope = RegistryEnvelope()
    candidates: list[RegistryCandidate] = []
    expected_believed_best: str | None = None
    live: RegistryLive | None = None
    decision_records: list[str] = []


class RegistryConfig(_StrictModel):
    """Top-level registry.json contract (schema v1)."""

    schema_version: int
    policy: RegistryPolicy
    lanes: list[RegistryLane] = []

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: int) -> int:
        if value != REGISTRY_SCHEMA_VERSION:
            raise ValueError(f"registry schema_version must be {REGISTRY_SCHEMA_VERSION}")
        return value


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #


def load_registry_config(payload: Mapping[str, object]) -> RegistryConfig:
    """Validate one registry payload into a `RegistryConfig` or raise a domain error."""

    if not isinstance(payload, Mapping):
        raise RegistryConfigError("registry_not_object")
    try:
        return RegistryConfig.model_validate(dict(payload))
    except ValidationError as exc:
        raise RegistryConfigError(f"registry_schema_invalid:{_first_error(exc)}") from exc


def _first_error(exc: ValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "invalid payload"
    first = errors[0]
    location = ".".join(str(part) for part in first.get("loc", ()))
    message = str(first.get("msg", "invalid value"))
    return f"{location}:{message}" if location else message


__all__ = [
    "REGISTRY_SCHEMA_VERSION",
    "RegistryCandidate",
    "RegistryConfig",
    "RegistryConfigError",
    "RegistryEnvelope",
    "RegistryExperiments",
    "RegistryLane",
    "RegistryLive",
    "RegistryPolicy",
    "RegistrySupersededExperiment",
    "RegistryWallHours",
    "load_registry_config",
]
