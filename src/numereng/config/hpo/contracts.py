"""Canonical HPO study config contracts."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_serializer, model_validator

from numereng.config.training import ensure_json_config_path

HpoDirection = Literal["maximize", "minimize"]
HpoSamplerKind = Literal["tpe", "random"]
HpoNeutralizationMode = Literal["era", "global"]
HpoParamKind = Literal["float", "int", "categorical"]

_SAFE_ID = re.compile(r"^[\w\-.]+$")


class _StrictConfigModel(BaseModel):
    """Base model that rejects unknown config keys."""

    model_config = ConfigDict(extra="forbid")


class HpoSearchSpaceSpec(_StrictConfigModel):
    """Typed spec for one search-space parameter path."""

    type: HpoParamKind
    low: float | int | None = None
    high: float | int | None = None
    step: float | int | None = None
    log: bool = False
    choices: list[str | int | float] | None = None

    @model_validator(mode="after")
    def _validate_shape(self) -> HpoSearchSpaceSpec:
        if self.type == "categorical":
            if self.choices is None or not self.choices:
                raise ValueError("choices is required when type is categorical")
            if self.low is not None or self.high is not None or self.step is not None:
                raise ValueError("low/high/step are not allowed when type is categorical")
            return self

        if self.low is None or self.high is None:
            raise ValueError("low and high are required when type is float or int")
        if self.choices is not None:
            raise ValueError("choices is not allowed when type is float or int")

        if self.type == "int":
            if not isinstance(self.low, int) or isinstance(self.low, bool):
                raise ValueError("low must be an integer when type is int")
            if not isinstance(self.high, int) or isinstance(self.high, bool):
                raise ValueError("high must be an integer when type is int")
            if self.step is not None and (not isinstance(self.step, int) or isinstance(self.step, bool)):
                raise ValueError("step must be an integer when type is int")
        else:
            if not isinstance(self.low, (int, float)) or isinstance(self.low, bool):
                raise ValueError("low must be numeric when type is float")
            if not isinstance(self.high, (int, float)) or isinstance(self.high, bool):
                raise ValueError("high must be numeric when type is float")
            if self.step is not None and (not isinstance(self.step, (int, float)) or isinstance(self.step, bool)):
                raise ValueError("step must be numeric when type is float")
        return self


class HpoNeutralizationConfig(_StrictConfigModel):
    """Prediction-stage neutralization controls for HPO trials."""

    enabled: bool = False
    neutralizer_path: str | None = None
    proportion: float = Field(default=0.5, ge=0.0, le=1.0)
    mode: HpoNeutralizationMode = "era"
    neutralizer_cols: list[str] | None = None
    rank_output: bool = True

    @model_validator(mode="after")
    def _validate_shape(self) -> HpoNeutralizationConfig:
        if self.enabled and self.neutralizer_path is None:
            raise ValueError("neutralizer_path is required when neutralization.enabled is true")
        return self


class HpoObjectiveConfig(_StrictConfigModel):
    """Objective block for one HPO study."""

    metric: str = "post_fold_champion_objective"
    direction: HpoDirection = "maximize"
    neutralization: HpoNeutralizationConfig = Field(default_factory=HpoNeutralizationConfig)

    @field_validator("metric")
    @classmethod
    def _validate_metric(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("metric must not be empty")
        return stripped


class HpoSamplerConfig(_StrictConfigModel):
    """Sampler block for one HPO study."""

    kind: HpoSamplerKind = "tpe"
    seed: int | None = 1337
    n_startup_trials: int = Field(default=10, ge=1)
    multivariate: bool = True
    group: bool = False

    @model_validator(mode="after")
    def _validate_shape(self) -> HpoSamplerConfig:
        if self.kind == "random":
            unsupported = {"n_startup_trials", "multivariate", "group"} & self.model_fields_set
            if unsupported:
                fields = ",".join(sorted(unsupported))
                raise ValueError(f"{fields} are not allowed when sampler.kind is random")
            return self
        if self.group and not self.multivariate:
            raise ValueError("group requires multivariate=true when sampler.kind is tpe")
        return self

    @model_serializer(mode="plain")
    def _serialize(self) -> dict[str, Any]:
        return canonicalize_hpo_sampler_payload(
            {
                "kind": self.kind,
                "seed": self.seed,
                "n_startup_trials": self.n_startup_trials,
                "multivariate": self.multivariate,
                "group": self.group,
            }
        )


class HpoPlateauConfig(_StrictConfigModel):
    """Study-level plateau stopping controls."""

    enabled: bool = False
    min_completed_trials: int = Field(default=15, ge=1)
    patience_completed_trials: int = Field(default=10, ge=1)
    min_improvement_abs: float = Field(default=0.00025, ge=0.0)


class HpoStoppingConfig(_StrictConfigModel):
    """Study-level stop controls."""

    max_trials: int = Field(default=100, ge=1)
    max_completed_trials: int | None = Field(default=None, ge=1)
    timeout_seconds: int | None = Field(default=None, ge=1)
    plateau: HpoPlateauConfig = Field(default_factory=HpoPlateauConfig)


class HpoStudyConfig(_StrictConfigModel):
    """Canonical v2 HPO study config contract."""

    study_id: str
    study_name: str
    config_path: str
    experiment_id: str | None = None
    objective: HpoObjectiveConfig
    search_space: dict[str, HpoSearchSpaceSpec]
    sampler: HpoSamplerConfig
    stopping: HpoStoppingConfig

    @field_validator("study_id")
    @classmethod
    def _validate_study_id(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("study_id must not be empty")
        if not _SAFE_ID.match(stripped):
            raise ValueError("study_id must contain only letters, numbers, underscore, dash, or dot")
        return stripped

    @field_validator("study_name")
    @classmethod
    def _validate_study_name(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("study_name must not be empty")
        return stripped

    @field_validator("experiment_id")
    @classmethod
    def _validate_experiment_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("experiment_id must not be empty")
        if not _SAFE_ID.match(stripped):
            raise ValueError("experiment_id must contain only letters, numbers, underscore, dash, or dot")
        return stripped

    @field_validator("config_path")
    @classmethod
    def _validate_config_path(cls, value: str) -> str:
        return ensure_json_config_path(value, field_name="config_path")

    @field_validator("search_space")
    @classmethod
    def _validate_search_space(cls, value: dict[str, HpoSearchSpaceSpec]) -> dict[str, HpoSearchSpaceSpec]:
        if not value:
            raise ValueError("search_space must not be empty")
        return value


def canonicalize_hpo_sampler_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical serialized sampler payload for one HPO study."""

    kind = payload.get("kind", "tpe")
    canonical: dict[str, Any] = {
        "kind": kind,
        "seed": payload.get("seed", 1337),
    }
    if kind != "random":
        canonical["n_startup_trials"] = payload.get("n_startup_trials", 10)
        canonical["multivariate"] = payload.get("multivariate", True)
        canonical["group"] = payload.get("group", False)
    return canonical


def canonicalize_hpo_study_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return one canonical HPO study payload with normalized sampler shape."""

    canonical = {str(key): value for key, value in payload.items()}
    sampler = canonical.get("sampler")
    if isinstance(sampler, Mapping):
        canonical["sampler"] = canonicalize_hpo_sampler_payload(sampler)
    return canonical


__all__ = [
    "canonicalize_hpo_sampler_payload",
    "canonicalize_hpo_study_payload",
    "HpoDirection",
    "HpoNeutralizationConfig",
    "HpoNeutralizationMode",
    "HpoObjectiveConfig",
    "HpoParamKind",
    "HpoPlateauConfig",
    "HpoSamplerConfig",
    "HpoSamplerKind",
    "HpoSearchSpaceSpec",
    "HpoStoppingConfig",
    "HpoStudyConfig",
]
