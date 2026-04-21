"""HPO request and response contracts."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_serializer, model_validator

from numereng.api._contracts.shared import NeutralizationMode, WorkspaceBoundRequest
from numereng.config.hpo.contracts import canonicalize_hpo_sampler_payload

_SAFE_ID = re.compile(r"^[\w\-.]+$")


class HpoSearchSpaceSpecRequest(BaseModel):
    type: Literal["float", "int", "categorical"]
    low: float | int | None = None
    high: float | int | None = None
    step: float | int | None = None
    log: bool = False
    choices: list[str | int | float] | None = None

    @model_validator(mode="after")
    def _validate_shape(self) -> HpoSearchSpaceSpecRequest:
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
        return self


class HpoNeutralizationRequest(BaseModel):
    enabled: bool = False
    neutralizer_path: str | None = None
    proportion: float = Field(default=0.5, ge=0.0, le=1.0)
    mode: NeutralizationMode = "era"
    neutralizer_cols: list[str] | None = None
    rank_output: bool = True

    @model_validator(mode="after")
    def _validate_shape(self) -> HpoNeutralizationRequest:
        if self.enabled and self.neutralizer_path is None:
            raise ValueError("neutralizer_path is required when neutralization.enabled is true")
        return self


class HpoObjectiveRequest(BaseModel):
    metric: str = "bmc_last_200_eras.mean"
    direction: Literal["maximize", "minimize"] = "maximize"
    neutralization: HpoNeutralizationRequest = Field(default_factory=HpoNeutralizationRequest)

    @field_validator("metric")
    @classmethod
    def _validate_metric(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("metric must not be empty")
        return stripped


class HpoSamplerRequest(BaseModel):
    kind: Literal["tpe", "random"] = "tpe"
    seed: int | None = 1337
    n_startup_trials: int = Field(default=10, ge=1)
    multivariate: bool = True
    group: bool = False

    @model_validator(mode="after")
    def _validate_shape(self) -> HpoSamplerRequest:
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
    def _serialize(self) -> dict[str, object]:
        return canonicalize_hpo_sampler_payload(
            {
                "kind": self.kind,
                "seed": self.seed,
                "n_startup_trials": self.n_startup_trials,
                "multivariate": self.multivariate,
                "group": self.group,
            }
        )


class HpoPlateauRequest(BaseModel):
    enabled: bool = False
    min_completed_trials: int = Field(default=15, ge=1)
    patience_completed_trials: int = Field(default=10, ge=1)
    min_improvement_abs: float = Field(default=0.00025, ge=0.0)


class HpoStoppingRequest(BaseModel):
    max_trials: int = Field(default=100, ge=1)
    max_completed_trials: int | None = Field(default=None, ge=1)
    timeout_seconds: int | None = Field(default=None, ge=1)
    plateau: HpoPlateauRequest = Field(default_factory=HpoPlateauRequest)


class HpoStudySpecResponse(BaseModel):
    study_id: str
    study_name: str
    config_path: str
    experiment_id: str | None = None
    objective: HpoObjectiveRequest
    search_space: dict[str, HpoSearchSpaceSpecRequest]
    sampler: HpoSamplerRequest
    stopping: HpoStoppingRequest


class HpoStudyCreateRequest(WorkspaceBoundRequest):
    study_id: str
    study_name: str
    config_path: str
    experiment_id: str | None = None
    objective: HpoObjectiveRequest
    search_space: dict[str, HpoSearchSpaceSpecRequest]
    sampler: HpoSamplerRequest
    stopping: HpoStoppingRequest

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

    @field_validator("search_space")
    @classmethod
    def _validate_search_space(
        cls,
        value: dict[str, HpoSearchSpaceSpecRequest],
    ) -> dict[str, HpoSearchSpaceSpecRequest]:
        if not value:
            raise ValueError("search_space must not be empty")
        return value


class HpoStudyListRequest(WorkspaceBoundRequest):
    experiment_id: str | None = None
    status: str | None = None
    limit: int = Field(default=50, ge=1)
    offset: int = Field(default=0, ge=0)


class HpoStudyGetRequest(WorkspaceBoundRequest):
    study_id: str


class HpoStudyTrialsRequest(WorkspaceBoundRequest):
    study_id: str


class HpoTrialResponse(BaseModel):
    study_id: str
    trial_number: int
    status: str
    value: float | None = None
    run_id: str | None = None
    config_path: str | None = None
    params: dict[str, object] = Field(default_factory=dict)
    error_message: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    updated_at: str


class HpoStudyResponse(BaseModel):
    study_id: str
    experiment_id: str | None = None
    study_name: str
    status: str
    best_trial_number: int | None = None
    best_value: float | None = None
    best_run_id: str | None = None
    spec: HpoStudySpecResponse
    attempted_trials: int
    completed_trials: int
    failed_trials: int
    stop_reason: str | None = None
    storage_path: str | None = None
    error_message: str | None = None
    created_at: str
    updated_at: str


class HpoStudyListResponse(BaseModel):
    studies: list[HpoStudyResponse]


class HpoStudyTrialsResponse(BaseModel):
    study_id: str
    trials: list[HpoTrialResponse]


__all__ = [
    "HpoNeutralizationRequest",
    "HpoObjectiveRequest",
    "HpoPlateauRequest",
    "HpoSamplerRequest",
    "HpoSearchSpaceSpecRequest",
    "HpoStoppingRequest",
    "HpoStudyCreateRequest",
    "HpoStudyGetRequest",
    "HpoStudyListRequest",
    "HpoStudyListResponse",
    "HpoStudyResponse",
    "HpoStudySpecResponse",
    "HpoStudyTrialsRequest",
    "HpoStudyTrialsResponse",
    "HpoTrialResponse",
]
