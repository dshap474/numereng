"""Canonical HPO study config contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from numereng.config.training import ensure_json_config_path

HpoDirection = Literal["maximize", "minimize"]
HpoSampler = Literal["tpe", "random"]
HpoNeutralizationMode = Literal["era", "global"]
HpoParamKind = Literal["float", "int", "categorical"]


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


class HpoStudyConfig(_StrictConfigModel):
    """Canonical HPO study config contract."""

    study_name: str
    config_path: str
    experiment_id: str | None = None
    metric: str = "bmc_last_200_eras.mean"
    direction: HpoDirection = "maximize"
    n_trials: int = Field(default=100, ge=1)
    sampler: HpoSampler = "tpe"
    seed: int | None = 1337
    search_space: dict[str, HpoSearchSpaceSpec] | None = None
    neutralization: HpoNeutralizationConfig = Field(default_factory=HpoNeutralizationConfig)

    @field_validator("study_name")
    @classmethod
    def _validate_study_name(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("study_name must not be empty")
        return stripped

    @field_validator("config_path")
    @classmethod
    def _validate_config_path(cls, value: str) -> str:
        return ensure_json_config_path(value, field_name="config_path")


__all__ = [
    "HpoDirection",
    "HpoNeutralizationConfig",
    "HpoNeutralizationMode",
    "HpoParamKind",
    "HpoSampler",
    "HpoSearchSpaceSpec",
    "HpoStudyConfig",
]
