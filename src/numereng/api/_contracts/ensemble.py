"""Baseline and ensemble request and response contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from numereng.api._contracts.shared import NeutralizationMode, WorkspaceBoundRequest


class BaselineBuildRequest(WorkspaceBoundRequest):
    run_ids: list[str]
    name: str = Field(min_length=1)
    default_target: str = "target_ender_20"
    description: str | None = None
    promote_active: bool = False


class BaselineBuildResponse(BaseModel):
    name: str
    baseline_dir: str
    predictions_path: str
    metadata_path: str
    available_targets: list[str]
    default_target: str
    source_run_ids: list[str]
    source_experiment_id: str | None = None
    active_predictions_path: str | None = None
    active_metadata_path: str | None = None
    created_at: str


class EnsembleBuildRequest(WorkspaceBoundRequest):
    run_ids: list[str]
    experiment_id: str | None = None
    method: Literal["rank_avg"] = "rank_avg"
    metric: str = "corr_sharpe"
    target: str = "target_ender_20"
    name: str | None = None
    ensemble_id: str | None = None
    weights: list[float] | None = None
    optimize_weights: bool = False
    include_heavy_artifacts: bool = False
    selection_note: str | None = None
    regime_buckets: int = Field(default=4, ge=2, le=50)
    neutralize_members: bool = False
    neutralize_final: bool = False
    neutralizer_path: str | None = None
    neutralization_proportion: float = Field(default=0.5, ge=0.0, le=1.0)
    neutralization_mode: NeutralizationMode = "era"
    neutralizer_cols: list[str] | None = None
    neutralization_rank_output: bool = True

    @model_validator(mode="after")
    def _validate_neutralization(self) -> EnsembleBuildRequest:
        if (self.neutralize_members or self.neutralize_final) and self.neutralizer_path is None:
            raise ValueError("neutralizer_path is required when neutralize_members or neutralize_final is true")
        return self


class EnsembleSelectionSourceRuleRequest(BaseModel):
    experiment_id: str
    selection_mode: Literal["explicit_targets", "top_n"]
    explicit_targets: list[str] = Field(default_factory=list)
    top_n: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _validate_mode_fields(self) -> EnsembleSelectionSourceRuleRequest:
        if self.selection_mode == "explicit_targets":
            if not self.explicit_targets:
                raise ValueError("explicit_targets is required when selection_mode is explicit_targets")
            if self.top_n is not None:
                raise ValueError("top_n must be omitted when selection_mode is explicit_targets")
            return self
        if self.top_n is None:
            raise ValueError("top_n is required when selection_mode is top_n")
        if self.explicit_targets:
            raise ValueError("explicit_targets must be empty when selection_mode is top_n")
        return self


class EnsembleSelectRequest(WorkspaceBoundRequest):
    experiment_id: str
    source_experiment_ids: list[str]
    source_rules: list[EnsembleSelectionSourceRuleRequest]
    selection_id: str | None = None
    target: str = "target_ender_20"
    primary_metric: str = "bmc_last_200_eras.mean"
    tie_break_metric: str = "bmc.mean"
    correlation_threshold: float = Field(default=0.85, gt=0.0, le=1.0)
    top_weighted_variants: int = Field(default=2, ge=1)
    weight_step: float = Field(default=0.05, gt=0.0)
    bundle_policy: Literal["seed_avg"] = "seed_avg"
    required_seed_count: int = Field(default=1, ge=1)
    require_full_seed_bundle: bool = False
    blend_variants: list[
        Literal["all_surviving", "medium_only", "small_only", "top2_medium_top2_small", "top3_overall"]
    ] = Field(
        default_factory=lambda: [
            "all_surviving",
            "medium_only",
            "small_only",
            "top2_medium_top2_small",
            "top3_overall",
        ]
    )
    weighted_promotion_min_gain: float = Field(default=0.0005, ge=0.0)

    @model_validator(mode="after")
    def _validate_sources(self) -> EnsembleSelectRequest:
        if [rule.experiment_id for rule in self.source_rules] != self.source_experiment_ids:
            raise ValueError("source_rules experiment_id order must match source_experiment_ids exactly")
        return self


class EnsembleListRequest(WorkspaceBoundRequest):
    experiment_id: str | None = None
    limit: int = Field(default=50, ge=1)
    offset: int = Field(default=0, ge=0)


class EnsembleGetRequest(WorkspaceBoundRequest):
    ensemble_id: str


class EnsembleComponentResponse(BaseModel):
    run_id: str
    weight: float
    rank: int


class EnsembleMetricResponse(BaseModel):
    name: str
    value: float | None = None


class EnsembleResponse(BaseModel):
    ensemble_id: str
    experiment_id: str | None = None
    name: str
    method: Literal["rank_avg"]
    target: str
    metric: str
    status: str
    components: list[EnsembleComponentResponse] = Field(default_factory=list)
    metrics: list[EnsembleMetricResponse] = Field(default_factory=list)
    artifacts_path: str | None = None
    config: dict[str, object] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class EnsembleSelectionWinnerResponse(BaseModel):
    blend_id: str
    selection_mode: Literal["equal_weight", "weighted", "equal_weight_retained"]
    component_ids: list[str] = Field(default_factory=list)
    weights: list[float] = Field(default_factory=list)
    metrics: dict[str, float | None] = Field(default_factory=dict)


class EnsembleSelectResponse(BaseModel):
    selection_id: str
    experiment_id: str
    target: str
    primary_metric: str
    tie_break_metric: str
    status: str
    artifacts_path: str
    frozen_candidate_count: int
    surviving_candidate_count: int
    equal_weight_variant_count: int
    weighted_candidate_count: int
    winner: EnsembleSelectionWinnerResponse
    created_at: str
    updated_at: str


class EnsembleListResponse(BaseModel):
    ensembles: list[EnsembleResponse]


__all__ = [
    "BaselineBuildRequest",
    "BaselineBuildResponse",
    "EnsembleBuildRequest",
    "EnsembleComponentResponse",
    "EnsembleGetRequest",
    "EnsembleListRequest",
    "EnsembleListResponse",
    "EnsembleMetricResponse",
    "EnsembleResponse",
    "EnsembleSelectionSourceRuleRequest",
    "EnsembleSelectionWinnerResponse",
    "EnsembleSelectRequest",
    "EnsembleSelectResponse",
]
