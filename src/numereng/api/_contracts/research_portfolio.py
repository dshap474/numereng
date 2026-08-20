"""Research-portfolio request and response contracts (P1 status/report, P2 diversity)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from numereng.api._contracts.shared import WorkspaceBoundRequest


class PortfolioStatusRequest(WorkspaceBoundRequest):
    write: bool = False


class PortfolioSeedResponse(BaseModel):
    seed: int | None = None
    run_id: str | None = None
    bmc: float | None = None
    fnc: float | None = None
    artifact_mode: str
    training_profile: str | None = None
    experiment_id: str | None = None
    config: str | None = None
    config_hash_ok: bool | None = None
    surface_id: str | None = None
    surface_unavailable_reason: str | None = None
    journal_vs_disk_bmc_delta: float | None = None
    duplicate_run_ids: list[str] = Field(default_factory=list)


class PortfolioCandidateResponse(BaseModel):
    candidate_id: str
    role: str
    anchor_config: str
    recipe_key: str | None = None
    evidence_tier: str
    seeds_present: list[int] = Field(default_factory=list)
    trio_complete: bool
    trio_bmc_mean: float | None = None
    bmc_sd: float | None = None
    per_seed: list[PortfolioSeedResponse] = Field(default_factory=list)
    surface_ids: list[str] = Field(default_factory=list)
    surface_match: bool
    blockers: list[str] = Field(default_factory=list)


class PortfolioLaneResponse(BaseModel):
    lane_id: str
    axis: str
    structural: bool
    research_stage_asserted: str
    research_stage_evidenced: str
    deployment_stage_asserted: str
    deployment_stage_evidenced: str
    combination_stage_asserted: str
    combination_stage_evidenced: str
    tranche_rounds_completed: int | None = None
    tranche_approved_rounds: int | None = None
    tranche_max_rounds: int | None = None
    observed_seed_noise: float | None = None
    candidates: list[PortfolioCandidateResponse] = Field(default_factory=list)
    surface_match: bool
    drift: str | None = None
    blockers: list[str] = Field(default_factory=list)
    latest_diversity_report_id: str | None = None


class PortfolioStatusResponse(BaseModel):
    schema_version: int
    portfolio_present: bool
    generated_at: str
    policy_hash: str | None = None
    policy_gaps: list[str] = Field(default_factory=list)
    lanes: list[PortfolioLaneResponse] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    registry_path: str
    report_path: str | None = None


# --------------------------------------------------------------------------- #
# Diversity report contracts (P2)
# --------------------------------------------------------------------------- #


class PortfolioDiversityRequest(WorkspaceBoundRequest):
    lanes: list[str] | None = None
    block_length_eras: int = 10
    n_resamples: int = 2000
    rng_seed: int = 7


class DiversityMemberResponse(BaseModel):
    candidate_id: str
    lane_id: str
    recipe_key: str | None = None
    run_ids: list[str] = Field(default_factory=list)
    prediction_sha256: list[str] = Field(default_factory=list)
    trio_bmc200: float | None = None


class DiversityPairwiseResponse(BaseModel):
    left: str
    right: str
    spearman_mean: float | None = None
    spearman_p10: float | None = None
    spearman_p90: float | None = None
    spearman_min: float | None = None
    bmc_series_corr: float | None = None
    joint_drawdown_fraction: float | None = None


class DiversityLeaveOneOutResponse(BaseModel):
    lane_id: str
    blend_bmc_mean: float | None = None
    loo_bmc_mean: float | None = None
    mean_diff: float | None = None
    ci90_low: float | None = None
    ci90_high: float | None = None
    prob_positive: float | None = None


class DiversityInferenceResponse(BaseModel):
    block_length_eras: int
    n_resamples: int
    rng_seed: int


class PortfolioDiversityResponse(BaseModel):
    schema_version: int
    report_id: str
    generated_at: str
    report_dir: str
    surface_id: str | None = None
    policy_hash: str | None = None
    diversity_bmc_tolerance: float | None = None
    inference: DiversityInferenceResponse
    n_eras: int
    members: list[DiversityMemberResponse] = Field(default_factory=list)
    included_lanes: list[str] = Field(default_factory=list)
    excluded_candidates: list[list[str]] = Field(default_factory=list)
    blend_bmc_mean: float | None = None
    pairwise: list[DiversityPairwiseResponse] = Field(default_factory=list)
    leave_one_out: list[DiversityLeaveOneOutResponse] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Combination study contracts (P3)
# --------------------------------------------------------------------------- #


class StudyFreezeRequest(WorkspaceBoundRequest):
    config_path: str


class StudyRunRequest(WorkspaceBoundRequest):
    trials_path: str
    experiment_id: str | None = None


class StudyFinalizeRequest(WorkspaceBoundRequest):
    study_id: str
    select: str
    experiment_id: str | None = None


class StudyStatusRequest(WorkspaceBoundRequest):
    study_id: str
    experiment_id: str | None = None


class StudyFreezeResponse(BaseModel):
    study_id: str
    study_dir: str
    experiment_id: str
    frozen: bool
    n_members: int
    n_lanes: int
    n_search_folds: int
    holdout_n_eras: int
    surface_id: str | None = None
    holdout_fingerprint: str
    exploratory: bool


class StudyTrialResponse(BaseModel):
    trial_id: str
    pooled_search_bmc: float | None = None
    baseline_pooled_search_bmc: float | None = None
    diff_mean: float | None = None
    diff_ci90_low: float | None = None
    diff_ci90_high: float | None = None
    diff_prob_positive: float | None = None
    n_folds: int
    status: str


class StudyRunResponse(BaseModel):
    study_id: str
    study_dir: str
    executed: int
    skipped: int
    superseded: int
    trial_cap: int
    ledger_path: str
    trials: list[StudyTrialResponse] = Field(default_factory=list)


class StudyFinalizeResponse(BaseModel):
    study_id: str
    study_dir: str
    selected_trial: str
    is_baseline: bool
    holdout_bmc: float | None = None
    baseline_holdout_bmc: float | None = None
    holdout_diff: float | None = None
    degradation: float | None = None
    holdout_ci90_low: float | None = None
    holdout_ci90_high: float | None = None
    holdout_prob_positive: float | None = None
    sealed: bool
    artifacts_dir: str


class StudyStatusResponse(BaseModel):
    study_id: str
    study_dir: str
    frozen: bool
    sealed: bool
    trials_executed: int
    trial_cap: int
    selected_trial: str | None = None


__all__ = [
    "DiversityInferenceResponse",
    "DiversityLeaveOneOutResponse",
    "DiversityMemberResponse",
    "DiversityPairwiseResponse",
    "PortfolioCandidateResponse",
    "PortfolioDiversityRequest",
    "PortfolioDiversityResponse",
    "PortfolioLaneResponse",
    "PortfolioSeedResponse",
    "PortfolioStatusRequest",
    "PortfolioStatusResponse",
    "StudyFinalizeRequest",
    "StudyFinalizeResponse",
    "StudyFreezeRequest",
    "StudyFreezeResponse",
    "StudyRunRequest",
    "StudyRunResponse",
    "StudyStatusRequest",
    "StudyStatusResponse",
    "StudyTrialResponse",
]
