"""Schema constants, errors, and result dataclasses for the research portfolio.

Everything here is a plain frozen dataclass so the api layer can serialize it
into pydantic responses without importing feature internals.

USAGE:
    from numereng.features.research_portfolio import types as pf_types
    trio = pf_types.REQUIRED_TRIO_SEEDS
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Strict trio required for confirmed candidate status (§2.2.2). Order is the
# canonical scout->confirm seed order; membership, not order, is what is checked.
REQUIRED_TRIO_SEEDS: tuple[int, ...] = (42, 17, 99)

# Evidence tiers, weakest first; the lane tier is the weakest across its candidates.
EVIDENCE_TIERS: tuple[str, ...] = ("discovery", "seed-confirmed", "scale-confirmed")


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class PortfolioError(Exception):
    """Base error for the research-portfolio feature."""


class PortfolioValidationError(PortfolioError):
    """Raised on malformed registry/journal input that must hard-fail."""


# --------------------------------------------------------------------------- #
# Result dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SeedFact:
    """Resolved facts for one seed run of a candidate recipe."""

    seed: int | None
    run_id: str | None
    bmc: float | None
    fnc: float | None
    artifact_mode: str
    training_profile: str | None
    experiment_id: str | None
    config: str | None
    config_hash_ok: bool | None
    surface_id: str | None
    surface_unavailable_reason: str | None
    journal_vs_disk_bmc_delta: float | None
    duplicate_run_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateFact:
    """Resolved facts for one registry candidate."""

    candidate_id: str
    role: str
    anchor_config: str
    recipe_key: str | None
    evidence_tier: str
    seeds_present: tuple[int, ...]
    trio_complete: bool
    trio_bmc_mean: float | None
    bmc_sd: float | None
    per_seed: tuple[SeedFact, ...]
    surface_ids: tuple[str, ...]
    surface_match: bool
    blockers: tuple[str, ...] = ()


@dataclass(frozen=True)
class LaneStatus:
    """Resolved status for one portfolio lane."""

    lane_id: str
    axis: str
    structural: bool
    research_stage_asserted: str
    research_stage_evidenced: str
    deployment_stage_asserted: str
    deployment_stage_evidenced: str
    combination_stage_asserted: str
    combination_stage_evidenced: str
    tranche_rounds_completed: int | None
    tranche_approved_rounds: int | None
    tranche_max_rounds: int | None
    observed_seed_noise: float | None
    candidates: tuple[CandidateFact, ...]
    surface_match: bool
    drift: str | None
    blockers: tuple[str, ...]
    latest_diversity_report_id: str | None = None


@dataclass(frozen=True)
class PortfolioReport:
    """Full portfolio status report (disk-first, no SQLite)."""

    schema_version: int
    portfolio_present: bool
    generated_at: str
    policy_hash: str | None
    policy_gaps: tuple[str, ...]
    lanes: tuple[LaneStatus, ...]
    blockers: tuple[str, ...]
    registry_path: str
    report_path: str | None = None


# --------------------------------------------------------------------------- #
# Diversity report dataclasses (P2)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DiversityMember:
    """One candidate included in a diversity report."""

    candidate_id: str
    lane_id: str
    recipe_key: str | None
    run_ids: tuple[str, ...]
    prediction_sha256: tuple[str, ...]
    trio_bmc200: float | None


@dataclass(frozen=True)
class PairwiseDiagnostic:
    """One pair's Spearman distribution, BMC-series correlation, and joint drawdown."""

    left: str
    right: str
    spearman_mean: float | None
    spearman_p10: float | None
    spearman_p90: float | None
    spearman_min: float | None
    bmc_series_corr: float | None
    joint_drawdown_fraction: float | None


@dataclass(frozen=True)
class LaneLeaveOneOut:
    """Leave-one-lane-out marginal-value diagnostic for one lane."""

    lane_id: str
    blend_bmc_mean: float | None
    loo_bmc_mean: float | None
    mean_diff: float | None
    ci90_low: float | None
    ci90_high: float | None
    prob_positive: float | None


@dataclass(frozen=True)
class DiversityInference:
    """Recorded bootstrap inference parameters (never hidden constants)."""

    block_length_eras: int
    n_resamples: int
    rng_seed: int


@dataclass(frozen=True)
class DiversityReport:
    """Full cross-lane diversity report (P2)."""

    schema_version: int
    report_id: str
    generated_at: str
    report_dir: str
    surface_id: str | None
    policy_hash: str | None
    diversity_bmc_tolerance: float | None
    inference: DiversityInference
    n_eras: int
    members: tuple[DiversityMember, ...]
    included_lanes: tuple[str, ...]
    excluded_candidates: tuple[tuple[str, str], ...]
    blend_bmc_mean: float | None
    pairwise: tuple[PairwiseDiagnostic, ...]
    leave_one_out: tuple[LaneLeaveOneOut, ...]


DIVERSITY_SCHEMA_VERSION = 1


# --------------------------------------------------------------------------- #
# Combination study dataclasses (P3)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StudyFreezeResult:
    """Outcome of `study freeze`: a materialized, unsealed study snapshot."""

    study_id: str
    study_dir: str
    experiment_id: str
    frozen: bool
    n_members: int
    n_lanes: int
    n_search_folds: int
    holdout_n_eras: int
    surface_id: str | None
    holdout_fingerprint: str
    exploratory: bool


@dataclass(frozen=True)
class StudyTrialResult:
    """Search-region result for one executed trial (vs the fixed baseline)."""

    trial_id: str
    pooled_search_bmc: float | None
    baseline_pooled_search_bmc: float | None
    diff_mean: float | None
    diff_ci90_low: float | None
    diff_ci90_high: float | None
    diff_prob_positive: float | None
    n_folds: int
    status: str


@dataclass(frozen=True)
class StudyRunResult:
    """Outcome of `study run`: per-trial search results plus resume bookkeeping."""

    study_id: str
    study_dir: str
    executed: int
    skipped: int
    superseded: int
    trial_cap: int
    ledger_path: str
    trials: tuple[StudyTrialResult, ...]


@dataclass(frozen=True)
class StudyFinalizeResult:
    """Outcome of `study finalize`: holdout scoring of the selected trial + seal."""

    study_id: str
    study_dir: str
    selected_trial: str
    is_baseline: bool
    holdout_bmc: float | None
    baseline_holdout_bmc: float | None
    holdout_diff: float | None
    degradation: float | None
    holdout_ci90_low: float | None
    holdout_ci90_high: float | None
    holdout_prob_positive: float | None
    sealed: bool
    artifacts_dir: str


@dataclass(frozen=True)
class StudyStatusResult:
    """Read-only lifecycle snapshot for one study."""

    study_id: str
    study_dir: str
    frozen: bool
    sealed: bool
    trials_executed: int
    trial_cap: int
    selected_trial: str | None


STUDY_SCHEMA_VERSION = 1


__all__ = [
    "DIVERSITY_SCHEMA_VERSION",
    "EVIDENCE_TIERS",
    "REQUIRED_TRIO_SEEDS",
    "STUDY_SCHEMA_VERSION",
    "CandidateFact",
    "DiversityInference",
    "DiversityMember",
    "DiversityReport",
    "LaneLeaveOneOut",
    "LaneStatus",
    "PairwiseDiagnostic",
    "PortfolioError",
    "PortfolioReport",
    "PortfolioValidationError",
    "SeedFact",
    "StudyFinalizeResult",
    "StudyFreezeResult",
    "StudyRunResult",
    "StudyStatusResult",
    "StudyTrialResult",
]
