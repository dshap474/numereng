"""Research-portfolio API handlers (PackageError translation)."""

from __future__ import annotations

from numereng.api.contracts import (
    DiversityInferenceResponse,
    DiversityLeaveOneOutResponse,
    DiversityMemberResponse,
    DiversityPairwiseResponse,
    PortfolioCandidateResponse,
    PortfolioDiversityRequest,
    PortfolioDiversityResponse,
    PortfolioLaneResponse,
    PortfolioSeedResponse,
    PortfolioStatusRequest,
    PortfolioStatusResponse,
    StudyFinalizeRequest,
    StudyFinalizeResponse,
    StudyFreezeRequest,
    StudyFreezeResponse,
    StudyRunRequest,
    StudyRunResponse,
    StudyStatusRequest,
    StudyStatusResponse,
    StudyTrialResponse,
)
from numereng.config.research_portfolio import StudyConfigError
from numereng.features.experiments import ExperimentError
from numereng.features.research_portfolio import (
    PortfolioError,
)
from numereng.features.research_portfolio import (
    portfolio_diversity as _portfolio_diversity,
)
from numereng.features.research_portfolio import (
    portfolio_status as _portfolio_status,
)
from numereng.features.research_portfolio import (
    study_finalize as _study_finalize,
)
from numereng.features.research_portfolio import (
    study_freeze as _study_freeze,
)
from numereng.features.research_portfolio import (
    study_run as _study_run,
)
from numereng.features.research_portfolio import (
    study_status as _study_status,
)
from numereng.features.research_portfolio.types import (
    CandidateFact,
    DiversityMember,
    DiversityReport,
    LaneLeaveOneOut,
    LaneStatus,
    PairwiseDiagnostic,
    PortfolioReport,
    SeedFact,
    StudyFinalizeResult,
    StudyFreezeResult,
    StudyRunResult,
    StudyStatusResult,
    StudyTrialResult,
)
from numereng.platform.errors import PackageError


def portfolio_status(request: PortfolioStatusRequest) -> PortfolioStatusResponse:
    """Resolve the live portfolio status; optionally persist a report file."""

    try:
        report = _portfolio_status(store_root=request.store_root, write=request.write)
    except (PortfolioError, ExperimentError, ValueError) as exc:
        raise PackageError(str(exc)) from exc
    return _to_response(report)


def portfolio_diversity(request: PortfolioDiversityRequest) -> PortfolioDiversityResponse:
    """Build a cross-lane diversity report; translate feature failures to PackageError."""

    try:
        report = _portfolio_diversity(
            store_root=request.store_root,
            lanes=tuple(request.lanes) if request.lanes is not None else None,
            block_length_eras=request.block_length_eras,
            n_resamples=request.n_resamples,
            rng_seed=request.rng_seed,
        )
    except (PortfolioError, ExperimentError, ValueError) as exc:
        raise PackageError(str(exc)) from exc
    return _to_diversity_response(report)


# --------------------------------------------------------------------------- #
# Combination study handlers (P3)
# --------------------------------------------------------------------------- #

_STUDY_ERRORS = (PortfolioError, ExperimentError, StudyConfigError, ValueError)


def study_freeze(request: StudyFreezeRequest) -> StudyFreezeResponse:
    """Run the freeze preflight and materialize an unsealed study snapshot."""

    try:
        result = _study_freeze(store_root=request.store_root, config_path=request.config_path)
    except _STUDY_ERRORS as exc:
        raise PackageError(str(exc)) from exc
    return _freeze_response(result)


def study_run(request: StudyRunRequest) -> StudyRunResponse:
    """Score a trials file against a frozen study; translate failures to PackageError."""

    try:
        result = _study_run(
            store_root=request.store_root,
            trials_path=request.trials_path,
            experiment_id=request.experiment_id,
        )
    except _STUDY_ERRORS as exc:
        raise PackageError(str(exc)) from exc
    return _run_response(result)


def study_finalize(request: StudyFinalizeRequest) -> StudyFinalizeResponse:
    """Score the selected trial on holdout and seal the study."""

    try:
        result = _study_finalize(
            store_root=request.store_root,
            study_id=request.study_id,
            select=request.select,
            experiment_id=request.experiment_id,
        )
    except _STUDY_ERRORS as exc:
        raise PackageError(str(exc)) from exc
    return _finalize_response(result)


def study_status(request: StudyStatusRequest) -> StudyStatusResponse:
    """Return the read-only lifecycle snapshot for one study."""

    try:
        result = _study_status(
            store_root=request.store_root,
            study_id=request.study_id,
            experiment_id=request.experiment_id,
        )
    except _STUDY_ERRORS as exc:
        raise PackageError(str(exc)) from exc
    return _status_response(result)


def _freeze_response(result: StudyFreezeResult) -> StudyFreezeResponse:
    return StudyFreezeResponse(
        study_id=result.study_id,
        study_dir=result.study_dir,
        experiment_id=result.experiment_id,
        frozen=result.frozen,
        n_members=result.n_members,
        n_lanes=result.n_lanes,
        n_search_folds=result.n_search_folds,
        holdout_n_eras=result.holdout_n_eras,
        surface_id=result.surface_id,
        holdout_fingerprint=result.holdout_fingerprint,
        exploratory=result.exploratory,
    )


def _run_response(result: StudyRunResult) -> StudyRunResponse:
    return StudyRunResponse(
        study_id=result.study_id,
        study_dir=result.study_dir,
        executed=result.executed,
        skipped=result.skipped,
        superseded=result.superseded,
        trial_cap=result.trial_cap,
        ledger_path=result.ledger_path,
        trials=[_trial_response(trial) for trial in result.trials],
    )


def _trial_response(trial: StudyTrialResult) -> StudyTrialResponse:
    return StudyTrialResponse(
        trial_id=trial.trial_id,
        pooled_search_bmc=trial.pooled_search_bmc,
        baseline_pooled_search_bmc=trial.baseline_pooled_search_bmc,
        diff_mean=trial.diff_mean,
        diff_ci90_low=trial.diff_ci90_low,
        diff_ci90_high=trial.diff_ci90_high,
        diff_prob_positive=trial.diff_prob_positive,
        n_folds=trial.n_folds,
        status=trial.status,
    )


def _finalize_response(result: StudyFinalizeResult) -> StudyFinalizeResponse:
    return StudyFinalizeResponse(
        study_id=result.study_id,
        study_dir=result.study_dir,
        selected_trial=result.selected_trial,
        is_baseline=result.is_baseline,
        holdout_bmc=result.holdout_bmc,
        baseline_holdout_bmc=result.baseline_holdout_bmc,
        holdout_diff=result.holdout_diff,
        degradation=result.degradation,
        holdout_ci90_low=result.holdout_ci90_low,
        holdout_ci90_high=result.holdout_ci90_high,
        holdout_prob_positive=result.holdout_prob_positive,
        sealed=result.sealed,
        artifacts_dir=result.artifacts_dir,
    )


def _status_response(result: StudyStatusResult) -> StudyStatusResponse:
    return StudyStatusResponse(
        study_id=result.study_id,
        study_dir=result.study_dir,
        frozen=result.frozen,
        sealed=result.sealed,
        trials_executed=result.trials_executed,
        trial_cap=result.trial_cap,
        selected_trial=result.selected_trial,
    )


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #


def _to_response(report: PortfolioReport) -> PortfolioStatusResponse:
    return PortfolioStatusResponse(
        schema_version=report.schema_version,
        portfolio_present=report.portfolio_present,
        generated_at=report.generated_at,
        policy_hash=report.policy_hash,
        policy_gaps=list(report.policy_gaps),
        lanes=[_lane_response(lane) for lane in report.lanes],
        blockers=list(report.blockers),
        registry_path=report.registry_path,
        report_path=report.report_path,
    )


def _lane_response(lane: LaneStatus) -> PortfolioLaneResponse:
    return PortfolioLaneResponse(
        lane_id=lane.lane_id,
        axis=lane.axis,
        structural=lane.structural,
        research_stage_asserted=lane.research_stage_asserted,
        research_stage_evidenced=lane.research_stage_evidenced,
        deployment_stage_asserted=lane.deployment_stage_asserted,
        deployment_stage_evidenced=lane.deployment_stage_evidenced,
        combination_stage_asserted=lane.combination_stage_asserted,
        combination_stage_evidenced=lane.combination_stage_evidenced,
        tranche_rounds_completed=lane.tranche_rounds_completed,
        tranche_approved_rounds=lane.tranche_approved_rounds,
        tranche_max_rounds=lane.tranche_max_rounds,
        observed_seed_noise=lane.observed_seed_noise,
        candidates=[_candidate_response(candidate) for candidate in lane.candidates],
        surface_match=lane.surface_match,
        drift=lane.drift,
        blockers=list(lane.blockers),
        latest_diversity_report_id=lane.latest_diversity_report_id,
    )


def _candidate_response(candidate: CandidateFact) -> PortfolioCandidateResponse:
    return PortfolioCandidateResponse(
        candidate_id=candidate.candidate_id,
        role=candidate.role,
        anchor_config=candidate.anchor_config,
        recipe_key=candidate.recipe_key,
        evidence_tier=candidate.evidence_tier,
        seeds_present=list(candidate.seeds_present),
        trio_complete=candidate.trio_complete,
        trio_bmc_mean=candidate.trio_bmc_mean,
        bmc_sd=candidate.bmc_sd,
        per_seed=[_seed_response(seed) for seed in candidate.per_seed],
        surface_ids=list(candidate.surface_ids),
        surface_match=candidate.surface_match,
        blockers=list(candidate.blockers),
    )


def _seed_response(seed: SeedFact) -> PortfolioSeedResponse:
    return PortfolioSeedResponse(
        seed=seed.seed,
        run_id=seed.run_id,
        bmc=seed.bmc,
        fnc=seed.fnc,
        artifact_mode=seed.artifact_mode,
        training_profile=seed.training_profile,
        experiment_id=seed.experiment_id,
        config=seed.config,
        config_hash_ok=seed.config_hash_ok,
        surface_id=seed.surface_id,
        surface_unavailable_reason=seed.surface_unavailable_reason,
        journal_vs_disk_bmc_delta=seed.journal_vs_disk_bmc_delta,
        duplicate_run_ids=list(seed.duplicate_run_ids),
    )


def _to_diversity_response(report: DiversityReport) -> PortfolioDiversityResponse:
    return PortfolioDiversityResponse(
        schema_version=report.schema_version,
        report_id=report.report_id,
        generated_at=report.generated_at,
        report_dir=report.report_dir,
        surface_id=report.surface_id,
        policy_hash=report.policy_hash,
        diversity_bmc_tolerance=report.diversity_bmc_tolerance,
        inference=DiversityInferenceResponse(
            block_length_eras=report.inference.block_length_eras,
            n_resamples=report.inference.n_resamples,
            rng_seed=report.inference.rng_seed,
        ),
        n_eras=report.n_eras,
        members=[_member_response(member) for member in report.members],
        included_lanes=list(report.included_lanes),
        excluded_candidates=[list(item) for item in report.excluded_candidates],
        blend_bmc_mean=report.blend_bmc_mean,
        pairwise=[_pairwise_response(pair) for pair in report.pairwise],
        leave_one_out=[_leave_one_out_response(loo) for loo in report.leave_one_out],
    )


def _member_response(member: DiversityMember) -> DiversityMemberResponse:
    return DiversityMemberResponse(
        candidate_id=member.candidate_id,
        lane_id=member.lane_id,
        recipe_key=member.recipe_key,
        run_ids=list(member.run_ids),
        prediction_sha256=list(member.prediction_sha256),
        trio_bmc200=member.trio_bmc200,
    )


def _pairwise_response(pair: PairwiseDiagnostic) -> DiversityPairwiseResponse:
    return DiversityPairwiseResponse(
        left=pair.left,
        right=pair.right,
        spearman_mean=pair.spearman_mean,
        spearman_p10=pair.spearman_p10,
        spearman_p90=pair.spearman_p90,
        spearman_min=pair.spearman_min,
        bmc_series_corr=pair.bmc_series_corr,
        joint_drawdown_fraction=pair.joint_drawdown_fraction,
    )


def _leave_one_out_response(loo: LaneLeaveOneOut) -> DiversityLeaveOneOutResponse:
    return DiversityLeaveOneOutResponse(
        lane_id=loo.lane_id,
        blend_bmc_mean=loo.blend_bmc_mean,
        loo_bmc_mean=loo.loo_bmc_mean,
        mean_diff=loo.mean_diff,
        ci90_low=loo.ci90_low,
        ci90_high=loo.ci90_high,
        prob_positive=loo.prob_positive,
    )


__all__ = [
    "portfolio_diversity",
    "portfolio_status",
    "study_finalize",
    "study_freeze",
    "study_run",
    "study_status",
]
