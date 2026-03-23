"""Agentic research API handlers."""

from __future__ import annotations

from numereng.api.contracts import (
    ResearchBestRunResponse,
    ResearchInitRequest,
    ResearchInitResponse,
    ResearchPhaseResponse,
    ResearchRoundResponse,
    ResearchRunRequest,
    ResearchRunResponse,
    ResearchStatusRequest,
    ResearchStatusResponse,
)
from numereng.features.agentic_research import (
    AgenticResearchError,
    AgenticResearchNotInitializedError,
    AgenticResearchValidationError,
)
from numereng.features.experiments import ExperimentError
from numereng.features.training import (
    TrainingConfigError,
    TrainingDataError,
    TrainingError,
    TrainingMetricsError,
    TrainingModelError,
)
from numereng.platform.errors import PackageError


def research_init(request: ResearchInitRequest) -> ResearchInitResponse:
    """Initialize one agentic research supervisor rooted at one experiment."""
    from numereng import api as api_module

    try:
        result = api_module.init_research_program(
            store_root=request.store_root,
            experiment_id=request.experiment_id,
            strategy=request.strategy,
            improvement_threshold=request.improvement_threshold,
        )
    except (AgenticResearchValidationError, AgenticResearchError, ExperimentError, ValueError) as exc:
        raise PackageError(str(exc)) from exc

    return ResearchInitResponse(
        root_experiment_id=result.root_experiment_id,
        strategy=result.strategy,
        strategy_description=result.strategy_description,
        status=result.status,
        active_experiment_id=result.active_experiment_id,
        active_path_id=result.active_path_id,
        improvement_threshold=result.improvement_threshold,
        current_phase=_phase_response(result.current_phase),
        agentic_research_dir=str(result.agentic_research_dir),
        program_path=str(result.program_path),
        lineage_path=str(result.lineage_path),
    )


def research_status(request: ResearchStatusRequest) -> ResearchStatusResponse:
    """Load the current status for one agentic research supervisor."""
    from numereng import api as api_module

    try:
        result = api_module.get_research_program_status(
            store_root=request.store_root,
            experiment_id=request.experiment_id,
        )
    except (AgenticResearchNotInitializedError, AgenticResearchError, ExperimentError, ValueError) as exc:
        raise PackageError(str(exc)) from exc

    current_round = None
    if result.current_round is not None:
        current_round = ResearchRoundResponse(
            round_number=result.current_round.round_number,
            round_label=result.current_round.round_label,
            experiment_id=result.current_round.experiment_id,
            path_id=result.current_round.path_id,
            status=result.current_round.status,
            next_config_index=result.current_round.next_config_index,
            config_filenames=list(result.current_round.config_filenames),
            run_ids=list(result.current_round.run_ids),
            decision_action=result.current_round.decision_action,
            experiment_question=result.current_round.experiment_question,
            winner_criteria=result.current_round.winner_criteria,
            decision_rationale=result.current_round.decision_rationale,
            decision_path_hypothesis=result.current_round.decision_path_hypothesis,
            decision_path_slug=result.current_round.decision_path_slug,
            phase_id=result.current_round.phase_id,
            phase_action=result.current_round.phase_action,
            phase_transition_rationale=result.current_round.phase_transition_rationale,
            started_at=result.current_round.started_at,
            updated_at=result.current_round.updated_at,
        )

    return ResearchStatusResponse(
        root_experiment_id=result.root_experiment_id,
        strategy=result.strategy,
        strategy_description=result.strategy_description,
        status=result.status,
        active_experiment_id=result.active_experiment_id,
        active_path_id=result.active_path_id,
        next_round_number=result.next_round_number,
        total_rounds_completed=result.total_rounds_completed,
        total_paths_created=result.total_paths_created,
        improvement_threshold=result.improvement_threshold,
        last_checkpoint=result.last_checkpoint,
        stop_reason=result.stop_reason,
        best_overall=ResearchBestRunResponse(
            experiment_id=result.best_overall.experiment_id,
            run_id=result.best_overall.run_id,
            bmc_last_200_eras_mean=result.best_overall.bmc_last_200_eras_mean,
            bmc_mean=result.best_overall.bmc_mean,
            corr_mean=result.best_overall.corr_mean,
            mmc_mean=result.best_overall.mmc_mean,
            cwmm_mean=result.best_overall.cwmm_mean,
            updated_at=result.best_overall.updated_at,
        ),
        current_round=current_round,
        current_phase=_phase_response(result.current_phase),
        program_path=str(result.program_path),
        lineage_path=str(result.lineage_path),
    )


def research_run(request: ResearchRunRequest) -> ResearchRunResponse:
    """Run one foreground agentic research loop."""
    from numereng import api as api_module

    try:
        result = api_module.run_research_program(
            store_root=request.store_root,
            experiment_id=request.experiment_id,
            max_rounds=request.max_rounds,
            max_paths=request.max_paths,
        )
    except (
        AgenticResearchNotInitializedError,
        AgenticResearchValidationError,
        AgenticResearchError,
        ExperimentError,
        TrainingConfigError,
        TrainingDataError,
        TrainingModelError,
        TrainingMetricsError,
        TrainingError,
        ValueError,
    ) as exc:
        raise PackageError(str(exc)) from exc

    return ResearchRunResponse(
        root_experiment_id=result.root_experiment_id,
        strategy=result.strategy,
        strategy_description=result.strategy_description,
        status=result.status,
        active_experiment_id=result.active_experiment_id,
        active_path_id=result.active_path_id,
        next_round_number=result.next_round_number,
        total_rounds_completed=result.total_rounds_completed,
        total_paths_created=result.total_paths_created,
        last_checkpoint=result.last_checkpoint,
        stop_reason=result.stop_reason,
        current_phase=_phase_response(result.current_phase),
        interrupted=result.interrupted,
    )


def _phase_response(current_phase: object) -> ResearchPhaseResponse | None:
    if current_phase is None:
        return None
    return ResearchPhaseResponse(
        phase_id=current_phase.phase_id,
        phase_title=current_phase.phase_title,
        status=current_phase.status,
        round_count=current_phase.round_count,
        transition_rationale=current_phase.transition_rationale,
        started_at=current_phase.started_at,
        updated_at=current_phase.updated_at,
    )
