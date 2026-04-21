"""Agentic research runtime API handlers."""

from __future__ import annotations

from numereng.api._agentic_research.responses import phase_response, round_response
from numereng.api.contracts import ResearchBestRunResponse, ResearchRunRequest, ResearchRunResponse, ResearchStatusRequest, ResearchStatusResponse
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
    return ResearchStatusResponse(
        root_experiment_id=result.root_experiment_id,
        program_id=result.program_id,
        program_title=result.program_title,
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
        current_round=round_response(result.current_round),
        current_phase=phase_response(result.current_phase),
        program_path=str(result.program_path),
        lineage_path=str(result.lineage_path),
        session_program_path=str(result.session_program_path),
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
        program_id=result.program_id,
        program_title=result.program_title,
        status=result.status,
        active_experiment_id=result.active_experiment_id,
        active_path_id=result.active_path_id,
        next_round_number=result.next_round_number,
        total_rounds_completed=result.total_rounds_completed,
        total_paths_created=result.total_paths_created,
        last_checkpoint=result.last_checkpoint,
        stop_reason=result.stop_reason,
        current_phase=phase_response(result.current_phase),
        interrupted=result.interrupted,
    )
