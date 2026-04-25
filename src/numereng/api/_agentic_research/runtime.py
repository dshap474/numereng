"""Agentic config-research API handlers."""

from __future__ import annotations

from numereng.api.contracts import (
    ResearchBestRunResponse,
    ResearchRoundResponse,
    ResearchRunRequest,
    ResearchRunResponse,
    ResearchStatusRequest,
    ResearchStatusResponse,
)
from numereng.features.agentic_research import (
    AgenticResearchError,
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
    """Load the current status for one config-research loop."""
    from numereng import api as api_module

    try:
        result = api_module.get_research_status(
            store_root=request.store_root,
            experiment_id=request.experiment_id,
        )
    except (AgenticResearchError, ExperimentError, ValueError) as exc:
        raise PackageError(str(exc)) from exc
    return ResearchStatusResponse(
        experiment_id=result.experiment_id,
        status=result.status,
        next_round_number=result.next_round_number,
        total_rounds_completed=result.total_rounds_completed,
        last_checkpoint=result.last_checkpoint,
        stop_reason=result.stop_reason,
        best_overall=_best_response(result.best_overall),
        agentic_research_dir=str(result.agentic_research_dir),
        state_path=str(result.state_path),
        ledger_path=str(result.ledger_path),
        program_path=str(result.program_path),
    )


def research_run(request: ResearchRunRequest) -> ResearchRunResponse:
    """Run one foreground config-research loop."""
    from numereng import api as api_module

    try:
        result = api_module.run_research(
            store_root=request.store_root,
            experiment_id=request.experiment_id,
            max_rounds=request.max_rounds,
        )
    except (
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
        experiment_id=result.experiment_id,
        status=result.status,
        next_round_number=result.next_round_number,
        total_rounds_completed=result.total_rounds_completed,
        last_checkpoint=result.last_checkpoint,
        stop_reason=result.stop_reason,
        best_overall=_best_response(result.best_overall),
        rounds=[
            ResearchRoundResponse(
                round_number=item.round_number,
                round_label=item.round_label,
                action=item.action,
                status=item.status,
                config_path=str(item.config_path) if item.config_path is not None else None,
                run_id=item.run_id,
                metric_value=item.metric_value,
                learning=item.learning,
                artifact_dir=str(item.artifact_dir),
            )
            for item in result.rounds
        ],
        interrupted=result.interrupted,
    )


def _best_response(best: object) -> ResearchBestRunResponse:
    return ResearchBestRunResponse(
        experiment_id=getattr(best, "experiment_id", None),
        run_id=getattr(best, "run_id", None),
        bmc_last_200_eras_mean=getattr(best, "bmc_last_200_eras_mean", None),
        bmc_mean=getattr(best, "bmc_mean", None),
        corr_mean=getattr(best, "corr_mean", None),
        mmc_mean=getattr(best, "mmc_mean", None),
        cwmm_mean=getattr(best, "cwmm_mean", None),
        updated_at=getattr(best, "updated_at", None),
    )
