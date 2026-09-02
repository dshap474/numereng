"""Agentic config-research API handlers."""

from __future__ import annotations

from pathlib import Path

from numereng.agentic_research import (
    AgenticResearchError,
    AgenticResearchValidationError,
    closeout_memo_path,
)
from numereng.api.contracts import (
    ResearchCloseoutRequest,
    ResearchCloseoutResponse,
    ResearchRoundResponse,
    ResearchRunRequest,
    ResearchRunResponse,
    ResearchStatusRequest,
    ResearchStatusResponse,
)
from numereng.features.experiments import ExperimentError
from numereng.features.training.errors import (
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
        last_round_label=result.last_round_label,
        last_run_id=result.last_run_id,
        stop_reason=result.stop_reason,
        champion=result.champion,
        agentic_research_dir=str(result.agentic_research_dir),
        closeout_memo="present" if closeout_memo_path(Path(result.agentic_research_dir)).is_file() else "absent",
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
        champion=result.champion,
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


def research_closeout(request: ResearchCloseoutRequest) -> ResearchCloseoutResponse:
    """Build the closeout evidence bundle and write the decision memo for one experiment."""
    from numereng import api as api_module

    try:
        result = api_module.run_closeout(
            store_root=request.store_root,
            experiment_id=request.experiment_id,
            allow_incomplete=request.allow_incomplete,
        )
    except (AgenticResearchError, ExperimentError, ValueError) as exc:
        raise PackageError(str(exc)) from exc
    return ResearchCloseoutResponse(
        experiment_id=result.experiment_id,
        evidence_path=str(result.evidence_path),
        memo_path=str(result.memo_path),
        holdout_summary=result.holdout_summary,
    )
