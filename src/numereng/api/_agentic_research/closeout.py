"""Closeout-chain API handlers (delegate to the feature runner, translate errors to PackageError)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numereng.agentic_research import AgenticResearchError
from numereng.api.contracts import (
    ResearchCloseoutPhaseResponse,
    ResearchCloseoutRequest,
    ResearchCloseoutResponse,
    ResearchCloseoutStatusRequest,
)
from numereng.features.experiments import ExperimentError
from numereng.platform.errors import PackageError

if TYPE_CHECKING:
    from numereng.agentic_research import CloseoutResult


# --------------------------------------------------------------------------- #
# Handlers
# --------------------------------------------------------------------------- #
def research_closeout(request: ResearchCloseoutRequest) -> ResearchCloseoutResponse:
    """Run the closeout chain up to the requested boundary."""
    from numereng import api as api_module

    try:
        result = api_module.run_closeout(
            store_root=request.store_root,
            experiment_id=request.experiment_id,
            until=request.until,
            restart_from=request.restart_from,
            memory_root=request.memory_root,
            accept_stale_running=request.accept_stale_running,
            allow_incomplete=request.allow_incomplete,
        )
    except (AgenticResearchError, ExperimentError, ValueError) as exc:
        raise PackageError(str(exc)) from exc
    return _response(result)


def research_closeout_status(request: ResearchCloseoutStatusRequest) -> ResearchCloseoutResponse:
    """Read-only report of the closeout chain state."""
    from numereng import api as api_module

    try:
        result = api_module.get_closeout_status(
            store_root=request.store_root,
            experiment_id=request.experiment_id,
        )
    except (AgenticResearchError, ExperimentError, ValueError) as exc:
        raise PackageError(str(exc)) from exc
    return _response(result)


# --------------------------------------------------------------------------- #
# Mapping
# --------------------------------------------------------------------------- #
def _response(result: CloseoutResult) -> ResearchCloseoutResponse:
    return ResearchCloseoutResponse(
        experiment_id=result.experiment_id,
        phases=[
            ResearchCloseoutPhaseResponse(
                name=phase.name,
                status=phase.status,
                notes=phase.notes,
                duration_seconds=phase.duration_seconds,
                outputs=dict(phase.outputs),
            )
            for phase in result.phases
        ],
        stopped_at_phase=result.stopped_at_phase,
        error=result.error,
    )
