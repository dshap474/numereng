"""Agentic research program CORE check / re-splice API handlers."""

from __future__ import annotations

from numereng.agentic_research import AgenticResearchError, ProgramCoreResult
from numereng.api.contracts import ResearchProgramRequest, ResearchProgramResponse
from numereng.features.experiments import ExperimentError
from numereng.platform.errors import PackageError


def research_program_check(request: ResearchProgramRequest) -> ResearchProgramResponse:
    """Report whether an experiment program's CORE sections match the canonical PROGRAM.md."""
    from numereng import api as api_module

    try:
        result = api_module.check_program_core(store_root=request.store_root, experiment_id=request.experiment_id)
    except (AgenticResearchError, ExperimentError, ValueError) as exc:
        raise PackageError(str(exc)) from exc
    return _response(result)


def research_program_resplice(request: ResearchProgramRequest) -> ResearchProgramResponse:
    """Rewrite a drifted program's CORE sections from PROGRAM.md (backup kept beside the program)."""
    from numereng import api as api_module

    try:
        result = api_module.resplice_program_core(store_root=request.store_root, experiment_id=request.experiment_id)
    except (AgenticResearchError, ExperimentError, ValueError, OSError) as exc:
        raise PackageError(str(exc)) from exc
    return _response(result)


def _response(result: ProgramCoreResult) -> ResearchProgramResponse:
    return ResearchProgramResponse(
        experiment_id=result.experiment_id,
        program_path=str(result.program_path),
        base_program_path=str(result.base_program_path),
        is_base_program=result.is_base_program,
        in_sync=result.in_sync,
        diverging_section=result.diverging_section,
        written=result.written,
        backup_path=None if result.backup_path is None else str(result.backup_path),
    )
