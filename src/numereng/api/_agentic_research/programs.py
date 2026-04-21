"""Research program catalog and initialization API handlers."""

from __future__ import annotations

from numereng.api._agentic_research.responses import phase_response
from numereng.api.contracts import (
    ResearchInitRequest,
    ResearchInitResponse,
    ResearchProgramCatalogEntryResponse,
    ResearchProgramConfigPolicyResponse,
    ResearchProgramListRequest,
    ResearchProgramListResponse,
    ResearchProgramMetricPolicyResponse,
    ResearchProgramPhaseResponse,
    ResearchProgramRoundPolicyResponse,
    ResearchProgramShowRequest,
    ResearchProgramShowResponse,
)
from numereng.features.agentic_research import AgenticResearchError, AgenticResearchValidationError
from numereng.features.experiments import ExperimentError
from numereng.platform.errors import PackageError


def research_init(request: ResearchInitRequest) -> ResearchInitResponse:
    """Initialize one agentic research supervisor rooted at one experiment."""
    from numereng import api as api_module

    try:
        result = api_module.init_research_program(
            store_root=request.store_root,
            experiment_id=request.experiment_id,
            program_id=request.program_id,
            improvement_threshold=request.improvement_threshold,
        )
    except (AgenticResearchValidationError, AgenticResearchError, ExperimentError, ValueError) as exc:
        raise PackageError(str(exc)) from exc
    return ResearchInitResponse(
        root_experiment_id=result.root_experiment_id,
        program_id=result.program_id,
        program_title=result.program_title,
        status=result.status,
        active_experiment_id=result.active_experiment_id,
        active_path_id=result.active_path_id,
        improvement_threshold=result.improvement_threshold,
        current_phase=phase_response(result.current_phase),
        agentic_research_dir=str(result.agentic_research_dir),
        program_path=str(result.program_path),
        lineage_path=str(result.lineage_path),
        session_program_path=str(result.session_program_path),
    )


def research_program_list(request: ResearchProgramListRequest) -> ResearchProgramListResponse:
    """List available built-in and user research programs."""
    from numereng import api as api_module

    try:
        programs = api_module.list_research_program_records(
            user_programs_dir=request.workspace_layout.research_programs_root
        )
    except (AgenticResearchValidationError, AgenticResearchError, ValueError) as exc:
        raise PackageError(str(exc)) from exc
    return ResearchProgramListResponse(
        programs=[
            ResearchProgramCatalogEntryResponse(
                program_id=item.program_id,
                title=item.title,
                description=item.description,
                source=item.source,
                planner_contract=item.planner_contract,
                phase_aware=item.phase_aware,
                source_path=item.source_path,
            )
            for item in programs
        ]
    )


def research_program_show(request: ResearchProgramShowRequest) -> ResearchProgramShowResponse:
    """Show one research program definition and raw markdown."""
    from numereng import api as api_module

    try:
        details = api_module.get_research_program_record(
            request.program_id,
            user_programs_dir=request.workspace_layout.research_programs_root,
        )
    except (AgenticResearchValidationError, AgenticResearchError, ValueError) as exc:
        raise PackageError(str(exc)) from exc

    definition = details.definition
    return ResearchProgramShowResponse(
        program_id=definition.program_id,
        title=definition.title,
        description=definition.description,
        source=definition.source,
        planner_contract=definition.planner_contract,
        scoring_stage=definition.scoring_stage,
        metric_policy=ResearchProgramMetricPolicyResponse(
            primary=definition.metric_policy.primary,
            tie_break=definition.metric_policy.tie_break,
            sanity_checks=list(definition.metric_policy.sanity_checks),
        ),
        round_policy=ResearchProgramRoundPolicyResponse(
            plateau_non_improving_rounds=definition.round_policy.plateau_non_improving_rounds,
            require_scale_confirmation=definition.round_policy.require_scale_confirmation,
            scale_confirmation_rounds=definition.round_policy.scale_confirmation_rounds,
        ),
        improvement_threshold_default=definition.improvement_threshold_default,
        config_policy=ResearchProgramConfigPolicyResponse(
            allowed_paths=list(definition.config_policy.allowed_paths),
            min_candidate_configs=definition.config_policy.min_candidate_configs,
            max_candidate_configs=definition.config_policy.max_candidate_configs,
            min_changes=definition.config_policy.min_changes,
            max_changes=definition.config_policy.max_changes,
        ),
        phases=[
            ResearchProgramPhaseResponse(
                phase_id=phase.phase_id,
                title=phase.title,
                summary=phase.summary,
                gate=phase.gate,
            )
            for phase in definition.phases
        ],
        source_path=definition.source_path,
        raw_markdown=details.raw_markdown,
    )
