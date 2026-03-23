"""Public surface for agentic research supervisor services."""

from numereng.features.agentic_research.contracts import (
    CodexConfigPayload,
    CodexDecision,
    ResearchBestRun,
    ResearchInitResult,
    ResearchLineageLink,
    ResearchLineageState,
    ResearchPathState,
    ResearchPhaseState,
    ResearchProgramState,
    ResearchRoundState,
    ResearchRunResult,
    ResearchStatusResult,
    ResearchStrategyId,
)
from numereng.features.agentic_research.service import (
    AgenticResearchError,
    AgenticResearchNotInitializedError,
    AgenticResearchValidationError,
    get_research_status,
    init_research,
    run_research,
)

__all__ = [
    "AgenticResearchError",
    "AgenticResearchNotInitializedError",
    "AgenticResearchValidationError",
    "CodexConfigPayload",
    "CodexDecision",
    "ResearchBestRun",
    "ResearchInitResult",
    "ResearchLineageLink",
    "ResearchLineageState",
    "ResearchPathState",
    "ResearchPhaseState",
    "ResearchProgramState",
    "ResearchRoundState",
    "ResearchRunResult",
    "ResearchStrategyId",
    "ResearchStatusResult",
    "get_research_status",
    "init_research",
    "run_research",
]
