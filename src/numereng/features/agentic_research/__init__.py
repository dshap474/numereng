"""Public surface for the minimal agentic config-research loop."""

from numereng.features.agentic_research.run import (
    AgenticResearchError,
    AgenticResearchValidationError,
    ResearchBestRun,
    ResearchRoundResult,
    ResearchRunResult,
    ResearchStatusResult,
    get_research_status,
    program_markdown,
    run_research,
)

__all__ = [
    "AgenticResearchError",
    "AgenticResearchValidationError",
    "ResearchBestRun",
    "ResearchRoundResult",
    "ResearchRunResult",
    "ResearchStatusResult",
    "get_research_status",
    "program_markdown",
    "run_research",
]
