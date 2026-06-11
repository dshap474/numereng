"""Public surface for the minimal agentic config-research loop."""

from numereng.features.agentic_research.loop import (
    get_research_status,
    program_markdown,
    run_research,
)
from numereng.features.agentic_research.types import (
    AgenticResearchError,
    AgenticResearchValidationError,
    ResearchBestRun,
    ResearchRoundResult,
    ResearchRunResult,
    ResearchStatusResult,
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
