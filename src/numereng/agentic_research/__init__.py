"""Public surface for the minimal agentic config-research loop."""

from numereng.agentic_research.engine.closeout import (
    CloseoutPhaseReport,
    CloseoutResult,
    get_closeout_status,
    run_closeout,
)
from numereng.agentic_research.engine.loop import (
    get_research_status,
    program_markdown,
    run_research,
)
from numereng.agentic_research.engine.types import (
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
    "CloseoutPhaseReport",
    "CloseoutResult",
    "ResearchBestRun",
    "ResearchRoundResult",
    "ResearchRunResult",
    "ResearchStatusResult",
    "get_closeout_status",
    "get_research_status",
    "program_markdown",
    "run_closeout",
    "run_research",
]
