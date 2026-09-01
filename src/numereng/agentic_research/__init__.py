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
from numereng.agentic_research.engine.program import (
    ProgramCoreResult,
    check_program_core,
    resplice_program_core,
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
    "ProgramCoreResult",
    "ResearchBestRun",
    "ResearchRoundResult",
    "ResearchRunResult",
    "ResearchStatusResult",
    "get_closeout_status",
    "check_program_core",
    "get_research_status",
    "program_markdown",
    "resplice_program_core",
    "run_closeout",
    "run_research",
]
