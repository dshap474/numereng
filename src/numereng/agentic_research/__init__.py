"""Public surface for the minimal agentic config-research loop."""

from numereng.agentic_research.engine.closeout import (
    CloseoutResult,
    run_closeout,
)
from numereng.agentic_research.engine.closeout import memo_path as closeout_memo_path
from numereng.agentic_research.engine.loop import (
    get_research_status,
    run_research,
)
from numereng.agentic_research.engine.types import (
    AgenticResearchError,
    AgenticResearchValidationError,
    ResearchRoundResult,
    ResearchRunResult,
    ResearchStatusResult,
)

__all__ = [
    "AgenticResearchError",
    "AgenticResearchValidationError",
    "CloseoutResult",
    "ResearchRoundResult",
    "ResearchRunResult",
    "ResearchStatusResult",
    "closeout_memo_path",
    "get_research_status",
    "run_closeout",
    "run_research",
]
