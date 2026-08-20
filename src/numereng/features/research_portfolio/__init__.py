"""Public surface for the research-portfolio feature (registry + status/report)."""

from numereng.features.research_portfolio.combination import (
    study_finalize,
    study_freeze,
    study_run,
    study_status,
)
from numereng.features.research_portfolio.diversity import (
    latest_diversity_report_id,
    portfolio_diversity,
)
from numereng.features.research_portfolio.registry import load_registry, registry_path
from numereng.features.research_portfolio.resolve import resolve_lane
from numereng.features.research_portfolio.status import portfolio_report, portfolio_status
from numereng.features.research_portfolio.surface import SurfaceResult, compute_surface_id
from numereng.features.research_portfolio.types import (
    CandidateFact,
    DiversityReport,
    LaneStatus,
    PortfolioError,
    PortfolioReport,
    PortfolioValidationError,
    SeedFact,
    StudyFinalizeResult,
    StudyFreezeResult,
    StudyRunResult,
    StudyStatusResult,
    StudyTrialResult,
)

__all__ = [
    "CandidateFact",
    "DiversityReport",
    "LaneStatus",
    "PortfolioError",
    "PortfolioReport",
    "PortfolioValidationError",
    "SeedFact",
    "StudyFinalizeResult",
    "StudyFreezeResult",
    "StudyRunResult",
    "StudyStatusResult",
    "StudyTrialResult",
    "SurfaceResult",
    "compute_surface_id",
    "latest_diversity_report_id",
    "load_registry",
    "portfolio_diversity",
    "portfolio_report",
    "portfolio_status",
    "registry_path",
    "resolve_lane",
    "study_finalize",
    "study_freeze",
    "study_run",
    "study_status",
]
