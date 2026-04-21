"""Compatibility facade for shared/run/experiment/research contracts."""

from __future__ import annotations

from numereng.api._contracts.experiment import *  # noqa: F403
from numereng.api._contracts.experiment import __all__ as _experiment_all
from numereng.api._contracts.research import *  # noqa: F403
from numereng.api._contracts.research import __all__ as _research_all
from numereng.api._contracts.run import *  # noqa: F403
from numereng.api._contracts.run import __all__ as _run_all
from numereng.api._contracts.shared import *  # noqa: F403
from numereng.api._contracts.shared import __all__ as _shared_all

__all__ = [*_shared_all, *_run_all, *_experiment_all, *_research_all]
