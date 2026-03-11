"""Internal contract aggregator for the public `numereng.api.contracts` facade."""

from __future__ import annotations

from numereng.api._contracts_base import *  # noqa: F403
from numereng.api._contracts_base import __all__ as _base_all
from numereng.api._contracts_ops import *  # noqa: F403
from numereng.api._contracts_ops import __all__ as _ops_all

__all__ = [*_base_all, *_ops_all]
