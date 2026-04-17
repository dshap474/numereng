"""Internal contract aggregator for the public `numereng.api.contracts` facade."""

from __future__ import annotations

from numereng.api._contracts_base import *  # noqa: F403
from numereng.api._contracts_base import __all__ as _base_all
from numereng.api._contracts_monitor import *  # noqa: F403
from numereng.api._contracts_monitor import __all__ as _monitor_all
from numereng.api._contracts_ops import *  # noqa: F403
from numereng.api._contracts_ops import __all__ as _ops_all
from numereng.api._contracts_remote import *  # noqa: F403
from numereng.api._contracts_remote import __all__ as _remote_all
from numereng.api._contracts_serving import *  # noqa: F403
from numereng.api._contracts_serving import __all__ as _serving_all
from numereng.api._contracts_viz import *  # noqa: F403
from numereng.api._contracts_viz import __all__ as _viz_all

__all__ = [*_base_all, *_monitor_all, *_ops_all, *_serving_all, *_remote_all, *_viz_all]
