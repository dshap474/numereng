"""Compatibility facade for HPO, ensemble, store, docs, and serving contracts."""

from __future__ import annotations

from numereng.api._contracts.docs import *  # noqa: F403
from numereng.api._contracts.docs import __all__ as _docs_all
from numereng.api._contracts.ensemble import *  # noqa: F403
from numereng.api._contracts.ensemble import __all__ as _ensemble_all
from numereng.api._contracts.hpo import *  # noqa: F403
from numereng.api._contracts.hpo import __all__ as _hpo_all
from numereng.api._contracts.serving import *  # noqa: F403
from numereng.api._contracts.serving import __all__ as _serving_all
from numereng.api._contracts.store import *  # noqa: F403
from numereng.api._contracts.store import __all__ as _store_all

__all__ = [*_hpo_all, *_ensemble_all, *_store_all, *_serving_all, *_docs_all]
