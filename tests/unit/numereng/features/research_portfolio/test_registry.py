"""Feature-level registry loader tests (spec §2.1).

Missing file -> empty portfolio (None); malformed JSON or schema violations raise
a clear feature error.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from numereng.features.research_portfolio.registry import load_registry, registry_path
from numereng.features.research_portfolio.types import PortfolioValidationError
from tests.unit.numereng.features.research_portfolio import _portfolio_fixtures as fx


def test_missing_registry_returns_none(tmp_path: Path) -> None:
    store = fx.build_store(tmp_path)
    assert load_registry(store_root=store.root) is None


def test_valid_registry_loads(tmp_path: Path) -> None:
    store = fx.build_store(tmp_path)
    payload = fx.registry_payload(
        store=store,
        candidates=[{"candidate_id": "c1", "role": "believed_best", "anchor_config": "config_010_s42.json"}],
    )
    fx.write_registry(store, payload)
    config = load_registry(store_root=store.root)
    assert config is not None
    assert config.schema_version == 1
    assert config.lanes[0].lane_id == "medium_ender20"


def test_malformed_json_hard_fails(tmp_path: Path) -> None:
    store = fx.build_store(tmp_path)
    path = registry_path(store_root=store.root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(PortfolioValidationError, match="registry_read_failed"):
        load_registry(store_root=store.root)


def test_schema_violation_hard_fails(tmp_path: Path) -> None:
    store = fx.build_store(tmp_path)
    fx.write_registry(store, {"schema_version": 2, "policy": fx.policy_block()})
    with pytest.raises(PortfolioValidationError, match="registry_schema_invalid"):
        load_registry(store_root=store.root)
