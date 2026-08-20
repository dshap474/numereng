"""Store-layout portfolio constants + classify_run_mode helper tests (P1)."""

from __future__ import annotations

import json
from pathlib import Path

from numereng.features.store import (
    classify_run_mode,
    resolve_portfolio_registry_path,
    resolve_portfolio_reports_root,
    resolve_portfolio_root,
)
from numereng.features.store.layout import CANONICAL_STORE_TOP_LEVEL_DIRS

# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #


def test_portfolio_is_canonical_top_level_dir() -> None:
    assert "portfolio" in CANONICAL_STORE_TOP_LEVEL_DIRS


def test_portfolio_resolvers_nest_under_root(tmp_path: Path) -> None:
    root = tmp_path / ".numereng"
    assert resolve_portfolio_root(store_root=root) == root / "portfolio"
    assert resolve_portfolio_registry_path(store_root=root) == root / "portfolio" / "registry.json"
    assert resolve_portfolio_reports_root(store_root=root) == root / "portfolio" / "reports"


# --------------------------------------------------------------------------- #
# classify_run_mode
# --------------------------------------------------------------------------- #


def _finished_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    for name in ("run.json", "resolved.json", "results.json", "metrics.json"):
        (run_dir / name).write_text(json.dumps({}), encoding="utf-8")


def test_missing_run_dir(tmp_path: Path) -> None:
    assert classify_run_mode(run_dir=tmp_path / "runs" / "absent") == "missing"


def test_incomplete_when_required_file_absent(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text("{}", encoding="utf-8")
    assert classify_run_mode(run_dir=run_dir) == "incomplete"


def test_scoring_when_no_predictions(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "r2"
    _finished_run(run_dir)
    assert classify_run_mode(run_dir=run_dir) == "scoring"


def test_full_when_prediction_parquet_present(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "r3"
    _finished_run(run_dir)
    pred_dir = run_dir / "artifacts" / "predictions"
    pred_dir.mkdir(parents=True)
    (pred_dir / "pred_run.parquet").write_text("stub", encoding="utf-8")
    assert classify_run_mode(run_dir=run_dir) == "full"
