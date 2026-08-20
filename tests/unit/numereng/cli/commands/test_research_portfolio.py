"""CLI tests for `numereng research portfolio status|report` (exit codes 0/1/2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from numereng.cli.main import main
from tests.unit.numereng.features.research_portfolio import _portfolio_fixtures as fx


def _trio_store(tmp_path: Path) -> fx.Store:
    store = fx.build_store(tmp_path)
    for seed, name, run_id, bmc in (
        (42, "config_010_s42.json", "r42", 0.0050),
        (17, "config_010_s17.json", "r17", 0.0040),
        (99, "config_010_s99.json", "r99", 0.0045),
    ):
        config = fx.valid_config(random_state=seed, predictions_name=f"pred_s{seed}")
        fx.write_config(store, name, config)
        fx.build_run(store, run_id=run_id, config=config, bmc=bmc)
    fx.write_journal(
        store,
        [
            fx.journal_row("config_010_s42.json", seed=42, metric=0.0050, run_id="r42"),
            fx.journal_row("config_010_s17.json", seed=17, metric=0.0040, run_id="r17"),
            fx.journal_row("config_010_s99.json", seed=99, metric=0.0045, run_id="r99"),
        ],
    )
    fx.write_state(store, {"total_rounds_completed": 3, "believed_best": {"config": "config_010_s42.json"}})
    fx.write_registry(
        store,
        fx.registry_payload(
            store=store,
            candidates=[{"candidate_id": "c1", "role": "believed_best", "anchor_config": "config_010_s42.json"}],
        ),
    )
    return store


def test_status_table_exit_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _trio_store(tmp_path)
    rc = main(["research", "portfolio", "status", "--workspace", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "portfolio_present: True" in out
    assert "medium_ender20" in out


def test_status_json_exit_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _trio_store(tmp_path)
    rc = main(["research", "portfolio", "status", "--workspace", str(tmp_path), "--format", "json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["portfolio_present"] is True
    assert payload["lanes"][0]["candidates"][0]["trio_complete"] is True


def test_report_persists_and_exits_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    store = _trio_store(tmp_path)
    rc = main(["research", "portfolio", "report", "--workspace", str(tmp_path)])
    assert rc == 0
    assert any((store.root / "portfolio" / "reports").glob("status-*.json"))


def test_absent_portfolio_exits_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fx.build_store(tmp_path)
    rc = main(["research", "portfolio", "status", "--workspace", str(tmp_path)])
    assert rc == 0
    assert "portfolio_present: False" in capsys.readouterr().out


def test_bad_format_exits_two(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _trio_store(tmp_path)
    rc = main(["research", "portfolio", "status", "--workspace", str(tmp_path), "--format", "xml"])
    assert rc == 2
    assert "invalid value for --format" in capsys.readouterr().err


def test_unknown_subcommand_exits_two(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["research", "portfolio", "banana"])
    assert rc == 2


def test_malformed_registry_exits_one(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    store = fx.build_store(tmp_path)
    path = store.root / "portfolio" / "registry.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    rc = main(["research", "portfolio", "status", "--workspace", str(tmp_path)])
    assert rc == 1
    assert "registry_read_failed" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# diversity subcommand
# --------------------------------------------------------------------------- #


def test_diversity_table_exit_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fx.build_diversity_store(tmp_path)
    rc = main(["research", "portfolio", "diversity", "--workspace", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "included_lanes: lane_alpha, lane_beta" in out
    assert "members: 2" in out


def test_diversity_json_shape_exit_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fx.build_diversity_store(tmp_path)
    rc = main(["research", "portfolio", "diversity", "--workspace", str(tmp_path), "--format", "json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["surface_id"] is not None
    assert payload["included_lanes"] == ["lane_alpha", "lane_beta"]
    assert {member["candidate_id"] for member in payload["members"]} == {"cand_alpha", "cand_beta"}
    assert len(payload["pairwise"]) == 1
    assert {loo["lane_id"] for loo in payload["leave_one_out"]} == {"lane_alpha", "lane_beta"}
    assert payload["inference"]["block_length_eras"] == 10


def test_diversity_blocked_exits_one(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # Only one lane selected -> need_two_lanes surfaces as a PackageError (exit 1).
    fx.build_diversity_store(tmp_path)
    rc = main(["research", "portfolio", "diversity", "--workspace", str(tmp_path), "--lanes", "lane_alpha"])
    assert rc == 1
    assert "need_two_lanes" in capsys.readouterr().err


def test_diversity_bad_format_exits_two(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fx.build_diversity_store(tmp_path)
    rc = main(["research", "portfolio", "diversity", "--workspace", str(tmp_path), "--format", "xml"])
    assert rc == 2
    assert "invalid value for --format" in capsys.readouterr().err
