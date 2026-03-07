from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import cast

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from numereng.features.viz.contracts import capabilities_payload
from numereng.features.viz.routes import create_router
from numereng.features.viz.services import VizService
from numereng.features.viz.store_adapter import VizStoreAdapter, VizStoreConfig, resolve_store_root


def test_resolve_store_root_prefers_explicit(tmp_path: Path) -> None:
    explicit = tmp_path / "store"
    explicit.mkdir()

    resolved = resolve_store_root(explicit)

    assert resolved == explicit.resolve()


def test_capabilities_payload_read_only_flag() -> None:
    assert capabilities_payload() == {
        "read_only": True,
        "write_controls": False,
    }


def test_list_experiments_fallback_without_db(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    (store_root / "experiments" / "exp-a").mkdir(parents=True)

    adapter = VizStoreAdapter(
        VizStoreConfig(
            store_root=store_root,
            repo_root=tmp_path
        )
    )

    payload = adapter.list_experiments()

    assert len(payload) == 1
    assert payload[0]["experiment_id"] == "exp-a"


def test_list_experiment_configs_discovers_json_configs(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    config_dir = store_root / "experiments" / "exp-json" / "configs"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "base.json"
    config_path.write_text(
        """
        {
          "run_id": "run-abc",
          "model": {"type": "LGBMRegressor"},
          "data": {"target": "target_20", "feature_set": "small"}
        }
        """.strip(),
        encoding="utf-8",
    )

    adapter = VizStoreAdapter(
        VizStoreConfig(
            store_root=store_root,
            repo_root=tmp_path,
        )
    )

    payload = adapter.list_experiment_configs("exp-json", runnable_only=True)

    assert payload["total"] == 1
    assert payload["items"][0]["config_id"].endswith("base.json")
    assert payload["items"][0]["summary"]["model_type"] == "LGBMRegressor"


def _make_readonly_client(*, repo_root: Path, store_root: Path) -> TestClient:
    return _make_client(repo_root=repo_root, store_root=store_root)


def _make_client(*, repo_root: Path, store_root: Path) -> TestClient:
    adapter = VizStoreAdapter(
        VizStoreConfig(
            store_root=store_root,
            repo_root=repo_root,
        )
    )
    service = VizService(adapter)
    app = FastAPI()
    app.state.viz_service = service
    app.include_router(create_router(service))
    return TestClient(app)


def test_numereng_docs_tree_uses_docs_numereng_root_only(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    store_root.mkdir(parents=True)

    docs_root = tmp_path / "docs"
    docs_root.mkdir(parents=True)
    (docs_root / "CUSTOM_MODELS.md").write_text("# Root-only doc\n")
    (docs_root / "SUMMARY.md").write_text("## Root\n* [Custom Models](CUSTOM_MODELS.md)\n")

    numereng_docs = docs_root / "numereng"
    numereng_docs.mkdir(parents=True)
    (numereng_docs / "README.md").write_text("# Numereng\n")
    (numereng_docs / "SUMMARY.md").write_text("## Getting Started\n* [Overview](README.md)\n")

    adapter = VizStoreAdapter(
        VizStoreConfig(
            store_root=store_root,
            repo_root=tmp_path
        )
    )

    tree = adapter.get_doc_tree("numereng")
    assert tree["sections"][0]["heading"] == "Getting Started"
    assert tree["sections"][0]["items"][0]["path"] == "README.md"
    assert adapter.get_doc_content("numereng", "README.md")["exists"] is True
    assert adapter.get_doc_content("numereng", "CUSTOM_MODELS.md")["exists"] is False


def test_summary_tree_normalizes_relative_paths_and_blocks_escape(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    store_root.mkdir(parents=True)

    docs_numereng = tmp_path / "docs" / "numereng"
    docs_numereng.mkdir(parents=True)
    (docs_numereng / "README.md").write_text("# Overview\n")
    (docs_numereng / "reference").mkdir(parents=True)
    (docs_numereng / "reference" / "signals-+-quantconnect.md").write_text("# Signals\n")
    (docs_numereng / "SUMMARY.md").write_text(
        "\n".join(
            [
                "# Guide",
                "## Reference",
                "* [Overview](./guides/../README.md)",
                "* [Signals + QuantConnect](reference/signals-+-quantconnect.md)",
                "* [Escape](../outside.md)",
            ]
        )
        + "\n"
    )

    adapter = VizStoreAdapter(
        VizStoreConfig(
            store_root=store_root,
            repo_root=tmp_path
        )
    )
    tree = adapter.get_doc_tree("numereng")
    items = tree["sections"][0]["items"]
    paths = [item["path"] for item in items]

    assert "README.md" in paths
    assert "reference/signals-+-quantconnect.md" in paths
    assert "../outside.md" not in paths
    assert adapter.get_doc_content("numereng", "reference/signals-+-quantconnect.md")["exists"] is True


def test_numerai_docs_tree_appends_generated_forum_archive(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    store_root.mkdir(parents=True)

    numerai_docs = tmp_path / "docs" / "numerai"
    numerai_docs.mkdir(parents=True)
    (numerai_docs / "SUMMARY.md").write_text("## Root\n* [Overview](README.md)\n")
    (numerai_docs / "README.md").write_text("# Numerai\n")
    forum_dir = numerai_docs / "forum"
    forum_dir.mkdir(parents=True)
    (forum_dir / "INDEX.md").write_text("# Forum Archive\n")
    (forum_dir / "posts" / "2020" / "03").mkdir(parents=True)
    (forum_dir / "posts" / "2020" / "03" / "INDEX.md").write_text("# 2020/03\n")
    (forum_dir / "posts" / "2020" / "INDEX.md").write_text("# 2020\n")

    adapter = VizStoreAdapter(
        VizStoreConfig(
            store_root=store_root,
            repo_root=tmp_path,
        )
    )

    tree = adapter.get_doc_tree("numerai")
    sections = tree["sections"]
    forum_section = next(section for section in sections if section["heading"] == "Forum Archive")
    forum_item = forum_section["items"][0]

    assert forum_item["title"] == "2020"
    assert forum_item["path"] == "forum/posts/2020/INDEX.md"
    assert forum_item["children"][0]["title"] == "2020/03"
    assert forum_item["children"][0]["path"] == "forum/posts/2020/03/INDEX.md"
    assert adapter.get_doc_content("numerai", "forum/INDEX.md")["exists"] is True


def test_docs_asset_endpoint_serves_domain_and_shared_assets(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    store_root.mkdir(parents=True)

    numerai_docs = tmp_path / "docs" / "numerai"
    (numerai_docs / ".gitbook" / "assets").mkdir(parents=True)
    (numerai_docs / ".gitbook" / "assets" / "domain.png").write_bytes(b"domain-image")

    shared_assets = tmp_path / "docs" / "assets"
    shared_assets.mkdir(parents=True)
    (shared_assets / "shared.png").write_bytes(b"shared-image")

    client = _make_readonly_client(repo_root=tmp_path, store_root=store_root)

    domain_resp = client.get("/api/docs/numerai/asset", params={"path": ".gitbook/assets/domain.png"})
    assert domain_resp.status_code == 200
    assert domain_resp.content == b"domain-image"

    shared_resp = client.get("/api/docs/numerai/asset", params={"path": ".gitbook/assets/shared.png"})
    assert shared_resp.status_code == 200
    assert shared_resp.content == b"shared-image"

    missing_resp = client.get("/api/docs/numerai/asset", params={"path": ".gitbook/assets/missing.png"})
    assert missing_resp.status_code == 404

    traversal_resp = client.get("/api/docs/numerai/asset", params={"path": "../secret.png"})
    assert traversal_resp.status_code == 400


def test_get_resolved_config_prefers_json_when_present(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    run_dir = store_root / "runs" / "run-1"
    run_dir.mkdir(parents=True)

    (run_dir / "resolved.json").write_text("{\"foo\": \"bar\"}")
    (run_dir / "resolved.yaml").write_text("foo: legacy")

    adapter = VizStoreAdapter(
        VizStoreConfig(
            store_root=store_root,
            repo_root=tmp_path
        )
    )

    payload = adapter.get_resolved_config("run-1")

    assert payload == {"yaml": "{\"foo\": \"bar\"}"}


def test_get_metrics_for_runs_normalizes_store_aliases(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    store_root.mkdir(parents=True)
    db_path = store_root / "numereng.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE metrics (run_id TEXT, name TEXT, value REAL, value_json TEXT)")
        conn.execute(
            "INSERT INTO metrics (run_id, name, value, value_json) VALUES (?, ?, ?, ?)",
            ("run-1", "corr.mean", 0.10, None),
        )
        conn.execute(
            "INSERT INTO metrics (run_id, name, value, value_json) VALUES (?, ?, ?, ?)",
            ("run-1", "corr.sharpe", 1.25, None),
        )
        conn.execute(
            "INSERT INTO metrics (run_id, name, value, value_json) VALUES (?, ?, ?, ?)",
            ("run-1", "mmc.mean", 0.02, None),
        )
        conn.execute(
            "INSERT INTO metrics (run_id, name, value, value_json) VALUES (?, ?, ?, ?)",
            ("run-1", "bmc.mean", 0.03, None),
        )
        conn.commit()
    finally:
        conn.close()

    adapter = VizStoreAdapter(
        VizStoreConfig(
            store_root=store_root,
            repo_root=tmp_path
        )
    )

    payload = adapter.get_metrics_for_runs(
        ["run-1"],
        ["corr20v2_mean", "corr20v2_sharpe", "mmc_mean", "payout_estimate_mean", "bmc_mean"],
    )

    assert payload["run-1"]["corr20v2_mean"] == pytest.approx(0.10)
    assert payload["run-1"]["corr20v2_sharpe"] == pytest.approx(1.25)
    assert payload["run-1"]["mmc_mean"] == pytest.approx(0.02)
    assert payload["run-1"]["bmc_mean"] == pytest.approx(0.03)
    assert payload["run-1"]["payout_estimate_mean"] == pytest.approx(0.05)  # clipped payout estimate


def test_get_metrics_for_runs_derives_payout_when_only_requested_metric(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    store_root.mkdir(parents=True)
    db_path = store_root / "numereng.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE metrics (run_id TEXT, name TEXT, value REAL, value_json TEXT)")
        conn.execute(
            "INSERT INTO metrics (run_id, name, value, value_json) VALUES (?, ?, ?, ?)",
            ("run-1", "corr.mean", 0.10, None),
        )
        conn.execute(
            "INSERT INTO metrics (run_id, name, value, value_json) VALUES (?, ?, ?, ?)",
            ("run-1", "mmc.mean", 0.02, None),
        )
        conn.commit()
    finally:
        conn.close()

    adapter = VizStoreAdapter(
        VizStoreConfig(
            store_root=store_root,
            repo_root=tmp_path
        )
    )

    payload = adapter.get_metrics_for_runs(["run-1"], ["payout_estimate_mean"])

    assert payload["run-1"]["payout_estimate_mean"] == pytest.approx(0.05)  # clipped payout estimate


def test_get_metrics_for_runs_preserves_explicit_null_payout(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    store_root.mkdir(parents=True)
    db_path = store_root / "numereng.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE metrics (run_id TEXT, name TEXT, value REAL, value_json TEXT)")
        conn.execute(
            "INSERT INTO metrics (run_id, name, value, value_json) VALUES (?, ?, ?, ?)",
            ("run-1", "corr.mean", 0.10, None),
        )
        conn.execute(
            "INSERT INTO metrics (run_id, name, value, value_json) VALUES (?, ?, ?, ?)",
            ("run-1", "mmc.mean", 0.02, None),
        )
        conn.execute(
            "INSERT INTO metrics (run_id, name, value, value_json) VALUES (?, ?, ?, ?)",
            ("run-1", "payout_estimate_mean", None, None),
        )
        conn.commit()
    finally:
        conn.close()

    adapter = VizStoreAdapter(
        VizStoreConfig(
            store_root=store_root,
            repo_root=tmp_path,
        )
    )

    payload = adapter.get_metrics_for_runs(["run-1"], ["payout_estimate_mean"])

    assert "payout_estimate_mean" in payload["run-1"]
    assert payload["run-1"]["payout_estimate_mean"] is None


def test_get_metrics_for_runs_non_ender_target_does_not_derive_payout(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    store_root.mkdir(parents=True)
    db_path = store_root / "numereng.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE runs (run_id TEXT, manifest_json TEXT)")
        conn.execute("CREATE TABLE metrics (run_id TEXT, name TEXT, value REAL, value_json TEXT)")
        conn.execute(
            "INSERT INTO runs (run_id, manifest_json) VALUES (?, ?)",
            (
                "run-1",
                '{"run_id":"run-1","data":{"target_col":"target_cyrus_20"}}',
            ),
        )
        conn.execute(
            "INSERT INTO metrics (run_id, name, value, value_json) VALUES (?, ?, ?, ?)",
            ("run-1", "corr.mean", 0.10, None),
        )
        conn.execute(
            "INSERT INTO metrics (run_id, name, value, value_json) VALUES (?, ?, ?, ?)",
            ("run-1", "mmc.mean", 0.02, None),
        )
        conn.commit()
    finally:
        conn.close()

    adapter = VizStoreAdapter(
        VizStoreConfig(
            store_root=store_root,
            repo_root=tmp_path,
        )
    )

    payload = adapter.get_metrics_for_runs(["run-1"], ["payout_estimate_mean"])

    assert payload["run-1"] == {}


def test_get_run_metrics_normalizes_nested_metrics_from_filesystem(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    run_dir = store_root / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(
        """
        {
          "corr": {"mean": 0.12, "sharpe": 0.8, "std": 0.2, "max_drawdown": 0.9},
          "mmc": {"mean": 0.01, "sharpe": 0.2},
          "bmc": {"mean": 0.04}
        }
        """.strip(),
        encoding="utf-8",
    )

    adapter = VizStoreAdapter(
        VizStoreConfig(
            store_root=store_root,
            repo_root=tmp_path
        )
    )
    payload = adapter.get_run_metrics("run-1")

    assert payload is not None
    assert payload["corr20v2_mean"] == pytest.approx(0.12)
    assert payload["corr20v2_sharpe"] == pytest.approx(0.8)
    assert payload["mmc_mean"] == pytest.approx(0.01)
    assert payload["bmc_mean"] == pytest.approx(0.04)
    assert payload["max_drawdown"] == pytest.approx(0.9)
    assert payload["payout_estimate_mean"] == pytest.approx(0.05)  # clipped payout estimate


def test_get_run_metrics_non_ender_target_does_not_derive_payout(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    run_dir = store_root / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        '{"run_id":"run-1","data":{"target_col":"target_cyrus_20"}}',
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        '{"corr":{"mean":0.12},"mmc":{"mean":0.01}}',
        encoding="utf-8",
    )

    adapter = VizStoreAdapter(
        VizStoreConfig(
            store_root=store_root,
            repo_root=tmp_path,
        )
    )

    payload = adapter.get_run_metrics("run-1")

    assert payload is not None
    assert "payout_estimate_mean" not in payload


def test_get_run_metrics_normalizes_nested_value_json_from_sqlite(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    store_root.mkdir(parents=True)
    db_path = store_root / "numereng.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE metrics (run_id TEXT, name TEXT, value REAL, value_json TEXT)")
        conn.execute(
            "INSERT INTO metrics (run_id, name, value, value_json) VALUES (?, ?, ?, ?)",
            ("run-1", "corr", None, '{"mean": 0.11, "sharpe": 0.9, "max_drawdown": 0.7}'),
        )
        conn.execute(
            "INSERT INTO metrics (run_id, name, value, value_json) VALUES (?, ?, ?, ?)",
            ("run-1", "mmc", None, '{"mean": 0.01}'),
        )
        conn.execute(
            "INSERT INTO metrics (run_id, name, value, value_json) VALUES (?, ?, ?, ?)",
            ("run-1", "bmc", None, '{"mean": 0.02}'),
        )
        conn.commit()
    finally:
        conn.close()

    adapter = VizStoreAdapter(
        VizStoreConfig(
            store_root=store_root,
            repo_root=tmp_path
        )
    )

    payload = adapter.get_run_metrics("run-1")

    assert payload is not None
    assert payload["corr20v2_mean"] == pytest.approx(0.11)
    assert payload["corr20v2_sharpe"] == pytest.approx(0.9)
    assert payload["mmc_mean"] == pytest.approx(0.01)
    assert payload["bmc_mean"] == pytest.approx(0.02)
    assert payload["max_drawdown"] == pytest.approx(0.7)
    assert payload["payout_estimate_mean"] == pytest.approx(0.05)  # clipped payout estimate


def test_list_experiment_runs_filesystem_uses_target_col(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    run_dir = store_root / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        """
        {
          "run_id": "run-1",
          "experiment_id": "exp-1",
          "created_at": "2026-02-22T00:00:00+00:00",
          "status": "FINISHED",
          "model": {"type": "LGBMRegressor"},
          "data": {"target_col": "target_ender_20", "feature_set": "small"}
        }
        """.strip(),
        encoding="utf-8",
    )

    adapter = VizStoreAdapter(
        VizStoreConfig(
            store_root=store_root,
            repo_root=tmp_path
        )
    )

    payload = adapter.list_experiment_runs("exp-1")
    assert len(payload) == 1
    assert payload[0]["target_train"] == "target_ender_20"
    assert payload[0]["target_col"] == "target_ender_20"
    assert payload[0]["target"] == "target_ender_20"


def test_get_run_metrics_enriches_mmc_coverage_ratio_from_provenance(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    run_dir = store_root / "runs" / "run-1"
    run_dir.mkdir(parents=True)

    (run_dir / "run.json").write_text(
        '{"run_id": "run-1", "data": {"target_col": "target_ender_20"}}',
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text('{"corr": {"mean": 0.12}, "mmc": {"mean": 0.01}}', encoding="utf-8")
    (run_dir / "score_provenance.json").write_text(
        """
        {
          "joins": {
            "predictions_rows": 100,
            "meta_overlap_rows": 80
          }
        }
        """.strip(),
        encoding="utf-8",
    )

    adapter = VizStoreAdapter(
        VizStoreConfig(
            store_root=store_root,
            repo_root=tmp_path
        )
    )

    payload = adapter.get_run_metrics("run-1")
    assert payload is not None
    assert payload["mmc_coverage_ratio_rows"] == pytest.approx(0.8)


def test_per_era_fallback_derives_corr_and_payout_map(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    run_dir = store_root / "runs" / "run-1"
    predictions_dir = run_dir / "artifacts" / "predictions"
    predictions_dir.mkdir(parents=True)

    predictions_path = predictions_dir / "preds.parquet"
    meta_model_path = store_root / "datasets" / "v5.2" / "meta_model.parquet"
    meta_model_path.parent.mkdir(parents=True)

    predictions = pd.DataFrame(
        {
            "id": [f"id-{idx}" for idx in range(10)],
            "era": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "target_ender_20": [0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 0.75, 0.5, 0.25, 0.0],
            "prediction": [0.1, 0.2, 0.3, 0.45, 0.6, 0.6, 0.45, 0.3, 0.2, 0.1],
        }
    )
    predictions.to_parquet(predictions_path, index=False)

    meta = pd.DataFrame(
        {
            "id": [f"id-{idx}" for idx in range(10)],
            "era": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "numerai_meta_model": [0.2, 0.1, 0.4, 0.35, 0.5, 0.5, 0.25, 0.4, 0.15, 0.3],
        }
    )
    meta.set_index("id").to_parquet(meta_model_path, index=True)

    (run_dir / "run.json").write_text(
        """
        {
          "run_id": "run-1",
          "experiment_id": "exp-1",
          "data": {
            "target_col": "target_ender_20",
            "version": "v5.2"
          },
          "artifacts": {
            "predictions": "artifacts/predictions/preds.parquet",
            "score_provenance": "score_provenance.json"
          }
        }
        """.strip(),
        encoding="utf-8",
    )
    (run_dir / "score_provenance.json").write_text(
        f"""
        {{
          "columns": {{
            "prediction_cols": ["prediction"],
            "target_col": "target_ender_20",
            "id_col": "id",
            "era_col": "era",
            "meta_model_col": "numerai_meta_model"
          }},
          "sources": {{
            "predictions": {{"path": "{predictions_path}"}},
            "meta_model": {{"path": "{meta_model_path}"}}
          }},
          "joins": {{
            "predictions_rows": 10,
            "meta_overlap_rows": 10
          }}
        }}
        """.strip(),
        encoding="utf-8",
    )

    adapter = VizStoreAdapter(
        VizStoreConfig(
            store_root=store_root,
            repo_root=tmp_path
        )
    )

    per_era_corr = adapter.get_per_era_corr("run-1")
    assert per_era_corr is not None
    assert [row["era"] for row in per_era_corr] == [1, 2]
    assert all("corr20v2" in row for row in per_era_corr)

    payout_map = adapter.get_per_era_payout_map("run-1")
    assert payout_map is not None
    assert [row["era"] for row in payout_map] == [1, 2]
    for row in payout_map:
        assert "corr20v2" in row
        assert "mmc" in row
        assert "payout_estimate" in row
        expected = max(-0.05, min(0.05, (0.75 * float(row["corr20v2"])) + (2.25 * float(row["mmc"]))))
        assert row["payout_estimate"] == pytest.approx(expected)


def test_get_per_era_payout_map_non_ender_target_returns_none(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    run_dir = store_root / "runs" / "run-1"
    payout_dir = run_dir / "artifacts" / "predictions"
    payout_dir.mkdir(parents=True)

    (run_dir / "run.json").write_text(
        '{"run_id":"run-1","data":{"target_col":"target_cyrus_20"}}',
        encoding="utf-8",
    )
    (payout_dir / "val_per_era_payout_map.csv").write_text(
        "era,corr20v2,mmc,payout_estimate\n1,0.1,0.02,0.05\n",
        encoding="utf-8",
    )

    adapter = VizStoreAdapter(
        VizStoreConfig(
            store_root=store_root,
            repo_root=tmp_path,
        )
    )

    assert adapter.get_per_era_payout_map("run-1") is None


@pytest.mark.parametrize(
    ("method", "path", "body", "expected_status"),
    [
        ("POST", "/api/run-jobs", {"config_ids": ["configs/base.json"]}, 405),
        ("POST", "/api/run-jobs/job-1/cancel", {}, 404),
        ("POST", "/api/run-jobs/job-1/retry", {}, 404),
        (
            "POST",
            "/api/hpo-jobs",
            {"study_name": "study-a", "config_path": "configs/base.json", "n_trials": 2},
            404,
        ),
        (
            "POST",
            "/api/ensemble-jobs",
            {"run_ids": ["run-a", "run-b"], "method": "rank_avg", "metric": "corr20v2_sharpe"},
            404,
        ),
        ("PUT", "/api/experiments/exp-1/docs/EXPERIMENT.md", {"content": "hello"}, 405),
        ("PUT", "/api/runs/run-1/docs/RUN.md", {"content": "hello"}, 405),
        ("PUT", "/api/notes/content?path=foo.md", {"content": "hello"}, 405),
    ],
)
def test_removed_write_routes_are_unavailable(
    method: str,
    path: str,
    body: dict[str, object],
    expected_status: int,
    tmp_path: Path,
) -> None:
    store_root = tmp_path / ".numereng"
    store_root.mkdir(parents=True)
    client = _make_client(repo_root=tmp_path, store_root=store_root)

    response = client.request(method, path, json=body)
    assert response.status_code == expected_status


@pytest.mark.parametrize(
    "path",
    [
        "/api/run-jobs/bad$id/events",
        "/api/run-jobs/bad$id/logs",
        "/api/run-jobs/bad$id/samples",
        "/api/run-jobs/bad$id/stream",
    ],
)
def test_run_job_read_endpoints_invalid_job_id_returns_400(path: str, tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    store_root.mkdir(parents=True)
    client = _make_client(repo_root=tmp_path, store_root=store_root)

    response = client.get(path)
    assert response.status_code == 400
    assert "Invalid job_id" in response.json()["detail"]


def test_run_job_stream_stops_when_job_disappears() -> None:
    class _StubService:
        def __init__(self) -> None:
            self._calls = 0

        def get_run_job(self, job_id: str) -> dict[str, object] | None:
            self._calls += 1
            if self._calls == 1:
                return {"job_id": job_id, "status": "running"}
            return None

        def list_run_job_events(
            self,
            job_id: str,
            *,
            after_id: int | None,
            limit: int,
        ) -> list[dict[str, object]]:
            _ = job_id
            _ = after_id
            _ = limit
            return []

        def list_run_job_logs(
            self,
            job_id: str,
            *,
            after_id: int | None,
            limit: int,
            stream: str,
        ) -> list[dict[str, object]]:
            _ = job_id
            _ = after_id
            _ = limit
            _ = stream
            return []

        def list_run_job_samples(
            self,
            job_id: str,
            *,
            after_id: int | None,
            limit: int,
        ) -> list[dict[str, object]]:
            _ = job_id
            _ = after_id
            _ = limit
            return []

    service = cast(VizService, _StubService())
    app = FastAPI()
    app.include_router(create_router(service))
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/api/run-jobs/job-1/stream")
    assert response.status_code == 200
    assert response.text == ""


def test_notes_content_accepts_plus_in_path(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    notes_root = store_root / "notes"
    notes_root.mkdir(parents=True)
    target = notes_root / "signals-+-qc.md"
    target.write_text("# Signals\n", encoding="utf-8")

    client = _make_client(repo_root=tmp_path, store_root=store_root)

    tree_response = client.get("/api/notes/tree")
    assert tree_response.status_code == 200
    tree_payload = tree_response.json()
    assert tree_payload["sections"][0]["items"][0]["path"] == "signals-+-qc.md"

    content_response = client.get("/api/notes/content", params={"path": "signals-+-qc.md"})
    assert content_response.status_code == 200
    assert content_response.json() == {"content": "# Signals\n", "exists": True}


def test_get_notes_tree_missing_root_returns_empty_without_creating_directory(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    store_root.mkdir(parents=True)
    notes_root = store_root / "notes"
    assert not notes_root.exists()

    adapter = VizStoreAdapter(
        VizStoreConfig(
            store_root=store_root,
            repo_root=tmp_path,
        )
    )

    payload = adapter.get_notes_tree()
    assert payload == {"sections": []}
    assert not notes_root.exists()


def test_get_ensemble_endpoints_invalid_id_returns_400(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    store_root.mkdir(parents=True)
    client = _make_readonly_client(repo_root=tmp_path, store_root=store_root)

    response = client.get("/api/ensembles/bad$id")
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid ensemble_id: bad$id"

    response = client.get("/api/ensembles/bad$id/correlations")
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid ensemble_id: bad$id"

    response = client.get("/api/ensembles/bad$id/artifacts")
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid ensemble_id: bad$id"


def test_get_ensemble_artifacts_endpoint_returns_payload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    store_root.mkdir(parents=True)
    client = _make_readonly_client(repo_root=tmp_path, store_root=store_root)
    app = cast(FastAPI, client.app)
    service = cast(VizService, app.state.viz_service)

    monkeypatch.setattr(service, "get_ensemble", lambda ensemble_id: {"ensemble_id": ensemble_id})
    monkeypatch.setattr(
        service,
        "get_ensemble_artifacts",
        lambda ensemble_id: {
            "weights": [{"run_id": "run-a", "weight": 0.5, "rank": 0}],
            "component_metrics": [],
            "era_metrics": [],
            "regime_metrics": [],
            "lineage": {"ensemble_id": ensemble_id},
            "bootstrap_metrics": None,
            "heavy_component_predictions_available": False,
            "available_files": ["weights.csv", "lineage.json"],
        },
    )

    response = client.get("/api/ensembles/ens-1/artifacts")
    assert response.status_code == 200
    payload = response.json()
    assert payload["weights"][0]["run_id"] == "run-a"
    assert payload["lineage"]["ensemble_id"] == "ens-1"


def test_adapter_rejects_ensemble_artifacts_path_outside_store(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    store_root.mkdir(parents=True)
    db_path = store_root / "numereng.db"
    outside = tmp_path / "outside-artifacts"
    outside.mkdir(parents=True)

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE ensembles (
                ensemble_id TEXT PRIMARY KEY,
                experiment_id TEXT,
                name TEXT,
                method TEXT,
                target TEXT,
                metric TEXT,
                status TEXT,
                config_json TEXT,
                artifacts_path TEXT,
                created_at TEXT,
                updated_at TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO ensembles (
                ensemble_id, experiment_id, name, method, target, metric, status,
                config_json, artifacts_path, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "ens-1",
                "exp-1",
                "ens-1",
                "rank_avg",
                "target_ender_20",
                "corr20v2_sharpe",
                "completed",
                "{}",
                str(outside),
                "2026-02-22T00:00:00+00:00",
                "2026-02-22T00:00:00+00:00",
            ),
        )
        conn.commit()
    finally:
        conn.close()

    adapter = VizStoreAdapter(
        VizStoreConfig(
            store_root=store_root,
            repo_root=tmp_path
        )
    )

    assert adapter.get_ensemble_artifacts("ens-1") is None
