from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import numereng.features.experiments.service as service_module
from numereng.features.experiments import (
    ExperimentAlreadyExistsError,
    ExperimentError,
    ExperimentNotFoundError,
    ExperimentRunNotFoundError,
    ExperimentValidationError,
)
from numereng.features.store import StoreError
from numereng.features.training import TrainingRunResult


def test_create_experiment_writes_manifest_and_indexes_db(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"

    record = service_module.create_experiment(
        store_root=store_root,
        experiment_id="2026-02-22_test-exp",
        name="Test Experiment",
        hypothesis="Track metrics",
        tags=["quick", "baseline"],
    )

    manifest_path = store_root / "experiments" / "2026-02-22_test-exp" / "experiment.json"
    assert manifest_path.is_file()
    assert record.experiment_id == "2026-02-22_test-exp"
    assert record.status == "draft"
    assert record.tags == ("quick", "baseline")

    with sqlite3.connect(store_root / "numereng.db") as conn:
        row = conn.execute(
            "SELECT name, status FROM experiments WHERE experiment_id = ?",
            ("2026-02-22_test-exp",),
        ).fetchone()
    assert row is not None
    assert row[0] == "Test Experiment"
    assert row[1] == "draft"


def test_create_experiment_requires_id_format(tmp_path: Path) -> None:
    with pytest.raises(ExperimentValidationError, match="experiment_id_format_invalid"):
        service_module.create_experiment(
            store_root=tmp_path / ".numereng",
            experiment_id="bad-id",
        )


def test_create_experiment_fails_when_existing(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    service_module.create_experiment(store_root=store_root, experiment_id="2026-02-22_test-exp")

    with pytest.raises(ExperimentAlreadyExistsError, match="experiment_already_exists"):
        service_module.create_experiment(store_root=store_root, experiment_id="2026-02-22_test-exp")


def test_train_experiment_appends_run_and_sets_active(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_root = tmp_path / ".numereng"
    service_module.create_experiment(store_root=store_root, experiment_id="2026-02-22_test-exp")

    config_path = tmp_path / "run.json"
    config_path.write_text("{}")

    def fake_run_training(
        *,
        config_path: str | Path,
        output_dir: str | Path | None,
        engine_mode: str | None,
        window_size_eras: int | None,
        embargo_eras: int | None,
        experiment_id: str | None,
    ) -> TrainingRunResult:
        _ = (
            config_path,
            engine_mode,
            window_size_eras,
            embargo_eras,
        )
        assert output_dir is not None
        assert Path(output_dir) == store_root
        assert experiment_id == "2026-02-22_test-exp"
        return TrainingRunResult(
            run_id="run-123",
            predictions_path=Path("/tmp/preds.parquet"),
            results_path=Path("/tmp/results.json"),
        )

    monkeypatch.setattr(service_module, "run_training", fake_run_training)
    indexed: dict[str, object] = {}

    def fake_index_run(*, store_root: str | Path, run_id: str) -> None:
        indexed["store_root"] = Path(store_root)
        indexed["run_id"] = run_id

    monkeypatch.setattr(service_module, "index_run", fake_index_run)

    result = service_module.train_experiment(
        store_root=store_root,
        experiment_id="2026-02-22_test-exp",
        config_path=config_path,
    )

    assert result.experiment_id == "2026-02-22_test-exp"
    assert result.run_id == "run-123"

    manifest = json.loads((store_root / "experiments" / "2026-02-22_test-exp" / "experiment.json").read_text())
    assert manifest["status"] == "active"
    assert manifest["runs"] == ["run-123"]
    assert indexed["store_root"] == store_root
    assert indexed["run_id"] == "run-123"


def test_train_experiment_rejects_mismatched_output_dir(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    service_module.create_experiment(store_root=store_root, experiment_id="2026-02-22_test-exp")

    config_path = tmp_path / "run.json"
    config_path.write_text("{}")

    with pytest.raises(ExperimentValidationError, match="experiment_output_dir_must_match_store_root"):
        service_module.train_experiment(
            store_root=store_root,
            experiment_id="2026-02-22_test-exp",
            config_path=config_path,
            output_dir=tmp_path / "other-root",
        )


def test_train_experiment_wraps_index_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    service_module.create_experiment(store_root=store_root, experiment_id="2026-02-22_test-exp")
    config_path = tmp_path / "run.json"
    config_path.write_text("{}")

    monkeypatch.setattr(
        service_module,
        "run_training",
        lambda **kwargs: TrainingRunResult(
            run_id="run-123",
            predictions_path=Path("/tmp/preds.parquet"),
            results_path=Path("/tmp/results.json"),
        ),
    )
    monkeypatch.setattr(
        service_module,
        "index_run",
        lambda **kwargs: (_ for _ in ()).throw(StoreError("store_run_not_found:run-123")),
    )

    with pytest.raises(ExperimentError, match="experiment_run_index_failed:run-123"):
        service_module.train_experiment(
            store_root=store_root,
            experiment_id="2026-02-22_test-exp",
            config_path=config_path,
        )


def test_promote_experiment_selects_best_metric(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    experiment_id = "2026-02-22_test-exp"
    service_module.create_experiment(store_root=store_root, experiment_id=experiment_id)

    exp_manifest_path = store_root / "experiments" / experiment_id / "experiment.json"
    manifest = json.loads(exp_manifest_path.read_text())
    manifest["runs"] = ["run-a", "run-b"]
    exp_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    run_a = store_root / "runs" / "run-a"
    run_b = store_root / "runs" / "run-b"
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)
    (run_a / "metrics.json").write_text(json.dumps({"bmc_last_200_eras": {"mean": 0.10}}))
    (run_b / "metrics.json").write_text(json.dumps({"bmc_last_200_eras": {"mean": 0.15}}))

    result = service_module.promote_experiment(
        store_root=store_root,
        experiment_id=experiment_id,
        metric="bmc_last_200_eras.mean",
    )

    assert result.auto_selected is True
    assert result.champion_run_id == "run-b"
    assert result.metric_value == 0.15


def test_promote_experiment_validates_explicit_run(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    experiment_id = "2026-02-22_test-exp"
    service_module.create_experiment(store_root=store_root, experiment_id=experiment_id)

    exp_manifest_path = store_root / "experiments" / experiment_id / "experiment.json"
    manifest = json.loads(exp_manifest_path.read_text())
    manifest["runs"] = ["run-a"]
    exp_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    with pytest.raises(ExperimentRunNotFoundError, match="experiment_run_not_found"):
        service_module.promote_experiment(
            store_root=store_root,
            experiment_id=experiment_id,
            run_id="run-missing",
        )


def test_report_experiment_ranks_rows(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    experiment_id = "2026-02-22_test-exp"
    service_module.create_experiment(store_root=store_root, experiment_id=experiment_id)

    exp_manifest_path = store_root / "experiments" / experiment_id / "experiment.json"
    manifest = json.loads(exp_manifest_path.read_text())
    manifest["runs"] = ["run-a", "run-b"]
    manifest["champion_run_id"] = "run-b"
    exp_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    run_a = store_root / "runs" / "run-a"
    run_b = store_root / "runs" / "run-b"
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)
    (run_a / "metrics.json").write_text(
        json.dumps(
            {
                "corr": {"mean": 0.10},
                "mmc": {"mean": 0.02},
                "cwmm": {"mean": 0.01},
                "bmc": {"mean": 0.09},
                "bmc_last_200_eras": {"mean": 0.09},
            }
        )
    )
    (run_b / "metrics.json").write_text(
        json.dumps(
            {
                "corr": {"mean": 0.11},
                "mmc": {"mean": 0.03},
                "cwmm": {"mean": 0.02},
                "bmc": {"mean": 0.12},
                "bmc_last_200_eras": {"mean": 0.12},
            }
        )
    )

    report = service_module.report_experiment(
        store_root=store_root,
        experiment_id=experiment_id,
        metric="bmc_last_200_eras.mean",
        limit=10,
    )
    assert report.total_runs == 2
    assert report.rows[0].run_id == "run-b"
    assert report.rows[0].is_champion is True


def test_get_experiment_not_found_raises(tmp_path: Path) -> None:
    with pytest.raises(ExperimentNotFoundError, match="experiment_not_found"):
        service_module.get_experiment(
            store_root=tmp_path / ".numereng",
            experiment_id="2026-02-22_missing",
        )


def test_list_and_get_experiment_success(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    service_module.create_experiment(
        store_root=store_root,
        experiment_id="2026-02-22_alpha",
        name="Alpha",
    )
    service_module.create_experiment(
        store_root=store_root,
        experiment_id="2026-02-22_beta",
        name="Beta",
    )

    listing = service_module.list_experiments(store_root=store_root)
    ids = [item.experiment_id for item in listing]
    assert set(ids) == {"2026-02-22_alpha", "2026-02-22_beta"}

    record = service_module.get_experiment(store_root=store_root, experiment_id="2026-02-22_alpha")
    assert record.experiment_id == "2026-02-22_alpha"
    assert record.name == "Alpha"
