from __future__ import annotations

import json
import sqlite3
import subprocess
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


def _write_run_artifacts(
    store_root: Path,
    run_id: str,
    *,
    status: str = "FINISHED",
    created_at: str = "2026-02-22T00:00:00+00:00",
    config_path: str | None = None,
    predictions_rel: str | None = "artifacts/predictions/pred.parquet",
    target_col: str | None = None,
    metrics: dict[str, object] | None = None,
    score_provenance: dict[str, object] | None = None,
) -> None:
    run_dir = store_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_manifest: dict[str, object] = {"run_id": run_id, "status": status, "created_at": created_at}
    if config_path is not None:
        run_manifest["config"] = {"path": config_path}
    if predictions_rel is not None:
        run_manifest["artifacts"] = {"predictions": predictions_rel}
        predictions_path = run_dir / predictions_rel
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_path.write_text("predictions", encoding="utf-8")
    if target_col is not None:
        run_manifest["data"] = {"target_col": target_col}
    (run_dir / "run.json").write_text(json.dumps(run_manifest))
    (run_dir / "metrics.json").write_text(json.dumps(metrics or {}))
    if score_provenance is not None:
        (run_dir / "score_provenance.json").write_text(json.dumps(score_provenance))


def _write_training_config(path: Path, *, post_training_scoring: str | None = None) -> None:
    payload: dict[str, object] = {
        "data": {"data_version": "v5.2", "dataset_variant": "non_downsampled"},
        "model": {"type": "LGBMRegressor", "params": {}},
        "training": {},
    }
    if post_training_scoring is not None:
        payload["training"] = {"post_training_scoring": post_training_scoring}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


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
    experiment_dir = manifest_path.parent
    assert manifest_path.is_file()
    assert (experiment_dir / "configs").is_dir()
    assert (experiment_dir / "run_scripts").is_dir()
    launch_all_py = experiment_dir / "run_scripts" / "launch_all.py"
    assert launch_all_py.is_file()
    assert (experiment_dir / "run_scripts" / "launch_all.sh").is_file()
    assert (experiment_dir / "run_scripts" / "launch_all.ps1").is_file()
    assert (experiment_dir / "run_plan.csv").read_text(encoding="utf-8") == (
        "plan_index,round,seed,target,horizon,config_path,score_stage_default\n"
    )
    experiment_doc = (experiment_dir / "EXPERIMENT.md").read_text(encoding="utf-8")
    assert 'EXPERIMENT_ID = "2026-02-22_test-exp"' in launch_all_py.read_text(encoding="utf-8")
    assert "## Round Log" in experiment_doc
    assert "run_scripts/launch_all.sh" in experiment_doc
    assert "uv run numereng experiment train --id 2026-02-22_test-exp" in experiment_doc
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


def test_create_experiment_generated_launcher_supports_external_store_root(tmp_path: Path) -> None:
    store_root = tmp_path / "external-workspace" / ".numereng"
    service_module.create_experiment(store_root=store_root, experiment_id="2026-02-22_test-exp")

    launcher_path = store_root / "experiments" / "2026-02-22_test-exp" / "run_scripts" / "launch_all.py"
    repo_root = Path(__file__).resolve().parents[5]
    result = subprocess.run(
        ["uv", "run", "python", str(launcher_path), "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--score-stage" in result.stdout


def test_train_experiment_appends_run_and_sets_active(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_root = tmp_path / ".numereng"
    service_module.create_experiment(store_root=store_root, experiment_id="2026-02-22_test-exp")

    config_path = tmp_path / "run.json"
    _write_training_config(config_path)

    def fake_run_training(
        *,
        config_path: str | Path,
        output_dir: str | Path | None,
        post_training_scoring: str | None,
        engine_mode: str | None,
        window_size_eras: int | None,
        embargo_eras: int | None,
        experiment_id: str | None,
        allow_round_batch_post_training_scoring: bool,
    ) -> TrainingRunResult:
        _ = (
            config_path,
            post_training_scoring,
            engine_mode,
            window_size_eras,
            embargo_eras,
        )
        assert output_dir is not None
        assert Path(output_dir) == store_root
        assert experiment_id == "2026-02-22_test-exp"
        assert allow_round_batch_post_training_scoring is True
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
    _write_training_config(config_path)

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
    _write_training_config(config_path)

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


def test_train_experiment_round_post_training_scoring_triggers_round_batch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_root = tmp_path / ".numereng"
    experiment_id = "2026-02-22_test-exp"
    service_module.create_experiment(store_root=store_root, experiment_id=experiment_id)
    config_path = store_root / "experiments" / experiment_id / "configs" / "r2_target_a_seed42.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(
            {
                "data": {"data_version": "v5.2", "dataset_variant": "non_downsampled"},
                "model": {"type": "LGBMRegressor", "params": {}},
                "training": {"post_training_scoring": "round_full"},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        service_module,
        "run_training",
        lambda **kwargs: TrainingRunResult(
            run_id="run-123",
            predictions_path=Path("/tmp/preds.parquet"),
            results_path=Path("/tmp/results.json"),
        ),
    )
    monkeypatch.setattr(service_module, "index_run", lambda **kwargs: None)
    scored_rounds: list[tuple[str, str]] = []

    def fake_score_round(*, store_root: str | Path, experiment_id: str, round: str, stage: str) -> object:
        _ = (store_root, experiment_id)
        scored_rounds.append((round, stage))
        return object()

    monkeypatch.setattr(service_module, "score_experiment_round", fake_score_round)

    result = service_module.train_experiment(
        store_root=store_root,
        experiment_id=experiment_id,
        config_path=config_path,
    )

    assert result.run_id == "run-123"
    assert scored_rounds == [("r2", "post_training_full")]


def test_train_experiment_round_post_training_scoring_requires_round_config(
    tmp_path: Path,
) -> None:
    store_root = tmp_path / ".numereng"
    experiment_id = "2026-02-22_test-exp"
    service_module.create_experiment(store_root=store_root, experiment_id=experiment_id)
    config_path = tmp_path / "not_a_round.json"
    config_path.write_text(
        json.dumps(
            {
                "data": {"data_version": "v5.2", "dataset_variant": "non_downsampled"},
                "model": {"type": "LGBMRegressor", "params": {}},
                "training": {"post_training_scoring": "round_core"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ExperimentValidationError, match="experiment_round_post_training_scoring_requires_round_config"):
        service_module.train_experiment(
            store_root=store_root,
            experiment_id=experiment_id,
            config_path=config_path,
        )


def test_train_experiment_round_batch_failure_is_recorded_and_training_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_root = tmp_path / ".numereng"
    experiment_id = "2026-02-22_test-exp"
    service_module.create_experiment(store_root=store_root, experiment_id=experiment_id)
    config_path = store_root / "experiments" / experiment_id / "configs" / "r1_target_a_seed42.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(
            {
                "data": {"data_version": "v5.2", "dataset_variant": "non_downsampled"},
                "model": {"type": "LGBMRegressor", "params": {}},
                "training": {"post_training_scoring": "round_core"},
            }
        ),
        encoding="utf-8",
    )
    run_dir = store_root / "runs" / "run-123"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "run-123",
                "status": "FINISHED",
                "training": {
                    "scoring": {
                        "policy": "round_core",
                        "status": "deferred",
                        "requested_stage": "post_training_core",
                        "refreshed_stages": [],
                        "reason": "experiment_round_batch_pending",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "results.json").write_text(
        json.dumps(
            {
                "metrics": {"status": "deferred", "reason": "experiment_round_batch_pending"},
                "training": {
                    "scoring": {
                        "policy": "round_core",
                        "status": "deferred",
                        "requested_stage": "post_training_core",
                        "refreshed_stages": [],
                        "reason": "experiment_round_batch_pending",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps({"status": "deferred", "reason": "experiment_round_batch_pending"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        service_module,
        "run_training",
        lambda **kwargs: TrainingRunResult(
            run_id="run-123",
            predictions_path=Path("/tmp/preds.parquet"),
            results_path=Path("/tmp/results.json"),
        ),
    )
    monkeypatch.setattr(service_module, "index_run", lambda **kwargs: None)
    monkeypatch.setattr(
        service_module,
        "score_experiment_round",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("batch boom")),
    )

    result = service_module.train_experiment(
        store_root=store_root,
        experiment_id=experiment_id,
        config_path=config_path,
    )

    assert result.run_id == "run-123"
    updated_run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    updated_results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert updated_run["training"]["scoring"]["status"] == "failed"
    assert updated_results["training"]["scoring"]["status"] == "failed"
    assert updated_results["metrics"]["status"] == "failed"
    run_log = (run_dir / "run.log").read_text(encoding="utf-8")
    assert "post_training_round_batch_failed" in run_log


def test_score_experiment_round_resolves_round_run_ids(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_root = tmp_path / ".numereng"
    experiment_id = "2026-02-22_test-exp"
    service_module.create_experiment(store_root=store_root, experiment_id=experiment_id)

    experiment_dir = store_root / "experiments" / experiment_id
    (experiment_dir / "run_plan.csv").write_text(
        "\n".join(
            [
                "target,horizon,seed,config_path",
                f"target_a,20d,42,{experiment_dir / 'configs' / 'r1_target_a_seed42.json'}",
                f"target_b,20d,43,{experiment_dir / 'configs' / 'r1_target_b_seed43.json'}",
                f"target_c,20d,44,{experiment_dir / 'configs' / 'r2_target_c_seed44.json'}",
            ]
        ),
        encoding="utf-8",
    )
    (experiment_dir / "configs" / "r1_target_a_seed42.json").write_text("{}", encoding="utf-8")
    (experiment_dir / "configs" / "r1_target_b_seed43.json").write_text("{}", encoding="utf-8")
    (experiment_dir / "configs" / "r2_target_c_seed44.json").write_text("{}", encoding="utf-8")

    manifest_path = experiment_dir / "experiment.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "active"
    manifest["runs"] = ["run-a", "run-b", "run-c"]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    for run_id, config_name in (
        ("run-a", "r1_target_a_seed42.json"),
        ("run-b", "r1_target_b_seed43.json"),
        ("run-c", "r2_target_c_seed44.json"),
    ):
        _write_run_artifacts(
            store_root,
            run_id,
            config_path=str(experiment_dir / "configs" / config_name),
        )

    scored_batches: list[dict[str, object]] = []

    def fake_score_run_batch(*, run_ids: tuple[str, ...], store_root: Path, stage: str) -> tuple[object, ...]:
        scored_batches.append(
            {
                "run_ids": run_ids,
                "store_root": store_root,
                "stage": stage,
            }
        )
        return ()

    monkeypatch.setattr(service_module, "score_run_batch", fake_score_run_batch)

    result = service_module.score_experiment_round(
        store_root=store_root,
        experiment_id=experiment_id,
        round="r1",
        stage="post_training_core",
    )

    assert result.experiment_id == experiment_id
    assert result.round == "r1"
    assert result.stage == "post_training_core"
    assert result.run_ids == ("run-a", "run-b")
    assert scored_batches == [
        {
            "run_ids": ("run-a", "run-b"),
            "store_root": store_root.resolve(),
            "stage": "post_training_core",
        }
    ]


def test_score_experiment_round_uses_explicit_run_plan_round(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_root = tmp_path / ".numereng"
    experiment_id = "2026-02-22_test-exp"
    service_module.create_experiment(store_root=store_root, experiment_id=experiment_id)

    experiment_dir = store_root / "experiments" / experiment_id
    config_path = experiment_dir / "configs" / "config_004.json"
    config_path.write_text("{}", encoding="utf-8")
    (experiment_dir / "run_plan.csv").write_text(
        "\n".join(
            [
                "plan_index,round,seed,target,horizon,config_path,score_stage_default",
                f"1,r004,,,,{config_path},post_training_full",
            ]
        ),
        encoding="utf-8",
    )
    manifest_path = experiment_dir / "experiment.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "active"
    manifest["runs"] = ["run-short"]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    _write_run_artifacts(store_root, "run-short", config_path=str(config_path))

    scored_batches: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        service_module,
        "score_run_batch",
        lambda *, run_ids, store_root, stage: scored_batches.append(tuple(run_ids)) or (),
    )

    result = service_module.score_experiment_round(
        store_root=store_root,
        experiment_id=experiment_id,
        round="r004",
        stage="post_training_full",
    )

    assert result.run_ids == ("run-short",)
    assert scored_batches == [("run-short",)]


def test_score_experiment_round_skips_failed_and_missing_prediction_runs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_root = tmp_path / ".numereng"
    experiment_id = "2026-02-22_test-exp"
    service_module.create_experiment(store_root=store_root, experiment_id=experiment_id)

    experiment_dir = store_root / "experiments" / experiment_id
    for name in ("r1_target_a_seed42.json", "r1_target_b_seed43.json", "r1_target_c_seed44.json"):
        (experiment_dir / "configs" / name).write_text("{}", encoding="utf-8")
    (experiment_dir / "run_plan.csv").write_text(
        "\n".join(
            [
                "target,horizon,seed,config_path",
                f"target_a,20d,42,{experiment_dir / 'configs' / 'r1_target_a_seed42.json'}",
                f"target_b,20d,43,{experiment_dir / 'configs' / 'r1_target_b_seed43.json'}",
                f"target_c,20d,44,{experiment_dir / 'configs' / 'r1_target_c_seed44.json'}",
            ]
        ),
        encoding="utf-8",
    )
    manifest_path = experiment_dir / "experiment.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "active"
    manifest["runs"] = ["run-failed", "run-missing", "run-finished"]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    _write_run_artifacts(
        store_root,
        "run-failed",
        status="FAILED",
        config_path=str(experiment_dir / "configs" / "r1_target_a_seed42.json"),
    )
    _write_run_artifacts(
        store_root,
        "run-missing",
        config_path=str(experiment_dir / "configs" / "r1_target_b_seed43.json"),
        predictions_rel=None,
    )
    _write_run_artifacts(
        store_root,
        "run-finished",
        config_path=str(experiment_dir / "configs" / "r1_target_c_seed44.json"),
    )

    scored_batches: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        service_module,
        "score_run_batch",
        lambda *, run_ids, store_root, stage: scored_batches.append(tuple(run_ids)) or (),
    )

    result = service_module.score_experiment_round(
        store_root=store_root,
        experiment_id=experiment_id,
        round="r1",
        stage="post_training_core",
    )

    assert result.run_ids == ("run-finished",)
    assert scored_batches == [("run-finished",)]


def test_score_experiment_round_prefers_latest_finished_run_for_duplicate_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_root = tmp_path / ".numereng"
    experiment_id = "2026-02-22_test-exp"
    service_module.create_experiment(store_root=store_root, experiment_id=experiment_id)

    experiment_dir = store_root / "experiments" / experiment_id
    config_path = experiment_dir / "configs" / "r1_target_a_seed42.json"
    config_path.write_text("{}", encoding="utf-8")
    (experiment_dir / "run_plan.csv").write_text(
        "\n".join(
            [
                "target,horizon,seed,config_path",
                f"target_a,20d,42,{config_path}",
            ]
        ),
        encoding="utf-8",
    )
    manifest_path = experiment_dir / "experiment.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "active"
    manifest["runs"] = ["run-old", "run-new"]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    _write_run_artifacts(
        store_root,
        "run-old",
        created_at="2026-02-22T00:00:00+00:00",
        config_path=str(config_path),
    )
    _write_run_artifacts(
        store_root,
        "run-new",
        created_at="2026-02-23T00:00:00+00:00",
        config_path=str(config_path),
    )

    scored_batches: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        service_module,
        "score_run_batch",
        lambda *, run_ids, store_root, stage: scored_batches.append(tuple(run_ids)) or (),
    )

    result = service_module.score_experiment_round(
        store_root=store_root,
        experiment_id=experiment_id,
        round="r1",
        stage="post_training_core",
    )

    assert result.run_ids == ("run-new",)
    assert scored_batches == [("run-new",)]


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


def test_pack_experiment_writes_markdown_summary(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    experiment_id = "2026-02-22_test-exp"
    service_module.create_experiment(store_root=store_root, experiment_id=experiment_id, name="Packed Experiment")

    exp_dir = store_root / "experiments" / experiment_id
    manifest_path = exp_dir / "experiment.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "complete"
    manifest["champion_run_id"] = "run-b"
    manifest["runs"] = ["run-c", "run-b", "run-a"]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (exp_dir / "EXPERIMENT.md").write_text("# Notes\n\nThis experiment converged.\n", encoding="utf-8")

    _write_run_artifacts(
        store_root,
        "run-c",
        created_at="2026-02-22T00:03:00+00:00",
        target_col="target_alpha_20",
        metrics={
            "corr": {"mean": 0.13, "sharpe": 1.5},
            "bmc": {"mean": 0.06},
            "bmc_last_200_eras": {"mean": 0.09},
            "cwmm": {"mean": 0.04},
        },
    )
    _write_run_artifacts(
        store_root,
        "run-b",
        created_at="2026-02-22T00:02:00+00:00",
        target_col="target_alpha_20",
        metrics={
            "corr": {"mean": 0.11, "sharpe": 1.2, "max_drawdown": -0.03},
            "mmc": {"mean": 0.02},
            "cwmm": {"mean": 0.03},
            "fnc": {"mean": 0.04},
            "bmc": {"mean": 0.05},
            "bmc_last_200_eras": {"mean": 0.06},
            "feature_exposure": {"mean": 0.07},
            "max_feature_exposure": {"mean": 0.08},
        },
        score_provenance={"joins": {"predictions_rows": 8, "meta_overlap_rows": 2}},
    )
    _write_run_artifacts(
        store_root,
        "run-a",
        created_at="2026-02-22T00:01:00+00:00",
        target_col="target_alpha_20",
        metrics={
            "corr": {"mean": 0.09, "sharpe": 0.9},
            "bmc": {"mean": 0.04},
            "bmc_last_200_eras": {"mean": 0.03},
        },
    )

    result = service_module.pack_experiment(store_root=store_root, experiment_id=experiment_id)

    assert result.run_count == 3
    assert result.output_path == exp_dir / "EXPERIMENT.pack.md"
    content = result.output_path.read_text(encoding="utf-8")
    assert "## Experiment Notes" in content
    assert "This experiment converged." in content
    assert "## Target Metrics Summary" in content
    assert "| target_alpha_20 |" in content
    assert "0.050000" in content
    assert "0.060000" in content
    assert "0.250000" in content
    assert "n/a" in content


def test_pack_experiment_supports_archived_experiments(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    experiment_id = "2026-02-22_test-exp"
    service_module.create_experiment(store_root=store_root, experiment_id=experiment_id)

    manifest_path = store_root / "experiments" / experiment_id / "experiment.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["runs"] = ["run-a"]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    _write_run_artifacts(
        store_root,
        "run-a",
        metrics={
            "corr": {"mean": 0.09, "sharpe": 0.9},
            "bmc": {"mean": 0.04},
            "bmc_last_200_eras": {"mean": 0.03},
        },
    )

    service_module.archive_experiment(store_root=store_root, experiment_id=experiment_id)
    result = service_module.pack_experiment(store_root=store_root, experiment_id=experiment_id)

    assert "_archive" in str(result.output_path)
    assert result.output_path.is_file()


def test_pack_experiment_requires_experiment_doc(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    experiment_id = "2026-02-22_test-exp"
    service_module.create_experiment(store_root=store_root, experiment_id=experiment_id)

    exp_doc = store_root / "experiments" / experiment_id / "EXPERIMENT.md"
    exp_doc.unlink()

    with pytest.raises(ExperimentValidationError, match="experiment_doc_missing"):
        service_module.pack_experiment(store_root=store_root, experiment_id=experiment_id)


@pytest.mark.parametrize(
    ("missing_name", "error_code"),
    (
        ("run.json", "experiment_pack_run_manifest_invalid"),
        ("metrics.json", "experiment_pack_run_metrics_invalid"),
    ),
)
def test_pack_experiment_requires_run_artifacts(
    tmp_path: Path,
    missing_name: str,
    error_code: str,
) -> None:
    store_root = tmp_path / ".numereng"
    experiment_id = "2026-02-22_test-exp"
    service_module.create_experiment(store_root=store_root, experiment_id=experiment_id)

    manifest_path = store_root / "experiments" / experiment_id / "experiment.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["runs"] = ["run-a"]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    _write_run_artifacts(
        store_root,
        "run-a",
        metrics={
            "corr": {"mean": 0.09, "sharpe": 0.9},
            "bmc": {"mean": 0.04},
            "bmc_last_200_eras": {"mean": 0.03},
        },
    )
    (store_root / "runs" / "run-a" / missing_name).unlink()

    with pytest.raises(ExperimentValidationError, match=error_code):
        service_module.pack_experiment(store_root=store_root, experiment_id=experiment_id)


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


def test_archive_and_unarchive_experiment_round_trip(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    experiment_id = "2026-02-22_test-exp"
    service_module.create_experiment(store_root=store_root, experiment_id=experiment_id, name="Alpha")
    exp_dir = store_root / "experiments" / experiment_id
    (exp_dir / "configs" / "base.json").write_text("{}")
    (exp_dir / "run_plan.csv").write_text("target,horizon,seed,config_path\n")
    manifest_path = exp_dir / "experiment.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "complete"
    manifest["metadata"] = {"note": "keep-me"}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    service_module._index_experiment_manifest(store_root, manifest)

    archived = service_module.archive_experiment(store_root=store_root, experiment_id=experiment_id)
    archived_dir = store_root / "experiments" / "_archive" / experiment_id
    assert archived.archived is True
    assert archived.status == "archived"
    assert archived_dir.is_dir()
    assert not exp_dir.exists()

    archived_manifest = json.loads((archived_dir / "experiment.json").read_text())
    assert archived_manifest["status"] == "archived"
    assert archived_manifest["metadata"]["pre_archive_status"] == "complete"
    assert (archived_dir / "configs" / "base.json").is_file()
    assert (archived_dir / "run_plan.csv").is_file()

    restored = service_module.unarchive_experiment(store_root=store_root, experiment_id=experiment_id)
    assert restored.archived is False
    assert restored.status == "complete"
    assert exp_dir.is_dir()
    assert not archived_dir.exists()

    restored_manifest = json.loads((exp_dir / "experiment.json").read_text())
    assert restored_manifest["status"] == "complete"
    assert restored_manifest["metadata"] == {"note": "keep-me"}

    with sqlite3.connect(store_root / "numereng.db") as conn:
        row = conn.execute(
            "SELECT status FROM experiments WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
    assert row == ("complete",)


def test_archive_destination_conflict_does_not_mutate_live_manifest(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    experiment_id = "2026-02-22_test-exp"
    service_module.create_experiment(store_root=store_root, experiment_id=experiment_id)
    live_manifest_path = store_root / "experiments" / experiment_id / "experiment.json"
    archive_dir = store_root / "experiments" / "_archive" / experiment_id
    archive_dir.mkdir(parents=True)
    (archive_dir / "placeholder.txt").write_text("occupied")

    with pytest.raises(ExperimentValidationError, match="experiment_archive_destination_exists"):
        service_module.archive_experiment(store_root=store_root, experiment_id=experiment_id)

    live_manifest = json.loads(live_manifest_path.read_text())
    assert live_manifest["status"] == "draft"
    assert live_manifest["metadata"] == {}


def test_unarchive_destination_conflict_does_not_mutate_archived_manifest(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    experiment_id = "2026-02-22_test-exp"
    service_module.create_experiment(store_root=store_root, experiment_id=experiment_id)
    service_module.archive_experiment(store_root=store_root, experiment_id=experiment_id)
    archived_manifest_path = store_root / "experiments" / "_archive" / experiment_id / "experiment.json"
    live_dir = store_root / "experiments" / experiment_id
    live_dir.mkdir(parents=True)
    (live_dir / "placeholder.txt").write_text("occupied")

    with pytest.raises(ExperimentValidationError, match="experiment_unarchive_destination_exists"):
        service_module.unarchive_experiment(store_root=store_root, experiment_id=experiment_id)

    archived_manifest = json.loads(archived_manifest_path.read_text())
    assert archived_manifest["status"] == "archived"
    assert archived_manifest["metadata"]["pre_archive_status"] == "draft"


def test_train_and_promote_reject_archived_experiment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_root = tmp_path / ".numereng"
    experiment_id = "2026-02-22_test-exp"
    service_module.create_experiment(store_root=store_root, experiment_id=experiment_id)
    config_path = tmp_path / "run.json"
    _write_training_config(config_path)
    service_module.archive_experiment(store_root=store_root, experiment_id=experiment_id)

    monkeypatch.setattr(
        service_module,
        "run_training",
        lambda **kwargs: TrainingRunResult(
            run_id="run-123",
            predictions_path=Path("/tmp/preds.parquet"),
            results_path=Path("/tmp/results.json"),
        ),
    )

    with pytest.raises(ExperimentValidationError, match="experiment_archived_read_only"):
        service_module.train_experiment(
            store_root=store_root,
            experiment_id=experiment_id,
            config_path=config_path,
        )

    with pytest.raises(ExperimentValidationError, match="experiment_archived_read_only"):
        service_module.promote_experiment(
            store_root=store_root,
            experiment_id=experiment_id,
        )


def test_list_experiments_excludes_archived_by_default(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    service_module.create_experiment(store_root=store_root, experiment_id="2026-02-22_alpha")
    service_module.create_experiment(store_root=store_root, experiment_id="2026-02-22_beta")
    service_module.archive_experiment(store_root=store_root, experiment_id="2026-02-22_beta")

    default_listing = service_module.list_experiments(store_root=store_root)
    archived_listing = service_module.list_experiments(store_root=store_root, status="archived")

    assert [item.experiment_id for item in default_listing] == ["2026-02-22_alpha"]
    assert [item.experiment_id for item in archived_listing] == ["2026-02-22_beta"]


def test_list_experiments_ignores_bad_archived_manifest_by_default(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    service_module.create_experiment(store_root=store_root, experiment_id="2026-02-22_alpha")
    archived_dir = store_root / "experiments" / "_archive" / "2026-02-22_beta"
    archived_dir.mkdir(parents=True)
    (archived_dir / "experiment.json").write_text("{not-json")

    listing = service_module.list_experiments(store_root=store_root)

    assert [item.experiment_id for item in listing] == ["2026-02-22_alpha"]
