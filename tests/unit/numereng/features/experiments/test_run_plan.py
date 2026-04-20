from __future__ import annotations

import json
from pathlib import Path

import pytest

import numereng.features.experiments.run_plan as run_plan_module
from numereng.features.experiments import service as experiment_service
from numereng.features.training import TrainingRunResult


def _setup_experiment_with_plan(store_root: Path, *, experiment_id: str, config_names: list[str]) -> Path:
    experiment_service.create_experiment(store_root=store_root, experiment_id=experiment_id)
    experiment_dir = store_root / "experiments" / experiment_id
    for name in config_names:
        config_path = experiment_dir / "configs" / name
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("{}", encoding="utf-8")
    rows = [
        f"{index},r1,42,target_alpha,20,experiments/{experiment_id}/configs/{name},post_training_core"
        for index, name in enumerate(config_names, start=1)
    ]
    (experiment_dir / "run_plan.csv").write_text(
        "plan_index,round,seed,target,horizon,config_path,score_stage_default\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    return experiment_dir


def test_run_experiment_plan_executes_rows_scores_once_and_marks_complete(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_root = tmp_path / ".numereng"
    experiment_id = "2026-04-09_run-plan-exp"
    experiment_dir = _setup_experiment_with_plan(
        store_root,
        experiment_id=experiment_id,
        config_names=["r1_target_alpha_seed42.json", "r1_target_alpha_seed43.json"],
    )
    trained_configs: list[str] = []
    scored_rounds: list[tuple[str, str]] = []

    def fake_train_experiment(**kwargs: object) -> TrainingRunResult:
        config_path = Path(str(kwargs["config_path"]))
        trained_configs.append(config_path.name)
        return TrainingRunResult(
            run_id=f"run-{len(trained_configs)}",
            predictions_path=Path(f"/tmp/preds-{len(trained_configs)}.parquet"),
            results_path=Path(f"/tmp/results-{len(trained_configs)}.json"),
        )

    def fake_score_experiment_round(**kwargs: object) -> object:
        scored_rounds.append((str(kwargs["round"]), str(kwargs["stage"])))
        return object()

    monkeypatch.setattr(run_plan_module, "train_experiment", fake_train_experiment)
    monkeypatch.setattr(run_plan_module, "score_experiment_round", fake_score_experiment_round)
    monkeypatch.setattr(run_plan_module, "_repair_manifest_links_for_round", lambda **kwargs: None)

    result = run_plan_module.run_experiment_plan(store_root=store_root, experiment_id=experiment_id)

    assert trained_configs == ["r1_target_alpha_seed42.json", "r1_target_alpha_seed43.json"]
    assert scored_rounds == [("r1", "post_training_core")]
    assert result.phase == "complete"
    assert result.window.start_index == 1
    assert result.window.end_index == 2
    assert result.last_completed_row_index == 2
    assert result.completed_score_stages == ("r1:post_training_core",)

    state_payload = json.loads(
        run_plan_module.resolve_experiment_run_plan_state_path(
            store_root=store_root,
            experiment_id=experiment_id,
            start_index=1,
            end_index=2,
        ).read_text(encoding="utf-8")
    )
    assert state_payload["phase"] == "complete"
    assert state_payload["last_completed_row_index"] == 2
    assert state_payload["completed_score_stages"] == ["r1:post_training_core"]
    assert (experiment_dir / "configs" / "r1_target_alpha_seed42.json").is_file()


def test_run_experiment_plan_resume_noops_for_terminal_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_root = tmp_path / ".numereng"
    experiment_id = "2026-04-09_resume-exp"
    _setup_experiment_with_plan(
        store_root,
        experiment_id=experiment_id,
        config_names=["r1_target_alpha_seed42.json"],
    )
    state_path = run_plan_module.resolve_experiment_run_plan_state_path(
        store_root=store_root,
        experiment_id=experiment_id,
        start_index=1,
        end_index=1,
    )
    state_path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "experiment_id": experiment_id,
                "window": {"start_index": 1, "end_index": 1, "total_rows": 1},
                "phase": "complete",
                "requested_score_stage": "post_training_core",
                "completed_score_stages": ["r1:post_training_core"],
                "current_index": None,
                "current_round": None,
                "current_config_path": None,
                "current_run_id": None,
                "last_completed_row_index": 1,
                "supervisor_pid": None,
                "active_worker_pid": None,
                "last_successful_heartbeat_at": "2026-04-09T00:00:00+00:00",
                "failure_classifier": None,
                "retry_count": 0,
                "terminal_error": None,
                "updated_at": "2026-04-09T00:00:00+00:00",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        run_plan_module,
        "train_experiment",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("train_experiment should not run on terminal resume")),
    )

    result = run_plan_module.run_experiment_plan(store_root=store_root, experiment_id=experiment_id, resume=True)

    assert result.phase == "complete"
    assert result.last_completed_row_index == 1
    assert result.completed_score_stages == ("r1:post_training_core",)


def test_run_experiment_plan_retries_once_after_stale_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_root = tmp_path / ".numereng"
    experiment_id = "2026-04-09_retry-exp"
    _setup_experiment_with_plan(
        store_root,
        experiment_id=experiment_id,
        config_names=["r1_target_alpha_seed42.json"],
    )
    attempts: list[str] = []

    def fake_train_experiment(**kwargs: object) -> TrainingRunResult:
        attempts.append(Path(str(kwargs["config_path"])).name)
        if len(attempts) == 1:
            raise experiment_service.ExperimentValidationError("training_run_lock_exists:run-123:stale-owner")
        return TrainingRunResult(
            run_id="run-123",
            predictions_path=Path("/tmp/preds.parquet"),
            results_path=Path("/tmp/results.json"),
        )

    monkeypatch.setattr(run_plan_module, "train_experiment", fake_train_experiment)
    monkeypatch.setattr(run_plan_module, "score_experiment_round", lambda **kwargs: object())
    monkeypatch.setattr(run_plan_module, "_repair_manifest_links_for_round", lambda **kwargs: None)
    monkeypatch.setattr(run_plan_module, "_maybe_clear_stale_run_lock", lambda **kwargs: True)

    result = run_plan_module.run_experiment_plan(store_root=store_root, experiment_id=experiment_id)

    assert attempts == ["r1_target_alpha_seed42.json", "r1_target_alpha_seed42.json"]
    assert result.phase == "complete"
    assert result.retry_count == 0
    assert result.last_completed_row_index == 1
