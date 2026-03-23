from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from numereng.features.agentic_research.codex_exec import (
    default_codex_command,
    ensure_learner_codex_home,
    learner_codex_home,
    run_codex_planner,
)
from numereng.features.agentic_research.contracts import (
    CodexConfigPayload,
    CodexDecision,
    CodexPlannerExecution,
    ResearchBestRun,
    ResearchLineageLink,
    ResearchLineageState,
    ResearchPathState,
    ResearchPhaseState,
    ResearchProgramState,
)
from numereng.features.agentic_research.planner import (
    force_action_for_path,
    materialize_config_payload,
    select_reference_configs,
    should_pivot_path,
    update_path_after_round,
    validate_decision,
)
from numereng.features.agentic_research.service import init_research, run_research
from numereng.features.agentic_research.state import (
    load_lineage_state,
    load_program_state,
    save_lineage_state,
    save_program_state,
)
from numereng.features.agentic_research.strategy import get_strategy_definition
from numereng.features.experiments import (
    ExperimentReport,
    ExperimentReportRow,
    ExperimentScoreRoundResult,
    ExperimentTrainResult,
    create_experiment,
)


def _valid_training_config() -> dict[str, object]:
    return {
        "data": {
            "data_version": "v5.2",
            "dataset_variant": "non_downsampled",
            "dataset_scope": "train_plus_validation",
            "feature_set": "small",
            "target_col": "target_ender_20",
            "target_horizon": "20d",
            "era_col": "era",
            "id_col": "id",
            "benchmark_source": {
                "source": "active",
                "predictions_path": None,
                "pred_col": "prediction",
                "name": None,
            },
            "meta_model_data_path": None,
            "meta_model_col": "numerai_meta_model",
            "embargo_eras": None,
            "baselines_dir": None,
            "loading": {
                "mode": "materialized",
                "scoring_mode": "materialized",
                "era_chunk_size": 64,
                "include_feature_neutral_metrics": True,
            },
        },
        "preprocessing": {
            "nan_missing_all_twos": False,
            "missing_value": 2.0,
        },
        "model": {
            "type": "LGBMRegressor",
            "params": {
                "n_estimators": 10,
                "learning_rate": 0.01,
                "max_depth": 6,
                "num_leaves": 64,
                "colsample_bytree": 0.1,
                "random_state": 42,
            },
            "x_groups": ["features"],
            "data_needed": None,
            "module_path": None,
            "target_transform": None,
            "benchmark": None,
            "baseline": None,
        },
        "training": {
            "engine": {
                "profile": "purged_walk_forward",
                "mode": None,
                "window_size_eras": None,
                "embargo_eras": None,
            },
            "resources": {
                "parallel_folds": 1,
                "parallel_backend": "joblib",
                "memmap_enabled": True,
                "max_threads_per_worker": "default",
                "sklearn_working_memory_mib": None,
            },
            "cache": {
                "mode": "deterministic",
                "cache_fold_specs": True,
                "cache_features": True,
                "cache_labels": True,
                "cache_fold_matrices": False,
            },
        },
        "output": {
            "output_dir": None,
            "baselines_dir": None,
            "predictions_name": "val_predictions_scout",
            "results_name": None,
        },
    }


def _decision() -> CodexDecision:
    return CodexDecision(
        experiment_question="Which target variant should win this round?",
        winner_criteria="Highest bmc_last_200_eras.mean with bmc.mean tie-break.",
        decision_rationale="Keep the sweep narrow and benchmark-relative.",
        next_action="continue",
        path_hypothesis="Tree models around Ender20 remain the best root path.",
        path_slug="ender20-tree-path",
        configs=[
            CodexConfigPayload(filename="base.json", rationale="Base", overrides={}),
            CodexConfigPayload(
                filename="lr.json",
                rationale="Lower LR",
                overrides={"model": {"params": {"learning_rate": 0.005}}},
            ),
            CodexConfigPayload(
                filename="leaves.json",
                rationale="More leaves",
                overrides={"model": {"params": {"num_leaves": 96}}},
            ),
            CodexConfigPayload(
                filename="features.json",
                rationale="Feature variant",
                overrides={"data": {"feature_set": "medium"}},
            ),
        ],
        phase_action=None,
        phase_transition_rationale=None,
    )


def _planner_execution(decision: CodexDecision | None = None) -> CodexPlannerExecution:
    resolved = decision or _decision()
    return CodexPlannerExecution(
        decision=resolved,
        attempts=[],
        stdout_jsonl='{"type":"thread.started","thread_id":"thread_123"}\n',
        stderr_text="",
        last_message={
            "experiment_question": resolved.experiment_question,
            "winner_criteria": resolved.winner_criteria,
            "decision_rationale": resolved.decision_rationale,
            "next_action": resolved.next_action,
            "path_hypothesis": resolved.path_hypothesis,
            "path_slug": resolved.path_slug,
            "phase_action": resolved.phase_action,
            "phase_transition_rationale": resolved.phase_transition_rationale,
            "configs": [
                {
                    "filename": item.filename,
                    "rationale": item.rationale,
                    "overrides": item.overrides,
                }
                for item in resolved.configs
            ],
        },
    )


def test_state_roundtrip(tmp_path: Path) -> None:
    program = ResearchProgramState(
        root_experiment_id="2026-03-20_root-exp",
        program_experiment_id="2026-03-20_root-exp",
        strategy="numerai-experiment-loop",
        strategy_description="Numerai Experiment Loop",
        status="initialized",
        active_path_id="p00",
        active_experiment_id="2026-03-20_root-exp",
        next_round_number=1,
        total_rounds_completed=0,
        total_paths_created=1,
        improvement_threshold=0.0002,
        scoring_stage="post_training_full",
        codex_command=default_codex_command(),
        last_checkpoint="initialized",
        stop_reason=None,
        current_round=None,
        current_phase=None,
        best_overall=ResearchBestRun(),
        paths=[
            ResearchPathState(
                path_id="p00",
                experiment_id="2026-03-20_root-exp",
                parent_experiment_id=None,
                generation=0,
                hypothesis="root",
                status="active",
                pivot_reason=None,
                source_round=None,
                rounds_completed=0,
                plateau_streak=0,
                scale_confirmation_used=False,
                needs_scale_confirmation=False,
                best_run_id=None,
                created_at="2026-03-20T00:00:00+00:00",
                updated_at="2026-03-20T00:00:00+00:00",
            )
        ],
        created_at="2026-03-20T00:00:00+00:00",
        updated_at="2026-03-20T00:00:00+00:00",
    )
    lineage = ResearchLineageState(
        root_experiment_id="2026-03-20_root-exp",
        program_experiment_id="2026-03-20_root-exp",
        active_path_id="p00",
        links=[
            ResearchLineageLink(
                path_id="p00",
                experiment_id="2026-03-20_root-exp",
                parent_experiment_id=None,
                generation=0,
                source_round=None,
                pivot_reason=None,
                created_at="2026-03-20T00:00:00+00:00",
            )
        ],
    )
    program_state_path = tmp_path / "program.json"
    lineage_state_path = tmp_path / "lineage.json"

    save_program_state(program_state_path, program)
    save_lineage_state(lineage_state_path, lineage)

    loaded_program = load_program_state(program_state_path)
    loaded_lineage = load_lineage_state(lineage_state_path)

    assert loaded_program.root_experiment_id == "2026-03-20_root-exp"
    assert loaded_program.strategy == "numerai-experiment-loop"
    assert loaded_program.paths[0].path_id == "p00"
    assert loaded_program.codex_command == default_codex_command()
    assert loaded_lineage.links[0].experiment_id == "2026-03-20_root-exp"


def test_strategy_registry_loads_both_profiles() -> None:
    numerai = get_strategy_definition("numerai-experiment-loop")
    kaggle = get_strategy_definition("kaggle-gm-loop")

    assert numerai.phase_aware is False
    assert kaggle.phase_aware is True
    assert kaggle.phases[0].phase_id == "phase1_eda_baseline"
    assert kaggle.schema_path.name == "planner_output.schema.json"


def test_load_program_state_requires_strategy(tmp_path: Path) -> None:
    path = tmp_path / "program.json"
    path.write_text(
        json.dumps(
            {
                "root_experiment_id": "2026-03-20_root-exp",
                "program_experiment_id": "2026-03-20_root-exp",
                "status": "initialized",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="agentic_research_program_strategy_missing"):
        load_program_state(path)


def test_ensure_learner_codex_home_writes_minimal_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_home = tmp_path / "source_codex"
    source_home.mkdir(parents=True)
    (source_home / "auth.json").write_text('{"token":"abc"}', encoding="utf-8")
    learner_home = tmp_path / "learner_codex"

    monkeypatch.setattr("numereng.features.agentic_research.codex_exec._USER_CODEX_HOME", source_home)
    monkeypatch.setattr("numereng.features.agentic_research.codex_exec._LEARNER_CODEX_HOME", learner_home)

    ensured = ensure_learner_codex_home()

    assert ensured == learner_home
    config_text = (learner_home / "config.toml").read_text(encoding="utf-8")
    assert 'model = "gpt-5.4"' in config_text
    assert 'model_reasoning_effort = "low"' in config_text
    assert "shell_tool = false" in config_text
    assert (learner_home / "auth.json").read_text(encoding="utf-8") == '{"token":"abc"}'


def test_run_codex_planner_parses_jsonl_and_retries_on_invalid_last_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"count": 0}

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls["count"] += 1
        command = list(args[0]) if args else []
        output_index = command.index("-o")
        output_path = Path(command[output_index + 1])
        if calls["count"] == 1:
            output_path.write_text("not-json", encoding="utf-8")
            stdout = json.dumps({"type": "thread.started", "thread_id": "thread_1"}) + "\n"
            return subprocess.CompletedProcess(args=command, returncode=0, stdout=stdout, stderr="warning")
        output_path.write_text(
            json.dumps(
                {
                    "experiment_question": "q",
                    "winner_criteria": "w",
                    "decision_rationale": "r",
                    "next_action": "continue",
                    "path_hypothesis": "h",
                    "path_slug": "slug",
                    "configs": [
                        {"filename": "base.json", "rationale": "Base", "overrides": []},
                        {
                            "filename": "b.json",
                            "rationale": "b",
                            "overrides": [
                                {
                                    "path": "model.params.learning_rate",
                                    "value_json": "0.005",
                                }
                            ],
                        },
                        {
                            "filename": "c.json",
                            "rationale": "c",
                            "overrides": [
                                {
                                    "path": "data.feature_set",
                                    "value_json": '"medium"',
                                }
                            ],
                        },
                        {
                            "filename": "d.json",
                            "rationale": "d",
                            "overrides": [
                                {
                                    "path": "model.params.num_leaves",
                                    "value_json": "96",
                                }
                            ],
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        stdout = "\n".join(
            [
                json.dumps({"type": "thread.started", "thread_id": "thread_2"}),
                json.dumps(
                    {
                        "type": "response.completed",
                        "response": {
                            "usage": {
                                "input_tokens": 123,
                                "cached_input_tokens": 45,
                                "output_tokens": 67,
                            }
                        },
                    }
                ),
            ]
        )
        return subprocess.CompletedProcess(args=command, returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        "numereng.features.agentic_research.codex_exec.ensure_learner_codex_home",
        lambda: learner_codex_home(),
    )

    execution = run_codex_planner(
        prompt="hello",
        command=default_codex_command(),
        schema_path=get_strategy_definition("numerai-experiment-loop").schema_path,
        artifact_dir=tmp_path,
    )

    assert calls["count"] == 2
    assert execution.decision.next_action == "continue"
    assert len(execution.decision.configs) == 4
    assert len(execution.attempts) == 2
    assert execution.attempts[-1].thread_id == "thread_2"
    assert execution.attempts[-1].input_tokens == 123
    assert execution.attempts[-1].cached_input_tokens == 45
    assert execution.attempts[-1].output_tokens == 67


def test_validate_decision_requires_4_to_5_configs() -> None:
    with pytest.raises(ValueError, match="agentic_research_codex_config_count_invalid"):
        validate_decision(
            CodexDecision(
                experiment_question="q",
                winner_criteria="w",
                decision_rationale="r",
                next_action="continue",
                path_hypothesis="h",
                path_slug="slug",
                configs=[],
            ),
            strategy=get_strategy_definition("numerai-experiment-loop"),
            current_phase=None,
        )


def test_validate_decision_requires_phase_action_for_phase_aware_strategy() -> None:
    with pytest.raises(ValueError, match="agentic_research_codex_phase_action_invalid"):
        validate_decision(
            CodexDecision(
                experiment_question="q",
                winner_criteria="w",
                decision_rationale="r",
                next_action="continue",
                path_hypothesis="h",
                path_slug="slug",
                configs=_decision().configs,
            ),
            strategy=get_strategy_definition("kaggle-gm-loop"),
            current_phase=ResearchPhaseState(
                phase_id="phase1_eda_baseline",
                phase_title="Phase 1 - EDA & Baseline",
                status="active",
                round_count=0,
                transition_rationale=None,
                started_at="2026-03-20T00:00:00+00:00",
                updated_at="2026-03-20T00:00:00+00:00",
            ),
        )


def test_materialize_config_payload_rejects_disallowed_paths() -> None:
    with pytest.raises(ValueError, match="agentic_research_override_path_not_allowed:output.output_dir"):
        materialize_config_payload(
            base_config=_valid_training_config(),
            overrides={"output": {"output_dir": "/tmp/out"}},
        )


def test_select_reference_configs_falls_back_to_validated_default(tmp_path: Path) -> None:
    base_config, source, examples = select_reference_configs(config_dirs=[tmp_path / "missing"])

    assert source == "built_in_fallback"
    assert base_config["data"]["data_version"] == "v5.2"
    assert len(examples) == 2
    assert examples[1]["materialized_config"]["model"]["params"]["learning_rate"] == 0.005


def test_plateau_logic_requires_scale_confirmation_before_pivot() -> None:
    path = ResearchPathState(
        path_id="p00",
        experiment_id="2026-03-20_root-exp",
        parent_experiment_id=None,
        generation=0,
        hypothesis="root",
        status="active",
        pivot_reason=None,
        source_round=None,
        rounds_completed=0,
        plateau_streak=1,
        scale_confirmation_used=False,
        needs_scale_confirmation=False,
        best_run_id=None,
        created_at="2026-03-20T00:00:00+00:00",
        updated_at="2026-03-20T00:00:00+00:00",
    )

    updated = update_path_after_round(path=path, round_best=None, improved=False)

    assert updated.plateau_streak == 2
    assert force_action_for_path(updated) == "scale"
    assert should_pivot_path(updated) is False

    after_scale = replace(updated, scale_confirmation_used=True, needs_scale_confirmation=False)
    assert should_pivot_path(after_scale) is True


def test_run_research_executes_one_full_round(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_root = tmp_path / ".numereng"
    root_exp = create_experiment(store_root=store_root, experiment_id="2026-03-20_root-exp")
    init_research(
        store_root=store_root,
        experiment_id=root_exp.experiment_id,
        strategy="numerai-experiment-loop",
    )

    run_ids: list[str] = []

    monkeypatch.setattr(
        "numereng.features.agentic_research.service.run_codex_planner",
        lambda **_: _planner_execution(),
    )

    def fake_train_experiment(
        *,
        store_root: str | Path,
        experiment_id: str,
        config_path: str | Path,
    ) -> ExperimentTrainResult:
        _ = (store_root, experiment_id)
        run_id = f"run-{len(run_ids) + 1}"
        run_ids.append(run_id)
        return ExperimentTrainResult(
            experiment_id=experiment_id,
            run_id=run_id,
            predictions_path=Path(config_path).with_suffix(".predictions.parquet"),
            results_path=Path(config_path).with_suffix(".results.json"),
        )

    def fake_score_round(
        *,
        store_root: str | Path,
        experiment_id: str,
        round: str,
        stage: str,
    ) -> ExperimentScoreRoundResult:
        _ = (store_root, experiment_id, stage)
        return ExperimentScoreRoundResult(
            experiment_id=experiment_id,
            round=round,
            stage="post_training_full",
            run_ids=tuple(run_ids),
        )

    def fake_report(*, store_root: str | Path, experiment_id: str, metric: str, limit: int) -> ExperimentReport:
        _ = (store_root, experiment_id, metric, limit)
        rows = tuple(
            ExperimentReportRow(
                run_id=run_id,
                status="FINISHED",
                created_at="2026-03-20T00:00:00+00:00",
                metric_value=0.10 + (idx * 0.01),
                corr_mean=0.01,
                mmc_mean=0.02,
                cwmm_mean=0.03,
                bmc_mean=0.04 + (idx * 0.01),
                bmc_last_200_eras_mean=0.10 + (idx * 0.01),
                is_champion=False,
            )
            for idx, run_id in enumerate(run_ids)
        )
        return ExperimentReport(
            experiment_id=experiment_id,
            metric="bmc_last_200_eras.mean",
            total_runs=len(rows),
            champion_run_id=None,
            rows=rows,
        )

    monkeypatch.setattr("numereng.features.agentic_research.service.train_experiment", fake_train_experiment)
    monkeypatch.setattr("numereng.features.agentic_research.service.score_experiment_round", fake_score_round)
    monkeypatch.setattr("numereng.features.agentic_research.service.report_experiment", fake_report)

    result = run_research(store_root=store_root, experiment_id=root_exp.experiment_id, max_rounds=1)

    assert result.status == "stopped"
    assert result.stop_reason == "max_rounds_reached"
    program_path = store_root / "experiments" / root_exp.experiment_id / "agentic_research" / "program.json"
    state = load_program_state(program_path)
    assert state.total_rounds_completed == 1
    assert state.next_round_number == 2
    assert state.best_overall.run_id == "run-4"
    assert state.strategy == "numerai-experiment-loop"
    round_dir = store_root / "experiments" / root_exp.experiment_id / "agentic_research" / "rounds" / "r1"
    assert (round_dir / "round_summary.json").is_file()
    assert (round_dir / "codex_usage.json").is_file()
    assert (round_dir / "codex_last_message.json").is_file()
    usage = json.loads((round_dir / "codex_usage.json").read_text(encoding="utf-8"))
    assert "attempts" in usage


def test_run_research_resumes_after_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_root = tmp_path / ".numereng"
    root_exp = create_experiment(store_root=store_root, experiment_id="2026-03-20_resume-exp")
    init_research(
        store_root=store_root,
        experiment_id=root_exp.experiment_id,
        strategy="numerai-experiment-loop",
    )

    run_ids: list[str] = []
    train_calls = {"count": 0}

    monkeypatch.setattr(
        "numereng.features.agentic_research.service.run_codex_planner",
        lambda **_: _planner_execution(),
    )

    def fake_train_experiment(
        *,
        store_root: str | Path,
        experiment_id: str,
        config_path: str | Path,
    ) -> ExperimentTrainResult:
        _ = (store_root, experiment_id)
        train_calls["count"] += 1
        if train_calls["count"] == 2:
            raise KeyboardInterrupt()
        run_id = f"run-{len(run_ids) + 1}"
        run_ids.append(run_id)
        return ExperimentTrainResult(
            experiment_id=experiment_id,
            run_id=run_id,
            predictions_path=Path(config_path).with_suffix(".predictions.parquet"),
            results_path=Path(config_path).with_suffix(".results.json"),
        )

    def fake_score_round(
        *,
        store_root: str | Path,
        experiment_id: str,
        round: str,
        stage: str,
    ) -> ExperimentScoreRoundResult:
        _ = (store_root, experiment_id, stage)
        return ExperimentScoreRoundResult(
            experiment_id=experiment_id,
            round=round,
            stage="post_training_full",
            run_ids=tuple(run_ids),
        )

    def fake_report(*, store_root: str | Path, experiment_id: str, metric: str, limit: int) -> ExperimentReport:
        _ = (store_root, experiment_id, metric, limit)
        rows = tuple(
            ExperimentReportRow(
                run_id=run_id,
                status="FINISHED",
                created_at="2026-03-20T00:00:00+00:00",
                metric_value=0.10 + (idx * 0.01),
                corr_mean=0.01,
                mmc_mean=0.02,
                cwmm_mean=0.03,
                bmc_mean=0.04 + (idx * 0.01),
                bmc_last_200_eras_mean=0.10 + (idx * 0.01),
                is_champion=False,
            )
            for idx, run_id in enumerate(run_ids)
        )
        return ExperimentReport(
            experiment_id=experiment_id,
            metric="bmc_last_200_eras.mean",
            total_runs=len(rows),
            champion_run_id=None,
            rows=rows,
        )

    monkeypatch.setattr("numereng.features.agentic_research.service.train_experiment", fake_train_experiment)
    monkeypatch.setattr("numereng.features.agentic_research.service.score_experiment_round", fake_score_round)
    monkeypatch.setattr("numereng.features.agentic_research.service.report_experiment", fake_report)

    first = run_research(store_root=store_root, experiment_id=root_exp.experiment_id, max_rounds=1)
    assert first.interrupted is True
    interrupted_state = load_program_state(
        store_root / "experiments" / root_exp.experiment_id / "agentic_research" / "program.json"
    )
    assert interrupted_state.current_round is not None
    assert interrupted_state.current_round.next_config_index == 1

    def resumed_train_experiment(
        *,
        store_root: str | Path,
        experiment_id: str,
        config_path: str | Path,
    ) -> ExperimentTrainResult:
        _ = (store_root, experiment_id)
        run_id = f"run-{len(run_ids) + 1}"
        run_ids.append(run_id)
        return ExperimentTrainResult(
            experiment_id=experiment_id,
            run_id=run_id,
            predictions_path=Path(config_path).with_suffix(".predictions.parquet"),
            results_path=Path(config_path).with_suffix(".results.json"),
        )

    monkeypatch.setattr("numereng.features.agentic_research.service.train_experiment", resumed_train_experiment)

    second = run_research(store_root=store_root, experiment_id=root_exp.experiment_id, max_rounds=1)
    assert second.interrupted is False
    resumed_state = load_program_state(
        store_root / "experiments" / root_exp.experiment_id / "agentic_research" / "program.json"
    )
    assert resumed_state.total_rounds_completed == 1


def test_init_research_sets_phase_for_kaggle_strategy(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    root_exp = create_experiment(store_root=store_root, experiment_id="2026-03-20_kaggle-exp")

    result = init_research(
        store_root=store_root,
        experiment_id=root_exp.experiment_id,
        strategy="kaggle-gm-loop",
    )

    assert result.strategy == "kaggle-gm-loop"
    assert result.current_phase is not None
    assert result.current_phase.phase_id == "phase1_eda_baseline"


def test_run_research_advances_kaggle_phase_when_planner_requests_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_root = tmp_path / ".numereng"
    root_exp = create_experiment(store_root=store_root, experiment_id="2026-03-20_kaggle-loop-exp")
    init_research(
        store_root=store_root,
        experiment_id=root_exp.experiment_id,
        strategy="kaggle-gm-loop",
    )

    run_ids: list[str] = []
    decision = replace(
        _decision(),
        phase_action="advance",
        phase_transition_rationale="The baseline is established; move into the diversity campaign.",
    )
    monkeypatch.setattr(
        "numereng.features.agentic_research.service.run_codex_planner",
        lambda **_: _planner_execution(decision),
    )

    def fake_train_experiment(
        *,
        store_root: str | Path,
        experiment_id: str,
        config_path: str | Path,
    ) -> ExperimentTrainResult:
        _ = (store_root, experiment_id)
        run_id = f"run-{len(run_ids) + 1}"
        run_ids.append(run_id)
        return ExperimentTrainResult(
            experiment_id=experiment_id,
            run_id=run_id,
            predictions_path=Path(config_path).with_suffix(".predictions.parquet"),
            results_path=Path(config_path).with_suffix(".results.json"),
        )

    def fake_score_round(
        *,
        store_root: str | Path,
        experiment_id: str,
        round: str,
        stage: str,
    ) -> ExperimentScoreRoundResult:
        _ = (store_root, experiment_id, stage)
        return ExperimentScoreRoundResult(
            experiment_id=experiment_id,
            round=round,
            stage="post_training_full",
            run_ids=tuple(run_ids),
        )

    def fake_report(*, store_root: str | Path, experiment_id: str, metric: str, limit: int) -> ExperimentReport:
        _ = (store_root, experiment_id, metric, limit)
        rows = tuple(
            ExperimentReportRow(
                run_id=run_id,
                status="FINISHED",
                created_at="2026-03-20T00:00:00+00:00",
                metric_value=0.10 + (idx * 0.01),
                corr_mean=0.01,
                mmc_mean=0.02,
                cwmm_mean=0.03,
                bmc_mean=0.04 + (idx * 0.01),
                bmc_last_200_eras_mean=0.10 + (idx * 0.01),
                is_champion=False,
            )
            for idx, run_id in enumerate(run_ids)
        )
        return ExperimentReport(
            experiment_id=experiment_id,
            metric="bmc_last_200_eras.mean",
            total_runs=len(rows),
            champion_run_id=None,
            rows=rows,
        )

    monkeypatch.setattr("numereng.features.agentic_research.service.train_experiment", fake_train_experiment)
    monkeypatch.setattr("numereng.features.agentic_research.service.score_experiment_round", fake_score_round)
    monkeypatch.setattr("numereng.features.agentic_research.service.report_experiment", fake_report)

    result = run_research(store_root=store_root, experiment_id=root_exp.experiment_id, max_rounds=1)

    assert result.status == "stopped"
    assert result.current_phase is not None
    assert result.current_phase.phase_id == "phase2_diversity_campaign"
    assert result.current_phase.round_count == 1


def test_run_research_uses_openrouter_planner_when_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_root = tmp_path / ".numereng"
    root_exp = create_experiment(store_root=store_root, experiment_id="2026-03-20_openrouter-exp")
    init_research(
        store_root=store_root,
        experiment_id=root_exp.experiment_id,
        strategy="numerai-experiment-loop",
    )

    run_ids: list[str] = []
    planner_calls = {"openrouter": 0}

    monkeypatch.setattr("numereng.features.agentic_research.service.active_model_source", lambda: "openrouter")

    def fake_openrouter_planner(**_: object) -> CodexPlannerExecution:
        planner_calls["openrouter"] += 1
        return _planner_execution()

    def fail_codex_planner(**_: object) -> CodexPlannerExecution:
        raise AssertionError("codex planner should not run when openrouter is selected")

    monkeypatch.setattr("numereng.features.agentic_research.service.run_openrouter_planner", fake_openrouter_planner)
    monkeypatch.setattr("numereng.features.agentic_research.service.run_codex_planner", fail_codex_planner)

    def fake_train_experiment(
        *,
        store_root: str | Path,
        experiment_id: str,
        config_path: str | Path,
    ) -> ExperimentTrainResult:
        _ = (store_root, experiment_id)
        run_id = f"run-{len(run_ids) + 1}"
        run_ids.append(run_id)
        return ExperimentTrainResult(
            experiment_id=experiment_id,
            run_id=run_id,
            predictions_path=Path(config_path).with_suffix(".predictions.parquet"),
            results_path=Path(config_path).with_suffix(".results.json"),
        )

    def fake_score_round(
        *,
        store_root: str | Path,
        experiment_id: str,
        round: str,
        stage: str,
    ) -> ExperimentScoreRoundResult:
        _ = (store_root, experiment_id, stage)
        return ExperimentScoreRoundResult(
            experiment_id=experiment_id,
            round=round,
            stage="post_training_full",
            run_ids=tuple(run_ids),
        )

    def fake_report(*, store_root: str | Path, experiment_id: str, metric: str, limit: int) -> ExperimentReport:
        _ = (store_root, experiment_id, metric, limit)
        rows = tuple(
            ExperimentReportRow(
                run_id=run_id,
                status="FINISHED",
                created_at="2026-03-20T00:00:00+00:00",
                metric_value=0.10 + (idx * 0.01),
                corr_mean=0.01,
                mmc_mean=0.02,
                cwmm_mean=0.03,
                bmc_mean=0.04 + (idx * 0.01),
                bmc_last_200_eras_mean=0.10 + (idx * 0.01),
                is_champion=False,
            )
            for idx, run_id in enumerate(run_ids)
        )
        return ExperimentReport(
            experiment_id=experiment_id,
            metric="bmc_last_200_eras.mean",
            total_runs=len(rows),
            champion_run_id=None,
            rows=rows,
        )

    monkeypatch.setattr("numereng.features.agentic_research.service.train_experiment", fake_train_experiment)
    monkeypatch.setattr("numereng.features.agentic_research.service.score_experiment_round", fake_score_round)
    monkeypatch.setattr("numereng.features.agentic_research.service.report_experiment", fake_report)

    result = run_research(store_root=store_root, experiment_id=root_exp.experiment_id, max_rounds=1)

    assert result.status == "stopped"
    assert planner_calls["openrouter"] == 1
