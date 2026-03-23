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
from numereng.features.agentic_research.config_evolution import (
    build_candidate_filename,
    materialize_mutation_config,
    parse_mutation_response,
    select_parent_config,
)
from numereng.features.agentic_research.contracts import (
    CodexConfigPayload,
    CodexDecision,
    CodexPlannerExecution,
    RawPlannerExecution,
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
from numereng.features.agentic_research.prompting import render_prompt_template
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
        raw_response_text='{"next_action":"continue"}',
    )


def _raw_planner_execution(response_text: str) -> RawPlannerExecution:
    return RawPlannerExecution(
        attempts=[],
        stdout_jsonl=response_text,
        stderr_text="",
        raw_response_text=response_text,
    )


def _seed_config(store_root: Path, experiment_id: str, filename: str = "base.json") -> Path:
    config_dir = store_root / "experiments" / experiment_id / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / filename
    config_path.write_text(json.dumps(_valid_training_config(), indent=2, sort_keys=True), encoding="utf-8")
    return config_path


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

    assert numerai.planner_contract == "config_mutation"
    assert numerai.phase_aware is False
    assert numerai.prompt_path.name == "mutation_prompt.md"
    assert kaggle.phase_aware is True
    assert kaggle.phases[0].phase_id == "phase1_eda_baseline"
    assert kaggle.schema_path is not None
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


def test_run_codex_planner_parses_jsonl_and_last_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        command = list(args[0]) if args else []
        output_index = command.index("-o")
        output_path = Path(command[output_index + 1])
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
        schema_path=get_strategy_definition("kaggle-gm-loop").schema_path,
        artifact_dir=tmp_path,
    )

    assert execution.decision.next_action == "continue"
    assert len(execution.decision.configs) == 4
    assert len(execution.attempts) == 1
    assert execution.attempts[-1].thread_id == "thread_2"
    assert execution.attempts[-1].input_tokens == 123
    assert execution.attempts[-1].cached_input_tokens == 45
    assert execution.attempts[-1].output_tokens == 67
    assert "thread.started" in execution.raw_response_text


def test_render_prompt_template_requires_explicit_placeholder_replacement(tmp_path: Path) -> None:
    template_path = tmp_path / "prompt.md"
    template_path.write_text("Prompt\n$NAME\n$VALUE\n", encoding="utf-8")

    rendered = render_prompt_template(template_path, {"NAME": "alpha", "VALUE": "42"})

    assert rendered == "Prompt\nalpha\n42\n"

    with pytest.raises(ValueError, match="agentic_research_prompt_placeholders_unresolved"):
        render_prompt_template(template_path, {"NAME": "alpha"})


def test_parse_mutation_response_accepts_path_equals_json_literal() -> None:
    proposal = parse_mutation_response(
        
            "RATIONALE:\n"
            "Tighten LR while keeping the rest stable.\n\n"
            "CHANGES:\n"
            "config.model.params.learning_rate = 0.005\n"
            'config.data.feature_set = "medium"'
        
    )

    assert proposal.rationale.startswith("Tighten LR")
    assert [item.path for item in proposal.changes] == [
        "model.params.learning_rate",
        "data.feature_set",
    ]
    assert proposal.changes[0].value == 0.005
    assert proposal.changes[1].value == "medium"


def test_parse_mutation_response_rejects_invalid_or_disallowed_paths() -> None:
    with pytest.raises(ValueError, match="agentic_research_mutation_value_invalid:model.params.learning_rate"):
        parse_mutation_response(
            "RATIONALE:\nTest.\n\nCHANGES:\nconfig.model.params.learning_rate = nope"
        )

    with pytest.raises(ValueError, match="agentic_research_mutation_path_not_allowed:output.output_dir"):
        parse_mutation_response(
            "RATIONALE:\nTest.\n\nCHANGES:\nconfig.output.output_dir = \"/tmp/out\""
        )


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


def test_materialize_mutation_config_retries_duplicate_fingerprint(tmp_path: Path) -> None:
    experiment = create_experiment(store_root=tmp_path, experiment_id="2026-03-20_cfg-exp")
    config_dir = tmp_path / "experiments" / experiment.experiment_id / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    parent_path = config_dir / "base.json"
    parent_path.write_text(json.dumps(_valid_training_config(), indent=2, sort_keys=True), encoding="utf-8")
    parent = select_parent_config(
        root=tmp_path,
        experiment=experiment,
        report=None,
        config_dirs=[config_dir],
    )

    with pytest.raises(ValueError, match="agentic_research_candidate_duplicate"):
        materialize_mutation_config(
            round_label="r1",
            config_dir=config_dir,
            parent=parent,
            proposal=parse_mutation_response(
                "RATIONALE:\nKeep it identical.\n\nCHANGES:\nconfig.model.params.learning_rate = 0.01"
            ),
            comparison_dirs=[config_dir],
        )


def test_build_candidate_filename_is_deterministic() -> None:
    proposal = parse_mutation_response(
        "RATIONALE:\nTest.\n\nCHANGES:\nconfig.model.params.learning_rate = 0.005"
    )

    filename = build_candidate_filename(
        round_label="r7",
        config_dir=Path("/tmp"),
        parent_filename="base.json",
        proposal=proposal,
    )

    assert filename == "r7_base__model-params-learning-rate-0p005.json"


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
    _seed_config(store_root, root_exp.experiment_id)
    init_research(
        store_root=store_root,
        experiment_id=root_exp.experiment_id,
        strategy="numerai-experiment-loop",
    )

    run_ids: list[str] = []

    monkeypatch.setattr(
        "numereng.features.agentic_research.service.run_codex_raw_planner",
        lambda **_: _raw_planner_execution(
            
                "RATIONALE:\n"
                "Lower learning rate slightly while keeping the rest of the config stable.\n\n"
                "CHANGES:\n"
                "config.model.params.learning_rate = 0.005"
            
        ),
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
    assert state.best_overall.run_id == "run-1"
    assert state.strategy == "numerai-experiment-loop"
    round_dir = store_root / "experiments" / root_exp.experiment_id / "agentic_research" / "rounds" / "r1"
    assert (round_dir / "round_summary.json").is_file()
    assert (round_dir / "codex_usage.json").is_file()
    assert (round_dir / "codex_last_message.json").is_file()
    assert (round_dir / "llm_trace.jsonl").is_file()
    assert (round_dir / "llm_trace.md").is_file()
    usage = json.loads((round_dir / "codex_usage.json").read_text(encoding="utf-8"))
    assert "attempts" in usage
    trace_markdown = (
        store_root / "experiments" / root_exp.experiment_id / "agentic_research" / "llm_trace.md"
    ).read_text(encoding="utf-8")
    assert "### Sent To LLM" in trace_markdown
    assert "### Raw LLM Response" in trace_markdown
    assert "### Parsed Final Response" in trace_markdown
    assert "`codex-exec`" in trace_markdown
    assert "schema-valid JSON object" not in trace_markdown
    assert "config.model.params.learning_rate = 0.005" in trace_markdown
    trace_lines = [
        json.loads(line)
        for line in (store_root / "experiments" / root_exp.experiment_id / "agentic_research" / "llm_trace.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(trace_lines) == 1
    assert trace_lines[0]["status"] == "succeeded"
    assert trace_lines[0]["planner_source"] == "codex-exec"
    assert trace_lines[0]["round_label"] == "r1"
    assert trace_lines[0]["parsed_response"]["changes"][0]["path"] == "model.params.learning_rate"
    round_summary = json.loads((round_dir / "round_summary.json").read_text(encoding="utf-8"))
    assert round_summary["parent_config_filename"] == "base.json"
    assert round_summary["change_set"] == [{"path": "model.params.learning_rate", "value": 0.005}]


def test_run_research_retries_once_when_mutation_is_duplicate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_root = tmp_path / ".numereng"
    root_exp = create_experiment(store_root=store_root, experiment_id="2026-03-20_duplicate-exp")
    _seed_config(store_root, root_exp.experiment_id)
    init_research(
        store_root=store_root,
        experiment_id=root_exp.experiment_id,
        strategy="numerai-experiment-loop",
    )

    run_ids: list[str] = []
    planner_calls = {"count": 0}

    def fake_raw_planner(**_: object) -> RawPlannerExecution:
        planner_calls["count"] += 1
        if planner_calls["count"] == 1:
            return _raw_planner_execution(
                "RATIONALE:\nKeep the parent identical.\n\nCHANGES:\nconfig.model.params.learning_rate = 0.01"
            )
        return _raw_planner_execution(
            
                "RATIONALE:\n"
                "Lower learning rate to create a real child.\n\n"
                "CHANGES:\n"
                "config.model.params.learning_rate = 0.005"
            
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

    monkeypatch.setattr("numereng.features.agentic_research.service.run_codex_raw_planner", fake_raw_planner)
    monkeypatch.setattr("numereng.features.agentic_research.service.train_experiment", fake_train_experiment)
    monkeypatch.setattr("numereng.features.agentic_research.service.score_experiment_round", fake_score_round)
    monkeypatch.setattr("numereng.features.agentic_research.service.report_experiment", fake_report)

    result = run_research(store_root=store_root, experiment_id=root_exp.experiment_id, max_rounds=1)

    assert result.status == "stopped"
    assert planner_calls["count"] == 2
    round_dir = store_root / "experiments" / root_exp.experiment_id / "agentic_research" / "rounds" / "r1"
    assert "agentic_research_candidate_duplicate" in (round_dir / "codex_validation_error.txt").read_text(
        encoding="utf-8"
    )


def test_run_research_resumes_after_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_root = tmp_path / ".numereng"
    root_exp = create_experiment(store_root=store_root, experiment_id="2026-03-20_resume-exp")
    _seed_config(store_root, root_exp.experiment_id)
    init_research(
        store_root=store_root,
        experiment_id=root_exp.experiment_id,
        strategy="numerai-experiment-loop",
    )

    run_ids: list[str] = []
    score_calls = {"count": 0}

    monkeypatch.setattr(
        "numereng.features.agentic_research.service.run_codex_raw_planner",
        lambda **_: _raw_planner_execution(
            "RATIONALE:\nNudge learning rate down.\n\nCHANGES:\nconfig.model.params.learning_rate = 0.005"
        ),
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
        score_calls["count"] += 1
        if score_calls["count"] == 1:
            raise KeyboardInterrupt()
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
    _seed_config(store_root, root_exp.experiment_id)
    init_research(
        store_root=store_root,
        experiment_id=root_exp.experiment_id,
        strategy="numerai-experiment-loop",
    )

    run_ids: list[str] = []
    planner_calls = {"openrouter": 0}

    monkeypatch.setattr("numereng.features.agentic_research.service.active_model_source", lambda: "openrouter")

    def fake_openrouter_planner(**_: object) -> RawPlannerExecution:
        planner_calls["openrouter"] += 1
        return _raw_planner_execution(
            "RATIONALE:\nIncrease leaves modestly.\n\nCHANGES:\nconfig.model.params.num_leaves = 96"
        )

    def fail_codex_planner(**_: object) -> RawPlannerExecution:
        raise AssertionError("codex planner should not run when openrouter is selected")

    monkeypatch.setattr(
        "numereng.features.agentic_research.service.run_openrouter_raw_planner",
        fake_openrouter_planner,
    )
    monkeypatch.setattr("numereng.features.agentic_research.service.run_codex_raw_planner", fail_codex_planner)

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
