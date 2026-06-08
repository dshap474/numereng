"""Tests for the minimal agentic config-research loop."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import cast

import pytest

from numereng.features.agentic_research import run as research_module
from numereng.features.agentic_research.run import (
    AgenticResearchValidationError,
    get_research_status,
    run_research,
)
from numereng.features.experiments import (
    ExperimentReport,
    ExperimentReportRow,
    ExperimentScoreRoundResult,
    ExperimentTrainResult,
    create_experiment,
)
from numereng.features.training.errors import TrainingError
from numereng.platform.clients.openrouter import OpenRouterConfig

EXPERIMENT_ID = "2026-02-22_test-exp"


def _write_training_config(path: Path, *, learning_rate: float = 0.01) -> None:
    payload: dict[str, object] = {
        "data": {"data_version": "v5.2", "dataset_variant": "non_downsampled"},
        "model": {"type": "LGBMRegressor", "params": {"learning_rate": learning_rate}},
        "training": {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_cell_config(path: Path, *, family: str, feature_set: str, target: str) -> None:
    payload: dict[str, object] = {
        "data": {
            "data_version": "v5.2",
            "dataset_variant": "non_downsampled",
            "feature_set": feature_set,
            "target_col": target,
        },
        "model": {"type": family, "params": {"learning_rate": 0.01}},
        "training": {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _report(*rows: ExperimentReportRow) -> ExperimentReport:
    return ExperimentReport(
        experiment_id=EXPERIMENT_ID,
        metric="bmc_last_200_eras.mean",
        total_runs=len(rows),
        champion_run_id=None,
        rows=rows,
    )


def _row(run_id: str, value: float) -> ExperimentReportRow:
    return ExperimentReportRow(
        run_id=run_id,
        status="FINISHED",
        created_at="2026-02-22T00:00:00+00:00",
        metric_value=value,
        corr_mean=0.01,
        mmc_mean=0.02,
        cwmm_mean=0.03,
        bmc_mean=0.04,
        bmc_last_200_eras_mean=value,
        is_champion=False,
    )


def _llm_response(
    decision_form: dict[str, object],
    *,
    round_markdown: str = "# r001 Research State\n\nMemo.",
    experiment_markdown: str | None = None,
) -> str:
    return json.dumps(
        {
            "decision_form": decision_form,
            "round_markdown": round_markdown,
            "experiment_markdown": experiment_markdown,
        }
    )


def _set_experiment_metadata(manifest_path: Path, metadata: dict[str, object]) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["metadata"] = {**payload.get("metadata", {}), **metadata}
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _set_experiment_runs(manifest_path: Path, run_ids: list[str]) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["runs"] = list(run_ids)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _trace_events(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _run_plan_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _append_decision_row(experiment_dir: Path, payload: dict[str, object]) -> None:
    path = experiment_dir / "agentic_research" / "rounds" / "decision.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def test_status_synthesizes_blank_state(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")

    status = get_research_status(store_root=store_root, experiment_id=EXPERIMENT_ID)

    assert status.status == "initialized"
    assert status.next_round_number == 1
    assert status.trace_path == store_root / "experiments" / EXPERIMENT_ID / "agentic_research" / "trace.jsonl"
    assert status.program_path.name == "PROGRAM.md"


def test_status_uses_experiment_metadata_program(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")
    custom_program_dir = tmp_path / "custom_programs"
    custom_program_dir.mkdir()
    (custom_program_dir / "TEST-PROGRAM.md").write_text("Test program", encoding="utf-8")
    monkeypatch.setattr(research_module, "CUSTOM_PROGRAM_DIR", custom_program_dir)
    manifest = json.loads(experiment.manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"] = {"agentic_research_program": "TEST-PROGRAM.md"}
    experiment.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    status = get_research_status(store_root=store_root, experiment_id=EXPERIMENT_ID)

    assert status.program_path.name == "TEST-PROGRAM.md"
    assert status.program_path.parent == custom_program_dir


def test_run_research_runs_baseline_before_llm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")
    _write_training_config(experiment.manifest_path.parent / "configs" / "seed.json")

    reports: list[ExperimentReport | None] = [None, _report(_row("run-1", 0.12))]
    monkeypatch.setattr(
        research_module,
        "_safe_report",
        lambda **_: reports.pop(0) if reports else _report(_row("run-1", 0.12)),
    )
    monkeypatch.setattr(
        research_module,
        "train_experiment",
        lambda **_: ExperimentTrainResult(
            experiment_id=EXPERIMENT_ID,
            run_id="run-1",
            predictions_path=store_root / "runs" / "run-1" / "predictions.parquet",
            results_path=store_root / "runs" / "run-1" / "results.json",
        ),
    )
    monkeypatch.setattr(
        research_module,
        "score_experiment_round",
        lambda **_: ExperimentScoreRoundResult(
            experiment_id=EXPERIMENT_ID,
            round="r001",
            stage="post_training_core",
            run_ids=("run-1",),
        ),
    )

    result = run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=1)

    assert result.rounds[0].action == "baseline"
    assert result.rounds[0].run_id == "run-1"
    assert (experiment.manifest_path.parent / "configs" / "config_001.json").is_file()
    artifact_dir = experiment.manifest_path.parent / "agentic_research" / "rounds"
    assert result.rounds[0].artifact_dir == artifact_dir
    assert (experiment.manifest_path.parent / "agentic_research" / "ledger.jsonl").exists() is False
    assert not (experiment.manifest_path.parent / "agentic_research" / "trace.jsonl").exists()
    decisions = (artifact_dir / "decision.json").read_text(encoding="utf-8").splitlines()
    assert len(decisions) == 1
    assert json.loads(decisions[0])["round_label"] == "r001"
    assert json.loads(decisions[0])["decision"]["generated_config"] == "config_001.json"
    run_plan_rows = _run_plan_rows(experiment.manifest_path.parent / "run_plan.csv")
    assert run_plan_rows == [
        {
            "plan_index": "1",
            "round": "r001",
            "seed": "",
            "target": "",
            "horizon": "",
            "config_path": "configs/config_001.json",
            "score_stage_default": "post_training_core",
        }
    ]
    assert (artifact_dir / "r001.md").is_file()
    assert not (artifact_dir / "r001").exists()
    assert not (artifact_dir / "context.json").exists()
    assert not (artifact_dir / "round.json").exists()
    assert not (artifact_dir / "learning.md").exists()


def test_run_research_falls_back_to_on_disk_metric_when_run_off_leaderboard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When _safe_report's truncated leaderboard omits the just-trained run,
    metric_value must still be populated from runs/<run_id>/metrics.json."""
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")
    _write_training_config(experiment.manifest_path.parent / "configs" / "seed.json")

    other_run_row = _row("other-run-on-leaderboard", 0.99)
    monkeypatch.setattr(
        research_module,
        "_safe_report",
        lambda **_: _report(other_run_row),
    )
    monkeypatch.setattr(
        research_module,
        "train_experiment",
        lambda **_: ExperimentTrainResult(
            experiment_id=EXPERIMENT_ID,
            run_id="run-off-board",
            predictions_path=store_root / "runs" / "run-off-board" / "predictions.parquet",
            results_path=store_root / "runs" / "run-off-board" / "results.json",
        ),
    )
    monkeypatch.setattr(
        research_module,
        "score_experiment_round",
        lambda **_: ExperimentScoreRoundResult(
            experiment_id=EXPERIMENT_ID,
            round="r001",
            stage="post_training_core",
            run_ids=("run-off-board",),
        ),
    )

    metrics_path = store_root / "runs" / "run-off-board" / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps({"bmc_last_200_eras": {"mean": 0.00275}}),
        encoding="utf-8",
    )

    result = run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=1)

    assert result.rounds[0].run_id == "run-off-board"
    assert result.rounds[0].metric_value == pytest.approx(0.00275)
    round_notes = (experiment.manifest_path.parent / "agentic_research" / "rounds" / "r001.md").read_text(
        encoding="utf-8"
    )
    assert "bmc_last_200_eras_mean: 0.00275" in round_notes


def test_run_research_materializes_llm_config_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")
    _write_training_config(experiment.manifest_path.parent / "configs" / "seed.json")

    reports = [
        _report(_row("run-0", 0.10)),
        _report(_row("run-1", 0.13), _row("run-0", 0.10)),
    ]
    monkeypatch.setattr(
        research_module,
        "_safe_report",
        lambda **_: reports.pop(0) if reports else _report(_row("run-1", 0.13)),
    )
    monkeypatch.setattr(
        research_module,
        "_call_research_llm",
        lambda **_: (
            _llm_response(
                {
                    "action": "run",
                    "learning": "Lower variance looked useful.",
                    "belief_update": "A slightly larger learning rate is worth testing.",
                    "next_hypothesis": "A higher learning rate improves BMC without hurting sanity checks.",
                    "parent_config": "seed.json",
                    "changes": [
                        {
                            "path": "model.params.learning_rate",
                            "value": 0.02,
                            "reason": "The baseline underfit at 0.01.",
                        }
                    ],
                    "stop_reason": None,
                },
                round_markdown="# r001 Research State\n\nThe baseline underfit. Try a modest learning-rate lift.",
            ),
            "test",
        ),
    )
    monkeypatch.setattr(
        research_module,
        "train_experiment",
        lambda **_: ExperimentTrainResult(
            experiment_id=EXPERIMENT_ID,
            run_id="run-1",
            predictions_path=store_root / "runs" / "run-1" / "predictions.parquet",
            results_path=store_root / "runs" / "run-1" / "results.json",
        ),
    )
    monkeypatch.setattr(
        research_module,
        "score_experiment_round",
        lambda **_: ExperimentScoreRoundResult(
            experiment_id=EXPERIMENT_ID,
            round="r001",
            stage="post_training_core",
            run_ids=("run-1",),
        ),
    )

    metrics_path = store_root / "runs" / "run-1" / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "bmc_last_200_eras": {"mean": 0.13},
                "bmc": {"mean": 0.041},
                "corr": {"mean": 0.018},
                "mmc": {"mean": 0.007},
                "cwmm": {"mean": 0.012},
            }
        ),
        encoding="utf-8",
    )

    result = run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=1)

    config_path = result.rounds[0].config_path
    assert config_path is not None
    assert config_path.name == "config_001.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert payload["model"]["params"]["learning_rate"] == 0.02
    assert result.rounds[0].metric_value == 0.13
    artifact_dir = experiment.manifest_path.parent / "agentic_research" / "rounds"
    decisions = (artifact_dir / "decision.json").read_text(encoding="utf-8").splitlines()
    assert len(decisions) == 1
    assert json.loads(decisions[0])["decision"]["action"] == "run"
    assert (artifact_dir / "r001.md").is_file()
    round_notes = (artifact_dir / "r001.md").read_text(encoding="utf-8")
    assert "The baseline underfit. Try a modest learning-rate lift." in round_notes
    assert "## Execution Result" in round_notes
    assert "Run ID: run-1" in round_notes
    assert "bmc_last_200_eras_mean: 0.13" in round_notes
    assert "Config: configs/config_001.json" in round_notes
    assert "## Diff vs parent" in round_notes
    assert "model.params.learning_rate" in round_notes
    assert "## Secondary Metrics" in round_notes
    assert "bmc_mean: 0.041" in round_notes
    assert "## Outcome" in round_notes
    assert "Trigger cleared:" in round_notes
    # "Confirmation round:" line was retired in favor of the round_type-aware
    # footer; discovery rounds no longer print it.
    assert "Promoted:" in round_notes
    assert not (artifact_dir / "r001").exists()
    assert not (artifact_dir / "context.json").exists()
    assert not (artifact_dir / "prompt.md").exists()
    assert not (artifact_dir / "llm_response.txt").exists()
    assert not (artifact_dir / "codex_stdout.jsonl").exists()
    assert not (artifact_dir / "codex_stderr.txt").exists()
    assert not (artifact_dir / "round.json").exists()
    assert not (artifact_dir / "learning.md").exists()
    trace = _trace_events(experiment.manifest_path.parent / "agentic_research" / "trace.jsonl")
    assert [item["event"] for item in trace] == [
        "prompt_rendered",
        "llm_response",
        "decision_parsed",
        "config_written",
        "round_completed",
    ]
    assert trace[0]["round_label"] == "r001"
    prompt_payload = cast(dict[str, object], trace[0]["payload"])
    assert "seed.json" in str(prompt_payload["prompt"])
    response_payload = cast(dict[str, object], trace[1]["payload"])
    assert response_payload["model_source"] == "test"
    assert "Lower variance looked useful." in str(response_payload["raw_response"])
    parsed_payload = cast(dict[str, object], trace[2]["payload"])
    parsed_decision = cast(dict[str, object], parsed_payload["decision"])
    assert parsed_decision["belief_update"] == "A slightly larger learning rate is worth testing."
    completed_payload = cast(dict[str, object], trace[-1]["payload"])
    assert completed_payload["run_id"] == "run-1"
    assert completed_payload["metric_value"] == 0.13


def test_run_research_writes_curated_experiment_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the LLM returns experiment_markdown, it replaces EXPERIMENT.md."""
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")
    _write_training_config(experiment.manifest_path.parent / "configs" / "seed.json")
    experiment_md_path = experiment.manifest_path.parent / "EXPERIMENT.md"
    experiment_md_path.write_text("# Stale\nPrior content.\n", encoding="utf-8")

    reports = [
        _report(_row("run-0", 0.10)),
        _report(_row("run-1", 0.13), _row("run-0", 0.10)),
    ]
    monkeypatch.setattr(
        research_module,
        "_safe_report",
        lambda **_: reports.pop(0) if reports else _report(_row("run-1", 0.13)),
    )
    curated = "# Active Beliefs\n- LR 0.02 beats 0.01.\n\n# Closed Hypotheses\n- depth=4 hurts.\n"
    monkeypatch.setattr(
        research_module,
        "_call_research_llm",
        lambda **_: (
            _llm_response(
                {
                    "action": "run",
                    "learning": "OK.",
                    "belief_update": "LR matters.",
                    "next_hypothesis": "Try lr=0.02.",
                    "parent_config": "seed.json",
                    "changes": [{"path": "model.params.learning_rate", "value": 0.02, "reason": "probe"}],
                    "stop_reason": None,
                },
                experiment_markdown=curated,
            ),
            "test",
        ),
    )
    monkeypatch.setattr(
        research_module,
        "train_experiment",
        lambda **_: ExperimentTrainResult(
            experiment_id=EXPERIMENT_ID,
            run_id="run-1",
            predictions_path=store_root / "runs" / "run-1" / "predictions.parquet",
            results_path=store_root / "runs" / "run-1" / "results.json",
        ),
    )
    monkeypatch.setattr(
        research_module,
        "score_experiment_round",
        lambda **_: ExperimentScoreRoundResult(
            experiment_id=EXPERIMENT_ID,
            round="r001",
            stage="post_training_core",
            run_ids=("run-1",),
        ),
    )

    run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=1)

    assert experiment_md_path.read_text(encoding="utf-8") == curated
    trace = _trace_events(experiment.manifest_path.parent / "agentic_research" / "trace.jsonl")
    [update_event] = [e for e in trace if e["event"] == "experiment_markdown_updated"]
    assert cast(dict[str, object], update_event["payload"])["bytes_written"] == len(curated)
    events = [e["event"] for e in trace]
    assert events.index("experiment_markdown_updated") > events.index("config_written")


def test_run_research_preserves_experiment_markdown_when_llm_omits_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When experiment_markdown is null, the prior EXPERIMENT.md is preserved."""
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")
    _write_training_config(experiment.manifest_path.parent / "configs" / "seed.json")
    experiment_md_path = experiment.manifest_path.parent / "EXPERIMENT.md"
    prior = "# Prior\nKept across rounds.\n"
    experiment_md_path.write_text(prior, encoding="utf-8")

    reports = [
        _report(_row("run-0", 0.10)),
        _report(_row("run-1", 0.13), _row("run-0", 0.10)),
    ]
    monkeypatch.setattr(
        research_module,
        "_safe_report",
        lambda **_: reports.pop(0) if reports else _report(_row("run-1", 0.13)),
    )
    monkeypatch.setattr(
        research_module,
        "_call_research_llm",
        lambda **_: (
            _llm_response(
                {
                    "action": "run",
                    "learning": "OK.",
                    "belief_update": "Stable.",
                    "next_hypothesis": "Probe.",
                    "parent_config": "seed.json",
                    "changes": [{"path": "model.params.learning_rate", "value": 0.02, "reason": "probe"}],
                    "stop_reason": None,
                },
                experiment_markdown=None,
            ),
            "test",
        ),
    )
    monkeypatch.setattr(
        research_module,
        "train_experiment",
        lambda **_: ExperimentTrainResult(
            experiment_id=EXPERIMENT_ID,
            run_id="run-1",
            predictions_path=store_root / "runs" / "run-1" / "predictions.parquet",
            results_path=store_root / "runs" / "run-1" / "results.json",
        ),
    )
    monkeypatch.setattr(
        research_module,
        "score_experiment_round",
        lambda **_: ExperimentScoreRoundResult(
            experiment_id=EXPERIMENT_ID,
            round="r001",
            stage="post_training_core",
            run_ids=("run-1",),
        ),
    )

    run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=1)

    assert experiment_md_path.read_text(encoding="utf-8") == prior
    trace = _trace_events(experiment.manifest_path.parent / "agentic_research" / "trace.jsonl")
    assert not [e for e in trace if e["event"] == "experiment_markdown_updated"]


def test_round_config_filename_stays_short() -> None:
    assert research_module._round_config_filename("r004") == "config_004.json"


def test_record_round_config_in_run_plan_is_idempotent(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")
    config_path = experiment.manifest_path.parent / "configs" / "config_004.json"
    config_path.write_text("{}", encoding="utf-8")

    research_module._record_round_config_in_run_plan(experiment=experiment, round_label="r004", config_path=config_path)
    research_module._record_round_config_in_run_plan(experiment=experiment, round_label="r004", config_path=config_path)

    rows = _run_plan_rows(experiment.manifest_path.parent / "run_plan.csv")
    assert len(rows) == 1
    assert rows[0]["round"] == "r004"
    assert rows[0]["config_path"] == "configs/config_004.json"


def test_run_plan_records_relative_config_path_for_portability(tmp_path: Path) -> None:
    """`run_plan.csv` must hold paths relative to the experiment dir so the artifact
    is portable across machines (PC → laptop sync)."""
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")
    config_path = experiment.manifest_path.parent / "configs" / "config_005.json"
    config_path.write_text("{}", encoding="utf-8")

    research_module._record_round_config_in_run_plan(experiment=experiment, round_label="r005", config_path=config_path)
    rows = _run_plan_rows(experiment.manifest_path.parent / "run_plan.csv")
    recorded = rows[0]["config_path"]
    assert recorded == "configs/config_005.json"
    assert not Path(recorded).is_absolute()


def test_run_plan_recorded_before_score_experiment_round(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """score_experiment_round resolves the round via run_plan.csv; the row must exist when it runs.

    Regression for the ordering bug where _record_round_config_in_run_plan was called after
    score_experiment_round, causing experiment_round_not_found:rNNN failures because
    configs/config_NNN.json doesn't match the score fallback glob configs/rNNN_*.json.
    """
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")
    _write_training_config(experiment.manifest_path.parent / "configs" / "seed.json")
    run_plan_path = experiment.manifest_path.parent / "run_plan.csv"

    seen_rounds: list[str] = []

    def _score_spy(**kwargs: object) -> ExperimentScoreRoundResult:
        round_label = cast(str, kwargs["round"])
        assert run_plan_path.is_file(), "run_plan.csv must exist when score_experiment_round runs"
        rows = _run_plan_rows(run_plan_path)
        assert any(row["round"] == round_label for row in rows), (
            f"run_plan.csv must contain {round_label} when score_experiment_round runs; "
            f"got rounds {[r['round'] for r in rows]}"
        )
        seen_rounds.append(round_label)
        return ExperimentScoreRoundResult(
            experiment_id=EXPERIMENT_ID, round=round_label, stage="post_training_core", run_ids=("run-1",)
        )

    monkeypatch.setattr(research_module, "_safe_report", lambda **_: _report(_row("run-1", 0.12)))
    monkeypatch.setattr(
        research_module,
        "train_experiment",
        lambda **_: ExperimentTrainResult(
            experiment_id=EXPERIMENT_ID,
            run_id="run-1",
            predictions_path=store_root / "runs" / "run-1" / "predictions.parquet",
            results_path=store_root / "runs" / "run-1" / "results.json",
        ),
    )
    monkeypatch.setattr(research_module, "score_experiment_round", _score_spy)

    run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=1)
    assert seen_rounds == ["r001"]


def test_context_includes_only_latest_round_markdown(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")
    rounds_dir = experiment.manifest_path.parent / "agentic_research" / "rounds"
    rounds_dir.mkdir(parents=True)
    (rounds_dir / "r001.md").write_text("old memo", encoding="utf-8")
    (rounds_dir / "r002.md").write_text("latest memo", encoding="utf-8")
    (rounds_dir / "r999.debug.prompt.md").write_text("debug prompt", encoding="utf-8")

    context = research_module._build_context(
        root=store_root,
        experiment=experiment,
        report=_report(_row("run-0", 0.10)),
        state={},
        eligible_ensemble_rows=[],
    )

    assert context["latest_round_markdown"] == "latest memo"
    assert "old memo" not in json.dumps(context)
    assert "debug prompt" not in json.dumps(context)


def test_coverage_surface_metadata_requires_three_nonempty_axes(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")
    _set_experiment_metadata(
        experiment.manifest_path,
        {
            "agentic_research_coverage_surface": {
                "families": ["LGBMRegressor", "LGBMRegressor", " XGBoostRegressor "],
                "feature_sets": ["small"],
                "targets": ["target_ender_20"],
            }
        },
    )
    experiment = research_module.get_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID)

    assert research_module._coverage_surface(experiment) == {
        "families": ("LGBMRegressor", "XGBoostRegressor"),
        "feature_sets": ("small",),
        "targets": ("target_ender_20",),
    }

    _set_experiment_metadata(
        experiment.manifest_path,
        {"agentic_research_coverage_surface": {"families": ["LGBMRegressor"], "feature_sets": [], "targets": []}},
    )
    experiment = research_module.get_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID)
    assert research_module._coverage_surface(experiment) is None


def test_coverage_context_uses_completed_decision_log_cells(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")
    experiment_dir = experiment.manifest_path.parent
    _write_cell_config(
        experiment_dir / "configs" / "config_001.json",
        family="LGBMRegressor",
        feature_set="small",
        target="target_ender_20",
    )
    _write_cell_config(
        experiment_dir / "configs" / "config_002.json",
        family="LGBMRegressor",
        feature_set="small",
        target="target_ender_20",
    )
    _write_cell_config(
        experiment_dir / "configs" / "config_003.json",
        family="XGBoostRegressor",
        feature_set="medium",
        target="target_alpha_60",
    )
    _write_cell_config(
        experiment_dir / "configs" / "config_004.json",
        family="LGBMRegressor",
        feature_set="medium",
        target="target_alpha_60",
    )
    _set_experiment_metadata(
        experiment.manifest_path,
        {
            "agentic_research_coverage_surface": {
                "families": ["LGBMRegressor", "XGBoostRegressor"],
                "feature_sets": ["small", "medium"],
                "targets": ["target_ender_20", "target_alpha_60"],
            }
        },
    )
    experiment = research_module.get_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID)
    windows_path = f"D:\\workspace\\numereng\\.numereng\\experiments\\{EXPERIMENT_ID}\\configs\\config_003.json"
    _append_decision_row(
        experiment_dir,
        {"action": "baseline", "status": "completed", "config_path": "configs\\config_001.json", "metric_value": 0.1},
    )
    _append_decision_row(
        experiment_dir,
        {"action": "run", "status": "completed", "config_path": "configs/config_002.json", "metric_value": 0.2},
    )
    _append_decision_row(
        experiment_dir,
        {"action": "run", "status": "completed", "config_path": windows_path, "metric_value": 0.05},
    )
    _append_decision_row(
        experiment_dir,
        {"action": "run", "status": "skipped", "config_path": "configs/config_004.json", "metric_value": 0.9},
    )
    _append_decision_row(
        experiment_dir,
        {"action": "ensemble", "status": "completed", "config_path": None, "metric_value": 0.8},
    )

    tested = research_module._coverage_tested_cells(root=store_root, experiment=experiment)
    assert len(tested) == 2
    assert ("LGBMRegressor", "small", "target_ender_20") in tested
    assert ("XGBoostRegressor", "medium", "target_alpha_60") in tested
    assert ("LGBMRegressor", "medium", "target_alpha_60") not in tested

    context = research_module._coverage_context(root=store_root, experiment=experiment)
    assert context["surface_declared"] is True
    assert context["cells_total"] == 8
    assert context["cells_tested"] == 2
    assert context["cells_untested"] == 6
    assert context["tested_outside_surface"] == 0
    assert len(cast(list[object], context["untested_sample"])) == 6


def test_coverage_context_degrades_without_surface_metadata(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")
    experiment_dir = experiment.manifest_path.parent
    _write_cell_config(
        experiment_dir / "configs" / "config_001.json",
        family="LGBMRegressor",
        feature_set="small",
        target="target_ender_20",
    )
    _append_decision_row(
        experiment_dir,
        {"action": "run", "status": "completed", "config_path": "configs/config_001.json", "metric_value": 0.1},
    )

    context = research_module._coverage_context(root=store_root, experiment=experiment)

    assert context == {
        "surface_declared": False,
        "cells_total": None,
        "cells_tested": 1,
        "cells_untested": None,
        "tested_outside_surface": 0,
        "untested_sample": [],
    }


def test_coverage_supplements_from_uncapped_report_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")
    run_ids = [f"run-{i:02d}" for i in range(30)]
    _set_experiment_runs(experiment.manifest_path, run_ids)
    experiment = research_module.get_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID)
    for index, run_id in enumerate(run_ids):
        run_dir = store_root / "runs" / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": "FINISHED",
                    "model": {"type": "LGBMRegressor"},
                    "data": {"feature_set": "small", "target_col": f"target_{index:02d}_20"},
                }
            ),
            encoding="utf-8",
        )

    def fake_report_experiment(**kwargs: object) -> ExperimentReport:
        assert kwargs["limit"] == len(run_ids)
        return _report(*[_row(run_id, float(index)) for index, run_id in enumerate(run_ids)])

    monkeypatch.setattr(research_module, "report_experiment", fake_report_experiment)

    tested = research_module._coverage_tested_cells(root=store_root, experiment=experiment)

    assert len(tested) == 30
    assert ("LGBMRegressor", "small", "target_29_20") in tested


def test_build_context_caps_coverage_sample_and_uses_program_allowed_paths(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")
    experiment_dir = experiment.manifest_path.parent
    _write_cell_config(
        experiment_dir / "configs" / "config_001.json",
        family="LGBMRegressor",
        feature_set="small",
        target="target_ender_20",
    )
    _append_decision_row(
        experiment_dir,
        {"action": "run", "status": "completed", "config_path": "configs/config_001.json", "metric_value": 0.1},
    )
    _set_experiment_metadata(
        experiment.manifest_path,
        {
            "agentic_research_allowed_change_paths": ["model.params.learning_rate"],
            "agentic_research_coverage_surface": {
                "families": ["LGBMRegressor", "XGBoostRegressor", "TabPFNRegressor"],
                "feature_sets": ["small", "medium", "all"],
                "targets": ["target_ender_20", "target_alpha_60"],
            },
        },
    )
    experiment = research_module.get_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID)

    context = research_module._build_context(
        root=store_root,
        experiment=experiment,
        report=_report(_row("run-0", 0.10)),
        state={},
        eligible_ensemble_rows=[],
    )

    assert context["allowed_change_paths"] == ["model.params.learning_rate"]
    coverage = cast(dict[str, object], context["coverage"])
    assert coverage["cells_total"] == 18
    assert coverage["cells_tested"] == 1
    assert coverage["cells_untested"] == 17
    assert len(cast(list[object], coverage["untested_sample"])) == research_module.COVERAGE_UNTESTED_SAMPLE_LIMIT


def test_run_research_traces_decision_parse_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")
    _write_training_config(experiment.manifest_path.parent / "configs" / "seed.json")
    monkeypatch.setattr(research_module, "_safe_report", lambda **_: _report(_row("run-0", 0.10)))
    monkeypatch.setattr(research_module, "_call_research_llm", lambda **_: ("not json", "test"))

    result = run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=1)

    assert len(result.rounds) == 1
    assert result.rounds[0].status == "failed"
    assert "agentic_research_json_missing" in result.rounds[0].learning
    trace = _trace_events(experiment.manifest_path.parent / "agentic_research" / "trace.jsonl")
    assert [item["event"] for item in trace] == [
        "prompt_rendered",
        "llm_response",
        "decision_parse_failed",
        "round_failed",
    ]
    failure_payload = cast(dict[str, object], trace[-2]["payload"])
    assert failure_payload["raw_response"] == "not json"
    assert failure_payload["error"] == "agentic_research_json_missing"
    artifact_dir = experiment.manifest_path.parent / "agentic_research" / "rounds"
    assert (artifact_dir / "r001.debug.prompt.md").is_file()
    assert (artifact_dir / "r001.debug.llm_response.txt").read_text(encoding="utf-8") == "not json"
    failure_memo = (artifact_dir / "r001.md").read_text(encoding="utf-8")
    assert "## Execution Result" in failure_memo
    assert "Status: failed" in failure_memo
    assert "agentic_research_json_missing" in failure_memo


def test_run_research_traces_llm_call_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")
    _write_training_config(experiment.manifest_path.parent / "configs" / "seed.json")
    monkeypatch.setattr(research_module, "_safe_report", lambda **_: _report(_row("run-0", 0.10)))

    def fail_llm(**_: object) -> tuple[str, str]:
        raise research_module.AgenticResearchError("agentic_research_codex_failed:1:boom")

    monkeypatch.setattr(research_module, "_call_research_llm", fail_llm)

    result = run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=1)

    assert len(result.rounds) == 1
    assert result.rounds[0].status == "failed"
    assert "agentic_research_codex_failed" in result.rounds[0].learning
    trace = _trace_events(experiment.manifest_path.parent / "agentic_research" / "trace.jsonl")
    assert [item["event"] for item in trace] == [
        "prompt_rendered",
        "llm_call_failed",
        "round_failed",
    ]
    llm_payload = cast(dict[str, object], trace[1]["payload"])
    assert llm_payload["error"] == "agentic_research_codex_failed:1:boom"
    artifact_dir = experiment.manifest_path.parent / "agentic_research" / "rounds"
    assert "agentic_research_codex_failed" in (artifact_dir / "r001.debug.error.txt").read_text(encoding="utf-8")


def test_call_codex_exec_uses_configured_model_and_reasoning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(
        cmd: list[str],
        *,
        input: str,
        text: bool,
        capture_output: bool,
        check: bool,
        timeout: float | None = None,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        captured["input"] = input
        captured["timeout"] = timeout
        captured["encoding"] = encoding
        captured["errors"] = errors
        _ = (text, capture_output, check)
        schema_path = Path(cmd[cmd.index("--output-schema") + 1])
        captured["schema"] = json.loads(schema_path.read_text(encoding="utf-8"))
        output_path = Path(cmd[cmd.index("-o") + 1])
        output_path.write_text(
            _llm_response(
                {
                    "action": "run",
                    "learning": "done",
                    "belief_update": "done",
                    "next_hypothesis": None,
                    "parent_config": "seed.json",
                    "changes": [{"path": "model.params.n_estimators", "value": 200, "reason": "probe"}],
                    "stop_reason": None,
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(research_module.subprocess, "run", fake_run)

    response = research_module._call_codex_exec(
        prompt="choose next run",
        artifact_dir=tmp_path,
        round_label="r001",
        config=OpenRouterConfig(
            active_model_source="codex-exec",
            active_model="gpt-5.5",
            active_model_reasoning_effort="high",
        ),
    )

    cmd = cast(list[str], captured["cmd"])
    assert cmd[cmd.index("--model") + 1] == "gpt-5.5"
    assert cmd[cmd.index("-c") + 1] == 'model_reasoning_effort="high"'
    assert "--output-schema" in cmd
    # Built-in image_generation tool is disabled — research only needs JSON back, and a
    # transient OpenAI 400 on its gpt-image-2 model bailed a live run (r625).
    assert cmd[cmd.index("--disable") + 1] == "image_generation"
    assert captured["timeout"] == research_module.CODEX_TIMEOUT_SECONDS
    assert captured["encoding"] == "utf-8"
    assert captured["errors"] == "replace"
    schema = cast(dict[str, object], captured["schema"])
    assert "decision_form" in cast(dict[str, object], schema["properties"])
    assert json.loads(response)["decision_form"]["action"] == "run"
    assert captured["input"] == "choose next run"
    assert not list(tmp_path.glob(".codex_schema_*.json"))
    assert not list(tmp_path.glob(".codex_output_*.txt"))
    assert not (tmp_path / "codex_stdout.jsonl").exists()
    assert not (tmp_path / "codex_stderr.txt").exists()


def test_resolve_codex_executable_prefers_windows_cmd(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_which(name: str) -> str | None:
        calls.append(name)
        return "C:\\tools\\npm\\codex.cmd" if name == "codex.cmd" else None

    monkeypatch.setattr(research_module.os, "name", "nt")
    monkeypatch.setattr(research_module.shutil, "which", fake_which)

    assert research_module._resolve_codex_executable().endswith("codex.cmd")
    assert calls == ["codex.cmd"]


def test_call_codex_exec_writes_debug_artifacts_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(
        cmd: list[str],
        *,
        input: str,
        text: bool,
        capture_output: bool,
        check: bool,
        timeout: float | None = None,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        _ = (cmd, input, text, capture_output, check, timeout, encoding, errors)
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout='{"event":"failed"}', stderr="boom")

    monkeypatch.setattr(research_module.subprocess, "run", fake_run)

    with pytest.raises(research_module.AgenticResearchError, match="agentic_research_codex_failed"):
        research_module._call_codex_exec(
            prompt="choose next run",
            artifact_dir=tmp_path,
            round_label="r002",
            config=OpenRouterConfig(
                active_model_source="codex-exec",
                active_model="gpt-5.5",
                active_model_reasoning_effort="high",
            ),
        )

    assert (tmp_path / "r002.debug.prompt.md").read_text(encoding="utf-8") == "choose next run"
    assert (tmp_path / "r002.debug.codex_stdout.jsonl").read_text(encoding="utf-8") == '{"event":"failed"}'
    assert (tmp_path / "r002.debug.codex_stderr.txt").read_text(encoding="utf-8") == "boom"
    assert "agentic_research_codex_failed" in (tmp_path / "r002.debug.error.txt").read_text(encoding="utf-8")


def test_materialize_clamps_value_outside_program_cap(tmp_path: Path) -> None:
    """Out-of-range numeric proposals are clamped into the phase range, not rejected —
    a cap violation no longer wastes a round (see harden 2026-06-06, item 2)."""
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")
    _write_training_config(experiment.manifest_path.parent / "configs" / "seed.json")
    _set_experiment_metadata(
        experiment.manifest_path,
        {"agentic_research_value_caps": {"model.params.learning_rate": [0.01, 0.30]}},
    )
    experiment = research_module.get_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID)
    decision = research_module.ResearchDecision(
        action="run",
        learning="x",
        belief_update="x",
        next_hypothesis="x",
        parent_config="seed.json",
        changes=(research_module.ResearchChange(path="model.params.learning_rate", value=0.9, reason="over the cap"),),
        stop_reason=None,
    )
    path = research_module._materialize_decision_config(experiment=experiment, round_label="r002", decision=decision)
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["model"]["params"]["learning_rate"] == 0.30


def test_materialize_clamps_num_leaves_up_to_phase_floor(tmp_path: Path) -> None:
    """The documented family-switch case: switching to LGBM in deep phase mirrors
    XGBoost's small max_leaves into num_leaves below the floor. Clamp UP to the floor
    instead of failing the round."""
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")
    _write_training_config(experiment.manifest_path.parent / "configs" / "seed.json")
    _set_experiment_metadata(
        experiment.manifest_path,
        {"agentic_research_value_caps": {"model.params.num_leaves": [128, 1024]}},
    )
    experiment = research_module.get_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID)
    decision = research_module.ResearchDecision(
        action="run",
        learning="x",
        belief_update="x",
        next_hypothesis="x",
        parent_config="seed.json",
        changes=(research_module.ResearchChange(path="model.params.num_leaves", value=8, reason="mirrored leaves"),),
        stop_reason=None,
    )
    path = research_module._materialize_decision_config(experiment=experiment, round_label="r002", decision=decision)
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["model"]["params"]["num_leaves"] == 128  # clamped up, int preserved


def test_derive_target_horizon_by_suffix() -> None:
    assert research_module._derive_target_horizon("target_alpha_20") == "20d"
    assert research_module._derive_target_horizon("target_ender_60") == "60d"
    assert research_module._derive_target_horizon("target") is None  # unknown suffix left untouched
    assert research_module._derive_target_horizon(None) is None


def test_materialize_auto_derives_target_horizon_from_target_col(tmp_path: Path) -> None:
    """target_horizon is controller-managed: changing target_col alone rewrites the
    horizon to match the suffix (item 1)."""
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")
    _write_training_config(experiment.manifest_path.parent / "configs" / "seed.json")
    experiment = research_module.get_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID)
    decision = research_module.ResearchDecision(
        action="run",
        learning="x",
        belief_update="x",
        next_hypothesis="x",
        parent_config="seed.json",
        changes=(research_module.ResearchChange(path="data.target_col", value="target_alpha_20", reason="new target"),),
        stop_reason=None,
    )
    path = research_module._materialize_decision_config(experiment=experiment, round_label="r002", decision=decision)
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["data"]["target_horizon"] == "20d"


def test_normalize_strips_controller_managed_target_horizon() -> None:
    """A bundled target_horizon change is dropped before it counts against the budget
    or shows up as no-op normalization noise (item 1)."""
    decision = research_module.ResearchDecision(
        action="run",
        learning="x",
        belief_update="x",
        next_hypothesis="x",
        parent_config="config_001.json",
        changes=(
            research_module.ResearchChange(path="data.target_col", value="target_alpha_60", reason="target"),
            research_module.ResearchChange(path="data.target_horizon", value="60d", reason="coupled"),
        ),
        stop_reason=None,
    )
    parent_payload = {"data": {"target_col": "target_alpha_20", "target_horizon": "20d"}}
    normalized, summary = research_module._normalize_decision_changes(decision=decision, parent_payload=parent_payload)
    assert [c.path for c in normalized.changes] == ["data.target_col"]
    assert summary["dropped_managed_paths"] == ["data.target_horizon"]
    assert summary["kept_count"] == 1


def test_materialize_rejects_path_outside_program_allowed_list(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")
    _write_training_config(experiment.manifest_path.parent / "configs" / "seed.json")
    _set_experiment_metadata(
        experiment.manifest_path,
        {"agentic_research_allowed_change_paths": ["model.params.learning_rate"]},
    )
    experiment = research_module.get_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID)
    decision = research_module.ResearchDecision(
        action="run",
        learning="x",
        belief_update="x",
        next_hypothesis="x",
        parent_config="seed.json",
        changes=(
            research_module.ResearchChange(path="model.params.num_leaves", value=8, reason="not in narrowed list"),
        ),
        stop_reason=None,
    )
    with pytest.raises(AgenticResearchValidationError, match="agentic_research_change_path_not_allowed"):
        research_module._materialize_decision_config(experiment=experiment, round_label="r002", decision=decision)


def test_family_switch_to_xgboost_strips_lgbm_only_params_and_device() -> None:
    payload = {
        "data": {"data_version": "v5.2"},
        "model": {
            "type": "XGBoostRegressor",
            "module_path": "xgboost_model.py",
            "device": "cuda",
            "params": {
                "max_depth": 5,
                "num_leaves": 32,
                "min_child_samples": 100,
                "bagging_freq": 5,
                "reg_alpha": 0.1,
                "device_type": "cuda",
                "max_leaves": 8,
                "min_child_weight": 200,
                "learning_rate": 0.1,
            },
        },
        "training": {},
    }
    research_module._apply_family_switch_cleanup(payload)
    assert "device" not in payload["model"]
    params = payload["model"]["params"]
    for key in ("num_leaves", "min_child_samples", "bagging_freq", "reg_alpha", "device_type"):
        assert key not in params
    assert params["max_leaves"] == 8
    assert params["min_child_weight"] == 200
    assert params["learning_rate"] == 0.1


def test_family_switch_to_lgbm_strips_xgboost_only_params_and_module_path() -> None:
    payload = {
        "data": {"data_version": "v5.2"},
        "model": {
            "type": "LGBMRegressor",
            "module_path": "xgboost_model.py",
            "params": {
                "max_depth": 5,
                "num_leaves": 16,
                "min_child_samples": 200,
                "max_leaves": 64,
                "min_child_weight": 100,
                "learning_rate": 0.1,
            },
        },
        "training": {},
    }
    research_module._apply_family_switch_cleanup(payload)
    assert "module_path" not in payload["model"]
    params = payload["model"]["params"]
    for key in ("max_leaves", "min_child_weight"):
        assert key not in params
    assert params["num_leaves"] == 16


def test_family_switch_cleanup_strips_null_params() -> None:
    payload = {
        "data": {"data_version": "v5.2"},
        "model": {
            "type": "XGBoostRegressor",
            "module_path": "xgboost_model.py",
            "params": {"max_leaves": 8, "min_child_weight": 100, "subsample": None},
        },
        "training": {},
    }
    research_module._apply_family_switch_cleanup(payload)
    assert "subsample" not in payload["model"]["params"]


def test_materialize_decision_accepts_null_value_for_capped_path(tmp_path: Path) -> None:
    """A `null` value on a capped LGBM-only path during a family switch must not
    trip the value-cap range check; the auto-cleanup removes the key entirely."""
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")
    configs_dir = experiment.manifest_path.parent / "configs"
    seed_payload = {
        "data": {"data_version": "v5.2", "dataset_variant": "non_downsampled"},
        "model": {
            "type": "LGBMRegressor",
            "params": {"max_depth": 3, "num_leaves": 8, "min_child_samples": 100, "learning_rate": 0.1},
        },
        "training": {},
    }
    (configs_dir / "seed.json").write_text(json.dumps(seed_payload), encoding="utf-8")
    _set_experiment_metadata(
        experiment.manifest_path,
        {"agentic_research_value_caps": {"model.params.num_leaves": [2, 1024]}},
    )
    experiment = research_module.get_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID)
    decision = research_module.ResearchDecision(
        action="run",
        learning="x",
        belief_update="x",
        next_hypothesis="x",
        parent_config="seed.json",
        changes=(
            research_module.ResearchChange(path="model.type", value="XGBoostRegressor", reason="switch"),
            research_module.ResearchChange(path="model.module_path", value="xgboost_model.py", reason="switch"),
            research_module.ResearchChange(path="model.params.num_leaves", value=None, reason="drop lgbm-only"),
            research_module.ResearchChange(path="model.params.max_leaves", value=8, reason="add xgb-only"),
            research_module.ResearchChange(path="model.params.min_child_weight", value=100, reason="add xgb-only"),
        ),
        stop_reason=None,
    )
    config_path = research_module._materialize_decision_config(
        experiment=experiment, round_label="r002", decision=decision
    )
    written = json.loads(config_path.read_text(encoding="utf-8"))
    assert written["model"]["type"] == "XGBoostRegressor"
    assert "num_leaves" not in written["model"].get("params", {})


def test_lgbm_num_leaves_above_depth_cap_normalized() -> None:
    payload = {
        "data": {"data_version": "v5.2"},
        "model": {"type": "LGBMRegressor", "params": {"max_depth": 5, "num_leaves": 64}},
        "training": {},
    }
    research_module._normalize_lgbm_effective_params(payload)
    assert payload["model"]["params"]["num_leaves"] == 32


def test_lgbm_num_leaves_below_cap_unchanged() -> None:
    payload = {
        "data": {"data_version": "v5.2"},
        "model": {"type": "LGBMRegressor", "params": {"max_depth": 5, "num_leaves": 16}},
        "training": {},
    }
    research_module._normalize_lgbm_effective_params(payload)
    assert payload["model"]["params"]["num_leaves"] == 16


def test_lgbm_max_depth_unlimited_skips_normalization() -> None:
    payload = {
        "data": {"data_version": "v5.2"},
        "model": {"type": "LGBMRegressor", "params": {"max_depth": -1, "num_leaves": 1024}},
        "training": {},
    }
    research_module._normalize_lgbm_effective_params(payload)
    assert payload["model"]["params"]["num_leaves"] == 1024


def test_lgbm_normalization_skips_non_lgbm_model() -> None:
    payload = {
        "data": {"data_version": "v5.2"},
        "model": {"type": "XGBRegressor", "params": {"max_depth": 5, "num_leaves": 64}},
        "training": {},
    }
    research_module._normalize_lgbm_effective_params(payload)
    assert payload["model"]["params"]["num_leaves"] == 64


def test_duplicate_via_effective_leaf_cap_rejected(tmp_path: Path) -> None:
    """A new config that collapses to an existing one via num_leaves normalization
    must be rejected by the duplicate-by-hash gate."""
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")
    configs_dir = experiment.manifest_path.parent / "configs"
    seed_payload = {
        "data": {"data_version": "v5.2", "dataset_variant": "non_downsampled"},
        "model": {"type": "LGBMRegressor", "params": {"max_depth": 5, "num_leaves": 32, "learning_rate": 0.1}},
        "training": {},
    }
    (configs_dir / "seed.json").parent.mkdir(parents=True, exist_ok=True)
    (configs_dir / "seed.json").write_text(json.dumps(seed_payload), encoding="utf-8")

    decision = research_module.ResearchDecision(
        action="run",
        learning="x",
        belief_update="x",
        next_hypothesis="x",
        parent_config="seed.json",
        changes=(research_module.ResearchChange(path="model.params.num_leaves", value=64, reason="leaves above cap"),),
        stop_reason=None,
    )
    with pytest.raises(AgenticResearchValidationError, match="agentic_research_candidate_duplicate"):
        research_module._materialize_decision_config(experiment=experiment, round_label="r002", decision=decision)


def test_duplicate_raises_typed_subclass_for_soft_skip_routing(tmp_path: Path) -> None:
    """A duplicate must raise AgenticResearchDuplicateCandidate specifically, so the
    driver routes it to the soft-skip recorder instead of the failure-bail path (item 3)."""
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")
    configs_dir = experiment.manifest_path.parent / "configs"
    _write_training_config(configs_dir / "seed.json", learning_rate=0.1)
    experiment = research_module.get_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID)
    # Re-proposing the parent's own value yields the identical config hash.
    decision = research_module.ResearchDecision(
        action="run",
        learning="x",
        belief_update="x",
        next_hypothesis="x",
        parent_config="seed.json",
        changes=(research_module.ResearchChange(path="model.params.learning_rate", value=0.1, reason="no-op core"),),
        stop_reason=None,
    )
    with pytest.raises(research_module.AgenticResearchDuplicateCandidate):
        research_module._materialize_decision_config(experiment=experiment, round_label="r002", decision=decision)


def test_record_duplicate_skip_round_does_not_count_toward_bail(tmp_path: Path) -> None:
    """The soft-skip recorder resets the consecutive-failure counter and completes the
    round, so a late-plateau cluster of duplicates can never trip the bail (item 3)."""
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")
    state: dict[str, object] = {
        "status": "running",
        "next_round_number": 5,
        "total_rounds_completed": 4,
        "failed_rounds_counter": 3,
        "pending_decision": {"action": "run", "parent_config": "config_004.json"},
    }
    result = research_module._record_duplicate_skip_round(
        experiment=experiment,
        state=state,
        error=research_module.AgenticResearchDuplicateCandidate("agentic_research_candidate_duplicate:abc123"),
    )
    assert result.status == "skipped"
    assert state["failed_rounds_counter"] == 0
    assert state["next_round_number"] == 6
    assert state["total_rounds_completed"] == 5
    assert state["status"] == "running"
    experiment_dir = experiment.manifest_path.parent
    assert (experiment_dir / "agentic_research" / "rounds" / "r005.md").is_file()
    events = {e["event"] for e in _trace_events(experiment_dir / "agentic_research" / "trace.jsonl")}
    assert "round_completed" in events


def test_tried_signatures_context_is_compact_and_capped() -> None:
    """The prompt projection drops verbose keys, rounds the metric, and caps the count
    so a deeper memory window does not re-bloat the prompt (item 4)."""
    sigs = [
        {
            "action": "run",
            "run_id": f"run-{i}",
            "family": "XGBoostRegressor",
            "feature_set": "small",
            "target": "target_alpha_20",
            "primary": 0.0039334567,
            "seed": 42,
        }
        for i in range(research_module.TRIED_SIGNATURES_CONTEXT_LIMIT + 25)
    ]
    out = research_module._tried_signatures_context({"tried_signatures": sigs})
    assert len(out) == research_module.TRIED_SIGNATURES_CONTEXT_LIMIT  # capped
    assert all("run_id" not in item and "action" not in item for item in out)  # verbose keys stripped
    assert out[0]["primary"] == 0.003933  # rounded to 6 dp


def test_ensemble_context_surfaces_rounds_since_improved() -> None:
    state = {"ensemble_rounds_since_improved": 7, "phase_plateau_counter": 3}
    ctx = research_module._ensemble_context(state=state, eligible_rows=[])
    assert ctx["rounds_since_improved"] == 7


def test_change_path_allows_model_module_path_and_predictions_name() -> None:
    assert research_module._change_path_allowed("model.module_path") is True
    assert research_module._change_path_allowed("output.predictions_name") is True


def test_parse_llm_response_accepts_five_changes_for_family_switch() -> None:
    decision = research_module._parse_llm_response(
        _llm_response(
            {
                "action": "run",
                "learning": "x",
                "belief_update": "x",
                "next_hypothesis": "x",
                "parent_config": "seed.json",
                "changes": [
                    {"path": "model.type", "value": "XGBoostRegressor", "reason": "switch"},
                    {"path": "model.module_path", "value": "xgboost_model.py", "reason": "switch"},
                    {"path": "model.params.max_leaves", "value": 8, "reason": "swap"},
                    {"path": "model.params.min_child_weight", "value": 100, "reason": "swap"},
                    {"path": "output.predictions_name", "value": "xgb_run", "reason": "label"},
                ],
                "stop_reason": None,
            }
        )
    )
    assert len(decision.decision.changes) == 5


def test_parse_llm_response_rejects_six_changes() -> None:
    with pytest.raises(AgenticResearchValidationError, match="agentic_research_change_count_invalid"):
        research_module._parse_llm_response(
            _llm_response(
                {
                    "action": "run",
                    "learning": "x",
                    "belief_update": "x",
                    "next_hypothesis": "x",
                    "parent_config": "seed.json",
                    "changes": [
                        {"path": "model.params.learning_rate", "value": 0.05, "reason": "x"},
                        {"path": "model.params.max_depth", "value": 5, "reason": "x"},
                        {"path": "model.params.num_leaves", "value": 16, "reason": "x"},
                        {"path": "model.params.n_estimators", "value": 200, "reason": "x"},
                        {"path": "model.params.colsample_bytree", "value": 0.5, "reason": "x"},
                        {"path": "model.params.subsample", "value": 0.8, "reason": "x"},
                    ],
                    "stop_reason": None,
                }
            )
        )


def test_parse_llm_response_rejects_disallowed_change_path() -> None:
    with pytest.raises(AgenticResearchValidationError, match="agentic_research_change_path_not_allowed"):
        research_module._parse_llm_response(
            _llm_response(
                {
                    "action": "run",
                    "learning": "x",
                    "belief_update": "x",
                    "next_hypothesis": "x",
                    "parent_config": "seed.json",
                    "changes": [{"path": "output.results_name", "value": "bad", "reason": "not allowed"}],
                    "stop_reason": None,
                }
            )
        )


def test_parse_llm_response_requires_decision_form() -> None:
    with pytest.raises(AgenticResearchValidationError, match="agentic_research_decision_form_missing"):
        research_module._parse_llm_response(json.dumps({"round_markdown": "# Memo"}))


def test_parse_llm_response_requires_round_markdown() -> None:
    with pytest.raises(AgenticResearchValidationError, match="agentic_research_field_missing:round_markdown"):
        research_module._parse_llm_response(
            json.dumps(
                {
                    "decision_form": {
                        "action": "run",
                        "learning": "x",
                        "belief_update": "x",
                        "next_hypothesis": None,
                        "parent_config": "seed.json",
                        "changes": [{"path": "model.params.n_estimators", "value": 200, "reason": "probe"}],
                        "stop_reason": None,
                    }
                }
            )
        )


def test_stop_action_rejected() -> None:
    """Budget-bounded design: `stop` is not a valid LLM action — the parser rejects it."""
    with pytest.raises(AgenticResearchValidationError, match="agentic_research_action_invalid"):
        research_module._parse_decision_object(
            {
                "action": "stop",
                "learning": "x",
                "belief_update": "x",
                "next_hypothesis": None,
                "parent_config": None,
                "changes": [],
                "stop_reason": "converged",
            }
        )


_SHALLOW_PHASES_CONFIG: dict[str, object] = {
    "initial_phase": "shallow",
    "shallow": {
        "value_caps": {
            "model.params.n_estimators": [25, 300],
            "model.params.max_depth": [2, 3],
        },
        "transition": {
            "min_rounds_in_phase": 2,
            "plateau_threshold": 2,
            "require_confirmed_champion": False,
        },
        "next_phase": "medium",
    },
    "medium": {
        "value_caps": {
            "model.params.n_estimators": [250, 3000],
            "model.params.max_depth": [3, 6],
        },
        "transition": {
            "min_rounds_in_phase": 10,
            "plateau_threshold": 10,
            "is_terminal": True,
        },
    },
}


def test_phase_transition_fires_when_predicate_met() -> None:
    """Plateau + successful_rounds reached → phase transitions to next_phase."""
    experiment_metadata = {"agentic_research_phases": _SHALLOW_PHASES_CONFIG}

    class _FakeExperiment:
        metadata = experiment_metadata

    state: dict[str, object] = {
        "phase": "shallow",
        "phase_round_start": 1,
        "phase_best_metric": 0.001,
        "phase_plateau_counter": 2,
        "phase_successful_rounds": 2,
        "phase_history": [],
    }
    payload = research_module._maybe_transition_phase(
        experiment=cast(object, _FakeExperiment()),
        state=state,
        round_number=3,
    )
    assert payload == {"transition": "phase_change", "from": "shallow", "to": "medium"}
    assert state["phase"] == "medium"
    assert state["phase_round_start"] == 4
    assert state["phase_best_metric"] is None
    assert state["phase_plateau_counter"] == 0
    assert state["phase_successful_rounds"] == 0
    history = state["phase_history"]
    assert isinstance(history, list) and len(history) == 1
    assert history[0]["exit_reason"] == "phase_transition"


def test_phase_transition_blocked_when_no_confirmed_champion() -> None:
    """When require_confirmed_champion=True and rounds/plateau thresholds are met
    but no champion exists, _maybe_transition_phase emits a blocked_no_champion
    payload (informational) and leaves state untouched."""
    cfg = json.loads(json.dumps(_SHALLOW_PHASES_CONFIG))
    cfg["shallow"]["transition"]["require_confirmed_champion"] = True
    experiment_metadata = {"agentic_research_phases": cfg}

    class _FakeExperiment:
        metadata = experiment_metadata

    state: dict[str, object] = {
        "phase": "shallow",
        "phase_round_start": 1,
        "phase_best_metric": 0.001,
        "phase_plateau_counter": 2,
        "phase_successful_rounds": 2,
        "phase_history": [],
        "confirmations": {},
    }
    payload = research_module._maybe_transition_phase(
        experiment=cast(object, _FakeExperiment()),
        state=state,
        round_number=3,
    )
    assert payload is not None
    assert payload["transition"] == "blocked_no_champion"
    assert payload["from"] == "shallow"
    assert payload["successful_rounds"] == 2
    assert payload["plateau"] == 2
    # State must remain unchanged - no real transition fired.
    assert state["phase"] == "shallow"
    assert state["phase_history"] == []


def test_phase_transition_returns_none_when_min_rounds_not_met() -> None:
    """Gate ordering: min_rounds/plateau check fails before the champion check,
    so a phase that hasn't burned enough rounds yet returns None (silent), not
    blocked_no_champion."""
    cfg = json.loads(json.dumps(_SHALLOW_PHASES_CONFIG))
    cfg["shallow"]["transition"]["require_confirmed_champion"] = True
    experiment_metadata = {"agentic_research_phases": cfg}

    class _FakeExperiment:
        metadata = experiment_metadata

    state: dict[str, object] = {
        "phase": "shallow",
        "phase_round_start": 1,
        "phase_best_metric": 0.001,
        "phase_plateau_counter": 0,
        "phase_successful_rounds": 0,
        "phase_history": [],
        "confirmations": {},
    }
    payload = research_module._maybe_transition_phase(
        experiment=cast(object, _FakeExperiment()),
        state=state,
        round_number=1,
    )
    assert payload is None


def test_terminal_phase_plateau_does_not_stop() -> None:
    """Budget-bounded design: a terminal phase that meets the transition predicate
    never auto-stops — it returns None and keeps exploring, leaving state intact."""
    experiment_metadata = {"agentic_research_phases": _SHALLOW_PHASES_CONFIG}

    class _FakeExperiment:
        metadata = experiment_metadata

    state: dict[str, object] = {
        "phase": "medium",
        "phase_round_start": 4,
        "phase_best_metric": 0.003,
        "phase_plateau_counter": 10,
        "phase_successful_rounds": 10,
        "phase_history": [],
    }
    payload = research_module._maybe_transition_phase(
        experiment=cast(object, _FakeExperiment()),
        state=state,
        round_number=13,
    )
    assert payload is None
    assert state.get("status") != "stopped"
    assert "stop_reason" not in state
    assert state["phase"] == "medium"
    assert state["phase_history"] == []


def test_confirmation_detected_and_recorded_structurally() -> None:
    """A round with a single model.params.random_state change against config_NNN.json
    is recorded as a confirmation attempt."""
    state: dict[str, object] = {}
    decision_payload = {
        "action": "run",
        "parent_config": "config_023.json",
        "changes": [{"path": "model.params.random_state", "value": 17, "reason": "confirm seed 17"}],
    }
    assert research_module._is_confirmation_round(decision_payload) is True
    research_module._record_confirmation_attempt(
        state=state,
        parent_config="config_023.json",
        seed=17,
        run_id="run-17",
        metric_value=0.003357,
        round_number=24,
    )
    confirmations = cast(dict[str, dict[str, object]], state["confirmations"])
    entry = confirmations["config_023.json"]
    assert 17 in cast(list[int], entry["seeds_completed"])
    assert cast(dict[str, str], entry["runs"])["17"] == "run-17"
    assert cast(dict[str, float], entry["primary_metric_by_seed"])["17"] == pytest.approx(0.003357)


def test_confirmation_not_detected_for_seed_only_against_baseline() -> None:
    """parent_config = base.json (not config_NNN.json) → not a confirmation round."""
    decision_payload = {
        "action": "run",
        "parent_config": "base.json",
        "changes": [{"path": "model.params.random_state", "value": 17, "reason": "x"}],
    }
    assert research_module._is_confirmation_round(decision_payload) is False


def test_confirmation_not_detected_when_multiple_changes() -> None:
    decision_payload = {
        "action": "run",
        "parent_config": "config_023.json",
        "changes": [
            {"path": "model.params.random_state", "value": 17, "reason": "x"},
            {"path": "model.params.learning_rate", "value": 0.05, "reason": "y"},
        ],
    }
    assert research_module._is_confirmation_round(decision_payload) is False


def test_confirmation_promotion_when_seed_trio_beats_champion() -> None:
    """Complete seeds 42/17/99 with no prior champion → confirmed_champion set."""
    state: dict[str, object] = {"phase": "shallow"}
    for seed, metric, run_id in [(42, 0.0031, "run-42"), (17, 0.0033, "run-17"), (99, 0.0032, "run-99")]:
        research_module._record_confirmation_attempt(
            state=state,
            parent_config="config_023.json",
            seed=seed,
            run_id=run_id,
            metric_value=metric,
            round_number=23 + seed,
        )
    promotion = research_module._maybe_promote_confirmation(
        state=state, parent_config="config_023.json", round_number=99
    )
    assert promotion is not None
    assert promotion["parent_config"] == "config_023.json"
    assert promotion["seed_trio_primary_mean"] == pytest.approx((0.0031 + 0.0033 + 0.0032) / 3)
    assert promotion["phase"] == "shallow"
    champion = cast(dict[str, object], state["confirmed_champion"])
    assert champion["parent_config"] == "config_023.json"
    assert champion["promoted_in_phase"] == "shallow"


def test_confirmation_no_promotion_when_below_threshold() -> None:
    """New trio mean beats champion by less than CONFIRMATION_PROMOTION_MARGIN → no promotion."""
    state: dict[str, object] = {
        "phase": "medium",
        "confirmed_champion": {
            "parent_config": "config_023.json",
            "seed_trio_primary_mean": 0.0032,
            "promoted_at_round": 25,
            "promoted_in_phase": "shallow",
        },
    }
    for seed, metric in [(42, 0.00320), (17, 0.00322), (99, 0.00321)]:
        research_module._record_confirmation_attempt(
            state=state,
            parent_config="config_055.json",
            seed=seed,
            run_id=f"run-{seed}",
            metric_value=metric,
            round_number=55,
        )
    promotion = research_module._maybe_promote_confirmation(
        state=state, parent_config="config_055.json", round_number=58
    )
    assert promotion is None
    champion = cast(dict[str, object], state["confirmed_champion"])
    assert champion["parent_config"] == "config_023.json"


def test_confirmation_promotes_above_trio_margin_below_old_threshold() -> None:
    """Regression for ADR 2026-05-31 (§7C flaw 2): a trio mean that beats the
    champion by MORE than the trio-mean margin (1.5e-4) but LESS than the old
    single-seed floor (3e-4) must now promote — the old gate wrongly rejected it."""
    margin = research_module.CONFIRMATION_PROMOTION_MARGIN
    assert margin < 3e-4  # the fix: trio margin is below the single-seed noise floor
    champion_mean = 0.0035
    challenger_mean = champion_mean + 2.0e-4  # between 1.5e-4 and 3e-4
    state: dict[str, object] = {
        "phase": "deep",
        "confirmed_champion": {"parent_config": "config_040.json", "seed_trio_primary_mean": champion_mean},
    }
    for seed in (42, 17, 99):
        research_module._record_confirmation_attempt(
            state=state,
            parent_config="config_232.json",
            seed=seed,
            run_id=f"run-{seed}",
            metric_value=challenger_mean,  # equal seeds → trio mean == challenger_mean
            round_number=232,
        )
    promotion = research_module._maybe_promote_confirmation(
        state=state, parent_config="config_232.json", round_number=233
    )
    assert promotion is not None
    assert cast(dict[str, object], state["confirmed_champion"])["parent_config"] == "config_232.json"


def test_consecutive_failures_bail_after_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After CONSECUTIVE_FAILURE_BAIL_THRESHOLD failed rounds in a row, the loop stops."""
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")
    _write_training_config(experiment.manifest_path.parent / "configs" / "seed.json")

    monkeypatch.setattr(
        research_module,
        "_safe_report",
        lambda **_: _report(_row("run-0", 0.10)),
    )

    def _fail(**_):
        raise research_module.AgenticResearchValidationError("agentic_research_test_failure")

    monkeypatch.setattr(research_module, "_call_research_llm", _fail)

    result = run_research(
        store_root=store_root,
        experiment_id=EXPERIMENT_ID,
        max_rounds=research_module.CONSECUTIVE_FAILURE_BAIL_THRESHOLD + 3,
    )
    assert result.stop_reason is not None
    assert result.stop_reason.startswith("consecutive_failures:")
    assert len(result.rounds) == research_module.CONSECUTIVE_FAILURE_BAIL_THRESHOLD

    trace = _trace_events(experiment.manifest_path.parent / "agentic_research" / "trace.jsonl")
    bail_events = [e for e in trace if e["event"] == "round_failed_with_retry_exhausted"]
    assert len(bail_events) == 1
    payload = cast(dict[str, object], bail_events[0]["payload"])
    assert payload["failed_rounds_counter"] == research_module.CONSECUTIVE_FAILURE_BAIL_THRESHOLD


def test_artifact_rotation_preserves_essential_and_prunes_others(tmp_path: Path) -> None:
    """Essential runs (leaderboard + confirmation) keep heavy artifacts; others lose them."""
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")
    _set_experiment_metadata(experiment.manifest_path, {"agentic_research_artifact_rotation": "enabled"})
    runs_root = store_root / "runs"
    essential_id = "run-essential"
    confirmation_id = "run-conf"
    non_essential_id = "run-old"
    for run_id in (essential_id, confirmation_id, non_essential_id):
        d = runs_root / run_id
        d.mkdir(parents=True)
        (d / "metrics.json").write_text("{}", encoding="utf-8")
        (d / "predictions.parquet").write_bytes(b"x" * 2_000_000)
    _set_experiment_runs(experiment.manifest_path, [essential_id, confirmation_id, non_essential_id])

    report = _report(_row(essential_id, 0.9))
    state: dict[str, object] = {
        "confirmations": {
            "config_005.json": {
                "runs": {"42": confirmation_id, "17": "missing-1", "99": "missing-2"},
            }
        },
        "tried_signatures": [],
    }
    experiment_record = research_module.get_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(research_module, "_safe_report", lambda **_: report)
        payload = research_module._rotate_run_artifacts(
            root=store_root,
            experiment=experiment_record,
            state=state,
            last_round_number=50,
        )

    assert payload is not None
    assert non_essential_id in payload["rotated_run_ids"]
    assert essential_id not in payload["rotated_run_ids"]
    assert confirmation_id not in payload["rotated_run_ids"]
    assert (runs_root / non_essential_id / "predictions.parquet").exists() is False
    assert (runs_root / non_essential_id / "metrics.json").exists() is True
    assert (runs_root / essential_id / "predictions.parquet").exists() is True
    assert (runs_root / confirmation_id / "predictions.parquet").exists() is True


def test_artifact_rotation_dry_run_does_not_delete(tmp_path: Path) -> None:
    """dry_run mode emits payload describing what WOULD rotate, but files remain intact.
    Payload includes a `targets` list so operators can audit before flipping to enabled."""
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")
    _set_experiment_metadata(experiment.manifest_path, {"agentic_research_artifact_rotation": "dry_run"})
    run_dir = store_root / "runs" / "run-x"
    run_dir.mkdir(parents=True)
    (run_dir / "predictions.parquet").write_bytes(b"x" * 2_000_000)
    (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
    _set_experiment_runs(experiment.manifest_path, ["run-x"])

    experiment_record = research_module.get_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(research_module, "_safe_report", lambda **_: None)
        payload = research_module._rotate_run_artifacts(
            root=store_root,
            experiment=experiment_record,
            state={},
            last_round_number=50,
        )

    assert payload is not None
    assert payload["mode"] == "dry_run"
    assert "run-x" in payload["rotated_run_ids"]
    assert (run_dir / "predictions.parquet").exists() is True
    targets = cast(list[str], payload["targets"])
    assert any("predictions.parquet" in t for t in targets)


def test_artifact_rotation_disabled_returns_none(tmp_path: Path) -> None:
    """Default mode (disabled) skips rotation entirely."""
    store_root = tmp_path / ".numereng"
    create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")
    run_dir = store_root / "runs" / "run-x"
    run_dir.mkdir(parents=True)
    (run_dir / "predictions.parquet").write_bytes(b"x" * 2_000_000)

    experiment_record = research_module.get_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID)

    payload = research_module._rotate_run_artifacts(
        root=store_root,
        experiment=experiment_record,
        state={},
        last_round_number=50,
    )
    assert payload is None
    assert (run_dir / "predictions.parquet").exists() is True


def test_tried_signatures_extracted_and_windowed(tmp_path: Path) -> None:
    """_extract_lgbm_signature reads config; _append_tried_signature caps the window."""
    config_path = tmp_path / "configs" / "config_001.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "data": {
                    "data_version": "v5.2",
                    "dataset_variant": "non_downsampled",
                    "target_col": "target_alpha_60",
                },
                "model": {
                    "type": "LGBMRegressor",
                    "params": {
                        "learning_rate": 0.03,
                        "n_estimators": 300,
                        "max_depth": 3,
                        "num_leaves": 8,
                        "colsample_bytree": 0.7,
                        "min_child_samples": 100,
                        "random_state": 42,
                    },
                },
                "training": {},
            }
        ),
        encoding="utf-8",
    )
    sig = research_module._extract_lgbm_signature(
        config_path=config_path,
        round_label="r001",
        run_id="run-abc",
        primary_metric=0.003079,
        action="run",
    )
    assert sig == {
        "r": "r001",
        "run_id": "run-abc",
        "action": "run",
        "primary": 0.003079,
        "family": "LGBMRegressor",
        "target": "target_alpha_60",
        "feature_set": "small",
        "lr": 0.03,
        "n": 300,
        "depth": 3,
        "leaves": 8,
        "cs": 0.7,
        "mcs": 100,
        "seed": 42,
    }

    state: dict[str, object] = {}
    for i in range(research_module.TRIED_SIGNATURES_WINDOW + 1):
        research_module._append_tried_signature(state, {"r": f"r{i:03d}"})
    sigs = cast(list[dict[str, object]], state["tried_signatures"])
    assert len(sigs) == research_module.TRIED_SIGNATURES_WINDOW
    assert sigs[0]["r"] == "r001"


def _diversification_sig(
    *,
    family: str = "LGBMRegressor",
    feature_set: str = "small",
    target: str = "target_alpha_20",
    seed: int = 42,
    r: str = "r000",
) -> dict[str, object]:
    return {
        "r": r,
        "action": "run",
        "family": family,
        "feature_set": feature_set,
        "target": target,
        "seed": seed,
    }


def test_diversification_streaks_counts_cell_and_target() -> None:
    """Consecutive same-cell discovery sigs count toward both cell and target streaks."""
    state: dict[str, object] = {"tried_signatures": []}
    for i in range(5):
        research_module._append_tried_signature(state, _diversification_sig(r=f"r{i + 1:03d}"))
    streaks = research_module._diversification_streaks(state)
    assert streaks["cell_streak"] == 5
    assert streaks["target_streak"] == 5
    assert streaks["cell"] == ("LGBMRegressor", "small", "target_alpha_20")
    assert streaks["target"] == "target_alpha_20"


def test_diversification_streaks_skip_confirmations() -> None:
    """Confirmations (seed != 42) neither count toward nor break a streak."""
    state: dict[str, object] = {"tried_signatures": []}
    for i in range(3):
        research_module._append_tried_signature(state, _diversification_sig(r=f"r{i + 1:03d}"))
    research_module._append_tried_signature(state, _diversification_sig(seed=17, r="r004"))
    research_module._append_tried_signature(state, _diversification_sig(seed=99, r="r005"))
    for i in range(3):
        research_module._append_tried_signature(state, _diversification_sig(r=f"r{i + 6:03d}"))
    streaks = research_module._diversification_streaks(state)
    assert streaks["cell_streak"] == 6
    assert streaks["target_streak"] == 6


def test_diversification_target_streak_survives_family_flip() -> None:
    """Cross-family tunneling on one target: the family flip resets the cell streak
    but target_streak keeps climbing — the exact flaw a cell-exact counter misses."""
    state: dict[str, object] = {"tried_signatures": []}
    for i in range(4):
        research_module._append_tried_signature(
            state, _diversification_sig(family="XGBoostRegressor", r=f"r{i + 1:03d}")
        )
    research_module._append_tried_signature(state, _diversification_sig(family="LGBMRegressor", r="r005"))
    streaks = research_module._diversification_streaks(state)
    assert streaks["cell_streak"] == 1
    assert streaks["target_streak"] == 5


def test_diversification_directive_fires_at_soft_threshold() -> None:
    below = {
        "cell_streak": 1,
        "target_streak": research_module.DIVERSIFICATION_SOFT_THRESHOLD - 1,
        "target": "target_alpha_60",
        "cell": None,
    }
    assert research_module._diversification_directive(below) is None
    at = {
        "cell_streak": 1,
        "target_streak": research_module.DIVERSIFICATION_SOFT_THRESHOLD,
        "target": "target_alpha_60",
        "cell": None,
    }
    directive = research_module._diversification_directive(at)
    assert directive is not None and "target_alpha_60" in directive


def test_reject_overconcentrated_discovery_blocks_at_hard() -> None:
    """A seed-42 discovery extending a target streak at the hard threshold is rejected;
    a different target is allowed; a confirmation seed is exempt."""
    state: dict[str, object] = {"tried_signatures": []}
    for i in range(research_module.DIVERSIFICATION_HARD_THRESHOLD):
        research_module._append_tried_signature(
            state, _diversification_sig(target="target_alpha_60", r=f"r{i + 1:03d}")
        )

    def cfg(target: str, seed: int = 42) -> dict[str, object]:
        return {
            "model": {"type": "LGBMRegressor", "params": {"random_state": seed}},
            "data": {"feature_set": "small", "target_col": target},
        }

    with pytest.raises(AgenticResearchValidationError, match="diversification_required"):
        research_module._reject_overconcentrated_discovery(state=state, validated_config=cfg("target_alpha_60"))
    research_module._reject_overconcentrated_discovery(state=state, validated_config=cfg("target_bravo_60"))
    research_module._reject_overconcentrated_discovery(state=state, validated_config=cfg("target_alpha_60", seed=17))


def test_normalize_xgb_effective_params_clamps_max_leaves() -> None:
    over = {"model": {"type": "XGBoostRegressor", "params": {"max_depth": 3, "max_leaves": 64}}}
    research_module._normalize_xgb_effective_params(over)
    assert over["model"]["params"]["max_leaves"] == 8  # 2**3
    within = {"model": {"type": "XGBoostRegressor", "params": {"max_depth": 4, "max_leaves": 16}}}
    research_module._normalize_xgb_effective_params(within)
    assert within["model"]["params"]["max_leaves"] == 16
    nolimit = {"model": {"type": "XGBoostRegressor", "params": {"max_depth": 0, "max_leaves": 256}}}
    research_module._normalize_xgb_effective_params(nolimit)
    assert nolimit["model"]["params"]["max_leaves"] == 256
    lgbm = {"model": {"type": "LGBMRegressor", "params": {"max_depth": 3, "max_leaves": 64}}}
    research_module._normalize_xgb_effective_params(lgbm)
    assert lgbm["model"]["params"]["max_leaves"] == 64


def test_detect_inert_change_flags_identical_metric() -> None:
    """A single-axis discovery whose metric equals the parent's seed-42 metric is inert."""
    state: dict[str, object] = {"confirmations": {"config_017.json": {"primary_metric_by_seed": {"42": 0.0032677941}}}}
    payload: dict[str, object] = {
        "parent_config": "config_017.json",
        "changes": [
            {"path": "model.params.min_child_weight", "value": 200},
            {"path": "output.predictions_name", "value": "x"},
        ],
    }
    assert research_module._detect_inert_change(state=state, decision_payload=payload, child_metric=0.0032677941) == (
        "config_017.json",
        "model.params.min_child_weight",
    )
    assert research_module._detect_inert_change(state=state, decision_payload=payload, child_metric=0.0034) is None
    multi: dict[str, object] = {
        "parent_config": "config_017.json",
        "changes": [
            {"path": "model.params.min_child_weight", "value": 200},
            {"path": "model.params.max_depth", "value": 4},
        ],
    }
    assert research_module._detect_inert_change(state=state, decision_payload=multi, child_metric=0.0032677941) is None


def test_reject_inert_change_blocks_single_axis_reprobe() -> None:
    state: dict[str, object] = {"inert_axes": {"config_017.json": ["model.params.min_child_weight"]}}

    def decision(
        paths_values: list[tuple[str, object]], parent: str = "config_017.json"
    ) -> research_module.ResearchDecision:
        changes = tuple(research_module.ResearchChange(path=p, value=v, reason="r") for p, v in paths_values)
        return research_module.ResearchDecision(
            action="run",
            learning="",
            belief_update="",
            next_hypothesis="",
            parent_config=parent,
            changes=changes,
            stop_reason=None,
        )

    with pytest.raises(AgenticResearchValidationError, match="inert_change"):
        research_module._reject_inert_change(
            state=state,
            decision=decision([("model.params.min_child_weight", 50), ("output.predictions_name", "x")]),
        )
    research_module._reject_inert_change(state=state, decision=decision([("model.params.colsample_bytree", 0.8)]))
    research_module._reject_inert_change(
        state=state,
        decision=decision([("model.params.min_child_weight", 50), ("model.params.max_depth", 4)]),
    )


def test_phase_transition_with_confirmed_champion_requirement_succeeds() -> None:
    """With require_confirmed_champion=True and a champion in the current phase, transition fires."""
    cfg = json.loads(json.dumps(_SHALLOW_PHASES_CONFIG))
    cfg["shallow"]["transition"]["require_confirmed_champion"] = True
    experiment_metadata = {"agentic_research_phases": cfg}

    class _FakeExperiment:
        metadata = experiment_metadata

    state: dict[str, object] = {
        "phase": "shallow",
        "phase_round_start": 1,
        "phase_best_metric": 0.003,
        "phase_plateau_counter": 2,
        "phase_successful_rounds": 2,
        "phase_history": [],
        "confirmations": {
            "config_023.json": {"promoted_in_phase": "shallow", "seed_trio_primary_mean": 0.003},
        },
    }
    payload = research_module._maybe_transition_phase(
        experiment=cast(object, _FakeExperiment()),
        state=state,
        round_number=3,
    )
    assert payload == {"transition": "phase_change", "from": "shallow", "to": "medium"}


def test_phase_transition_deferred_during_inflight_confirmation() -> None:
    """A mid-flight confirmation trio defers the transition so its champion is
    credited to the discovering phase rather than the next one."""
    cfg = json.loads(json.dumps(_SHALLOW_PHASES_CONFIG))
    cfg["shallow"]["transition"]["require_confirmed_champion"] = True
    experiment_metadata = {"agentic_research_phases": cfg}

    class _FakeExperiment:
        metadata = experiment_metadata

    state: dict[str, object] = {
        "phase": "shallow",
        "phase_round_start": 1,
        "phase_best_metric": 0.003,
        "phase_plateau_counter": 2,
        "phase_successful_rounds": 2,
        "phase_history": [],
        "confirmations": {
            "config_023.json": {"promoted_in_phase": "shallow", "seed_trio_primary_mean": 0.003},
            # config_040: seeds 42 + 17 done, not promoted, touched this round -> in flight
            "config_040.json": {"seeds_completed": [42, 17], "last_attempt_at_round": 3},
        },
    }
    payload = research_module._maybe_transition_phase(
        experiment=cast(object, _FakeExperiment()),
        state=state,
        round_number=3,
    )
    assert payload is not None and payload["transition"] == "deferred_inflight_confirmation"
    assert state["phase"] == "shallow"
    assert state["phase_history"] == []


def test_has_inflight_confirmation_recency_and_completion() -> None:
    base = {
        "phase": "shallow",
        "confirmations": {"c.json": {"seeds_completed": [42, 17], "last_attempt_at_round": 10}},
    }
    assert research_module._has_inflight_confirmation(base, round_number=10) is True
    assert research_module._has_inflight_confirmation(base, round_number=11) is True  # within recency window
    assert research_module._has_inflight_confirmation(base, round_number=13) is False  # stale partial unblocks
    promoted = {"confirmations": {"c.json": {"seeds_completed": [42, 17, 99], "last_attempt_at_round": 10}}}
    assert research_module._has_inflight_confirmation(promoted, round_number=10) is False  # full trio not in flight
    done = {
        "confirmations": {"c.json": {"seeds_completed": [42, 17], "promoted_at_round": 9, "last_attempt_at_round": 10}}
    }
    assert research_module._has_inflight_confirmation(done, round_number=10) is False  # already promoted


def test_inflight_confirmation_ignores_discovery_seed42_autocredit() -> None:
    """A recent seed-42-only entry whose metric is below the champion trio mean is
    an ordinary discovery auto-credit, NOT an in-flight trio.

    Reproduces the production lock: every discovery round auto-credits seed 42,
    so without this guard the phase transition would defer on every round."""
    state: dict[str, object] = {
        # Champion trio mean (0.0035) is the comparand; its luckiest seed (0.00393) is not.
        "confirmed_champion": {"parent_config": "config_040.json", "seed_trio_primary_mean": 0.0035},
        "confirmations": {
            "config_040.json": {
                "seeds_completed": [42, 17, 99],
                "promoted_at_round": 42,
                "primary_metric_by_seed": {"42": 0.00393},
            },
            "config_070.json": {  # recent discovery below the champion trio mean
                "seeds_completed": [42],
                "last_attempt_at_round": 70,
                "primary_metric_by_seed": {"42": 0.0034},
            },
        },
    }
    assert research_module._has_inflight_confirmation(state, round_number=71) is False


def test_inflight_confirmation_defers_for_above_trio_mean_seed42_candidate() -> None:
    """A recent seed-42-only entry that beats the champion's TRIO MEAN is a real
    champion candidate about to be confirmed — defer so it is credited to its
    discovering phase. Uses the real r232 scenario: seed-42 0.00397 sits above the
    champion trio mean (0.0035) but BELOW the champion's seed-42 (0.00393) + the old
    3e-4 bar (0.00423) — under the old comparand this challenger was wrongly ignored."""
    state: dict[str, object] = {
        "confirmed_champion": {"parent_config": "config_040.json", "seed_trio_primary_mean": 0.0035},
        "confirmations": {
            "config_040.json": {
                "seeds_completed": [42, 17, 99],
                "promoted_at_round": 42,
                "primary_metric_by_seed": {"42": 0.00393},
            },
            "config_232.json": {  # seed-42 above champion trio mean → a real candidate
                "seeds_completed": [42],
                "last_attempt_at_round": 72,
                "primary_metric_by_seed": {"42": 0.00397},
            },
        },
    }
    assert research_module._has_inflight_confirmation(state, round_number=72) is True


def test_phase_transition_proceeds_despite_discovery_seed42_autocredits() -> None:
    """End-to-end: at a met-threshold boundary whose confirmations are all
    sub-threshold discovery seed-42 auto-credits (the round-71 production
    scenario), the transition must FIRE rather than defer forever."""
    cfg = json.loads(json.dumps(_SHALLOW_PHASES_CONFIG))
    experiment_metadata = {"agentic_research_phases": cfg}

    class _FakeExperiment:
        metadata = experiment_metadata

    state: dict[str, object] = {
        "phase": "shallow",
        "phase_round_start": 1,
        "phase_best_metric": 0.00393,
        "phase_plateau_counter": 2,
        "phase_successful_rounds": 2,
        "phase_history": [],
        "confirmed_champion": {"parent_config": "config_040.json", "seed_trio_primary_mean": 0.0035},
        "confirmations": {
            "config_040.json": {
                "seeds_completed": [42, 17, 99],
                "promoted_in_phase": "shallow",
                "promoted_at_round": 2,
                "primary_metric_by_seed": {"42": 0.00393},
            },
            "config_006.json": {  # seed-42 below the champion trio mean → not a candidate
                "seeds_completed": [42],
                "last_attempt_at_round": 6,
                "primary_metric_by_seed": {"42": 0.0034},
            },
        },
    }
    payload = research_module._maybe_transition_phase(
        experiment=cast(object, _FakeExperiment()),
        state=state,
        round_number=7,
    )
    assert payload == {"transition": "phase_change", "from": "shallow", "to": "medium"}
    assert state["phase"] == "medium"
    history = state["phase_history"]
    assert isinstance(history, list) and len(history) == 1
    assert history[0]["exit_reason"] == "phase_transition"


def test_phase_transition_no_op_when_next_phase_missing() -> None:
    """A non-terminal phase missing next_phase returns a misconfigured payload and leaves state unchanged."""
    cfg = json.loads(json.dumps(_SHALLOW_PHASES_CONFIG))
    cfg["shallow"].pop("next_phase", None)
    experiment_metadata = {"agentic_research_phases": cfg}

    class _FakeExperiment:
        metadata = experiment_metadata

    state: dict[str, object] = {
        "phase": "shallow",
        "phase_round_start": 1,
        "phase_best_metric": 0.001,
        "phase_plateau_counter": 2,
        "phase_successful_rounds": 2,
        "phase_history": [],
    }
    first = research_module._maybe_transition_phase(
        experiment=cast(object, _FakeExperiment()),
        state=state,
        round_number=3,
    )
    second = research_module._maybe_transition_phase(
        experiment=cast(object, _FakeExperiment()),
        state=state,
        round_number=4,
    )
    assert first is not None and first["transition"] == "misconfigured"
    assert second is not None and second["transition"] == "misconfigured"
    assert state["phase"] == "shallow"
    assert state["phase_history"] == []


def test_phase_transition_records_best_run_id_in_history() -> None:
    """Phase transition history record includes best_run_id derived from the report."""
    experiment_metadata = {"agentic_research_phases": _SHALLOW_PHASES_CONFIG}

    class _FakeExperiment:
        metadata = experiment_metadata

    state: dict[str, object] = {
        "phase": "shallow",
        "phase_round_start": 1,
        "phase_best_metric": 0.123,
        "phase_plateau_counter": 2,
        "phase_successful_rounds": 2,
        "phase_history": [],
    }
    report = _report(_row("run-best", 0.123), _row("run-other", 0.05))
    payload = research_module._maybe_transition_phase(
        experiment=cast(object, _FakeExperiment()),
        state=state,
        round_number=3,
        report=report,
    )
    assert payload == {"transition": "phase_change", "from": "shallow", "to": "medium"}
    history = cast(list[dict[str, object]], state["phase_history"])
    assert history[0]["best_run_id"] == "run-best"


def test_update_phase_progress_skips_none_metric() -> None:
    """Missing metric must not advance plateau counter or successful_rounds."""
    state: dict[str, object] = {
        "phase": "shallow",
        "phase_plateau_counter": 0,
        "phase_successful_rounds": 0,
        "phase_best_metric": None,
    }
    research_module._update_phase_progress(state=state, metric_value=None)
    assert state["phase_plateau_counter"] == 0
    assert state["phase_successful_rounds"] == 0
    assert state["phase_best_metric"] is None


def test_update_phase_progress_counts_only_successful_rounds() -> None:
    """Every successful `run` round increments both successful_rounds and plateau.
    Plateau resets are owned by _reset_plateau_on_champion_promotion, not by per-round
    metric comparisons — see PROGRAM.md "Plateau And Progress Semantics"."""
    state: dict[str, object] = {"phase": "shallow"}
    research_module._update_phase_progress(state=state, metric_value=0.5, action="run")
    research_module._update_phase_progress(state=state, metric_value=0.5, action="run")
    assert state["phase_successful_rounds"] == 2
    assert state["phase_plateau_counter"] == 2


def test_update_phase_progress_baseline_does_not_tick_plateau() -> None:
    """Baseline establishes the incumbent rather than failing to improve on it,
    so it counts as a successful round but does not tick the plateau counter."""
    state: dict[str, object] = {"phase": "shallow"}
    research_module._update_phase_progress(state=state, metric_value=0.001, action="baseline")
    research_module._update_phase_progress(state=state, metric_value=0.002, action="run")
    assert state["phase_successful_rounds"] == 2
    assert state["phase_plateau_counter"] == 1


def test_reset_plateau_on_champion_promotion_zeroes_counter_and_updates_best() -> None:
    state: dict[str, object] = {"phase": "shallow", "phase_plateau_counter": 7, "phase_best_metric": 0.001}
    research_module._reset_plateau_on_champion_promotion(state=state, new_trio_mean=0.0032)
    assert state["phase_plateau_counter"] == 0
    assert state["phase_best_metric"] == 0.0032


def test_plateau_parity_replays_diversification_test_sequence() -> None:
    """Golden parity test using the r014-r020 sequence from the 2026-05-27
    wide-diversification-test experiment. Confirms that under documented
    semantics:
      * each `run` round ticks plateau by 1,
      * confirmation rounds tick plateau,
      * champion promotions zero plateau and refresh phase_best_metric,
      * baseline rounds do not tick plateau.
    Numbers come from the actual round-md output and state.json of that run."""
    state: dict[str, object] = {"phase": "shallow", "phase_plateau_counter": 4}

    # r014: discovery (LGBM Alpha60 seed=42). Plateau ticks (no promotion yet).
    research_module._update_phase_progress(state=state, metric_value=0.002581, action="run")
    assert state["phase_plateau_counter"] == 5

    # r015: confirmation seed=17.
    research_module._update_phase_progress(state=state, metric_value=0.002804, action="run")
    assert state["phase_plateau_counter"] == 6

    # r016: confirmation seed=99 closes the trio; promotion fires. Plateau resets.
    research_module._update_phase_progress(state=state, metric_value=0.003231, action="run")
    assert state["phase_plateau_counter"] == 7
    research_module._reset_plateau_on_champion_promotion(state=state, new_trio_mean=0.002872)
    assert state["phase_plateau_counter"] == 0
    assert state["phase_best_metric"] == pytest.approx(0.002872)

    # r017: discovery (XGB Alpha60 seed=42). Single-seed score above phase_best
    # but no promotion yet — plateau still ticks.
    research_module._update_phase_progress(state=state, metric_value=0.003268, action="run")
    assert state["phase_plateau_counter"] == 1

    # r018: confirmation seed=17. Still no promotion — plateau ticks.
    research_module._update_phase_progress(state=state, metric_value=0.003347, action="run")
    assert state["phase_plateau_counter"] == 2

    # r019: confirmation seed=99 closes the second trio. Promotion fires.
    # New trio mean 0.003224 beats prior 0.002872 by > PHASE_IMPROVEMENT_THRESHOLD.
    research_module._update_phase_progress(state=state, metric_value=0.003059, action="run")
    assert state["phase_plateau_counter"] == 3
    research_module._reset_plateau_on_champion_promotion(state=state, new_trio_mean=0.003224)
    assert state["phase_plateau_counter"] == 0
    assert state["phase_best_metric"] == pytest.approx(0.003224)

    # r020: post-promotion discovery (XGB Bravo60). No promotion — plateau ticks.
    research_module._update_phase_progress(state=state, metric_value=0.000987, action="run")
    assert state["phase_plateau_counter"] == 1


def test_run_research_continues_after_training_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-AgenticResearchError from train_experiment is caught and recorded as a failed round."""
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")
    _write_training_config(experiment.manifest_path.parent / "configs" / "seed.json")

    monkeypatch.setattr(
        research_module,
        "_safe_report",
        lambda **_: _report(_row("run-0", 0.10)),
    )
    monkeypatch.setattr(
        research_module,
        "_call_research_llm",
        lambda **_: (
            _llm_response(
                {
                    "action": "run",
                    "learning": "x",
                    "belief_update": "x",
                    "next_hypothesis": "x",
                    "parent_config": "seed.json",
                    "changes": [{"path": "model.params.learning_rate", "value": 0.02, "reason": "x"}],
                    "stop_reason": None,
                },
            ),
            "test",
        ),
    )

    def _boom(**_: object) -> object:
        raise RuntimeError("transient training failure")

    monkeypatch.setattr(research_module, "train_experiment", _boom)

    # One round: the training failure must be recorded and tick the failure counter.
    # (A 2nd round here would re-propose the identical config, which now soft-skips as a
    # duplicate and resets the counter — a separate, intended behavior tested elsewhere.)
    result = run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=1)
    assert result.rounds[0].status == "failed"
    state_path = experiment.manifest_path.parent / "agentic_research" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["failed_rounds_counter"] >= 1


def test_run_research_catches_codex_executable_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If codex is missing, FileNotFoundError is wrapped and the round is recorded as failed."""
    store_root = tmp_path / ".numereng"
    create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")

    monkeypatch.setattr(
        research_module,
        "_safe_report",
        lambda **_: _report(_row("run-0", 0.10)),
    )

    def _missing(**_: object) -> tuple[str, str]:
        raise FileNotFoundError("codex")

    monkeypatch.setattr(research_module, "_call_research_llm", _missing)

    result = run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=1)
    assert result.rounds[0].status == "failed"
    assert "codex" in (result.rounds[0].learning or "").lower()


def test_record_confirmation_attempt_rejects_bool_seed() -> None:
    """Boolean random_state must not be recorded as seed=1 (Python bool is subclass of int)."""
    decision_payload: dict[str, object] = {
        "action": "run",
        "parent_config": "config_023.json",
        "changes": [{"path": "model.params.random_state", "value": True, "reason": "buggy"}],
    }
    assert research_module._is_confirmation_round(decision_payload) is True

    seed_value = cast(list[dict[str, object]], decision_payload["changes"])[0]["value"]
    assert isinstance(seed_value, int) is True
    assert isinstance(seed_value, bool) is True


def test_is_confirmation_round_excludes_config_001() -> None:
    """config_001.json (baseline) is not a valid confirmation parent."""
    decision_payload = {
        "action": "run",
        "parent_config": "config_001.json",
        "changes": [{"path": "model.params.random_state", "value": 17, "reason": "confirm"}],
    }
    assert research_module._is_confirmation_round(decision_payload) is False


def test_config_random_state_handles_missing_and_malformed(tmp_path: Path) -> None:
    """The helper returns None for missing/malformed configs without raising."""
    missing = tmp_path / "missing.json"
    assert research_module._config_random_state(missing) is None

    schema_invalid = tmp_path / "schema_invalid.json"
    schema_invalid.write_text("{}", encoding="utf-8")
    assert research_module._config_random_state(schema_invalid) is None

    valid = tmp_path / "valid.json"
    valid.write_text(
        json.dumps(
            {
                "data": {"data_version": "v5.2", "dataset_variant": "non_downsampled"},
                "model": {"type": "LGBMRegressor", "params": {"learning_rate": 0.01, "random_state": 42}},
                "training": {},
            }
        ),
        encoding="utf-8",
    )
    assert research_module._config_random_state(valid) == 42


def test_discovery_seed_auto_credits_canonical_random_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A discovery round that materializes a config with random_state in CANONICAL_SEED_TRIO
    must record that seed against the generated config so the trio can eventually complete."""
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")
    _write_training_config(experiment.manifest_path.parent / "configs" / "seed.json")

    monkeypatch.setattr(research_module, "_safe_report", lambda **_: _report(_row("run-disc", 0.0025)))
    monkeypatch.setattr(
        research_module,
        "_call_research_llm",
        lambda **_: (
            _llm_response(
                {
                    "action": "run",
                    "learning": "x",
                    "belief_update": "x",
                    "next_hypothesis": "x",
                    "parent_config": "seed.json",
                    "changes": [
                        {"path": "model.params.learning_rate", "value": 0.07, "reason": "x"},
                        {"path": "model.params.random_state", "value": 42, "reason": "discovery seed"},
                    ],
                    "stop_reason": None,
                },
            ),
            "test",
        ),
    )
    monkeypatch.setattr(
        research_module,
        "train_experiment",
        lambda **_: ExperimentTrainResult(
            experiment_id=EXPERIMENT_ID,
            run_id="run-disc",
            predictions_path=store_root / "runs" / "run-disc" / "predictions.parquet",
            results_path=store_root / "runs" / "run-disc" / "results.json",
        ),
    )
    monkeypatch.setattr(
        research_module,
        "score_experiment_round",
        lambda **_: ExperimentScoreRoundResult(
            experiment_id=EXPERIMENT_ID, round="r001", stage="post_training_core", run_ids=("run-disc",)
        ),
    )

    run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=1)

    state = json.loads(
        (experiment.manifest_path.parent / "agentic_research" / "state.json").read_text(encoding="utf-8")
    )
    entry = state["confirmations"]["config_001.json"]
    assert entry["seeds_completed"] == [42]
    assert entry["runs"] == {"42": "run-disc"}
    assert entry["primary_metric_by_seed"]["42"] == pytest.approx(0.0025)

    trace = _trace_events(experiment.manifest_path.parent / "agentic_research" / "trace.jsonl")
    auto_credit = [e for e in trace if e["event"] == "discovery_seed_auto_credited"]
    assert len(auto_credit) == 1
    payload = cast(dict[str, object], auto_credit[0]["payload"])
    assert payload["generated_config"] == "config_001.json"
    assert payload["seed"] == 42


def test_discovery_seed_skips_non_canonical_random_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A discovery round with random_state=123 (not in trio) must not record a confirmation entry."""
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")
    _write_training_config(experiment.manifest_path.parent / "configs" / "seed.json")

    monkeypatch.setattr(research_module, "_safe_report", lambda **_: _report(_row("run-x", 0.001)))
    monkeypatch.setattr(
        research_module,
        "_call_research_llm",
        lambda **_: (
            _llm_response(
                {
                    "action": "run",
                    "learning": "x",
                    "belief_update": "x",
                    "next_hypothesis": "x",
                    "parent_config": "seed.json",
                    "changes": [
                        {"path": "model.params.random_state", "value": 123, "reason": "non-canonical"},
                    ],
                    "stop_reason": None,
                },
            ),
            "test",
        ),
    )
    monkeypatch.setattr(
        research_module,
        "train_experiment",
        lambda **_: ExperimentTrainResult(
            experiment_id=EXPERIMENT_ID,
            run_id="run-x",
            predictions_path=store_root / "runs" / "run-x" / "predictions.parquet",
            results_path=store_root / "runs" / "run-x" / "results.json",
        ),
    )
    monkeypatch.setattr(
        research_module,
        "score_experiment_round",
        lambda **_: ExperimentScoreRoundResult(
            experiment_id=EXPERIMENT_ID, round="r001", stage="post_training_core", run_ids=("run-x",)
        ),
    )

    run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=1)

    state = json.loads(
        (experiment.manifest_path.parent / "agentic_research" / "state.json").read_text(encoding="utf-8")
    )
    assert state.get("confirmations", {}) == {}
    trace = _trace_events(experiment.manifest_path.parent / "agentic_research" / "trace.jsonl")
    assert not any(e["event"] == "discovery_seed_auto_credited" for e in trace)


def test_confirmation_round_does_not_auto_credit_double(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A confirmation round routes through the existing confirmation_attempt path only;
    the auto-credit branch must not fire (it is `elif action == 'run'` AFTER the if-confirmation)."""
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")
    configs_dir = experiment.manifest_path.parent / "configs"
    _write_training_config(configs_dir / "seed.json")
    _write_training_config(configs_dir / "config_005.json", learning_rate=0.07)

    monkeypatch.setattr(research_module, "_safe_report", lambda **_: _report(_row("run-conf", 0.003)))
    monkeypatch.setattr(
        research_module,
        "_call_research_llm",
        lambda **_: (
            _llm_response(
                {
                    "action": "run",
                    "learning": "confirm",
                    "belief_update": "confirm",
                    "next_hypothesis": "confirm",
                    "parent_config": "config_005.json",
                    "changes": [{"path": "model.params.random_state", "value": 17, "reason": "trio"}],
                    "stop_reason": None,
                },
            ),
            "test",
        ),
    )
    monkeypatch.setattr(
        research_module,
        "train_experiment",
        lambda **_: ExperimentTrainResult(
            experiment_id=EXPERIMENT_ID,
            run_id="run-conf",
            predictions_path=store_root / "runs" / "run-conf" / "predictions.parquet",
            results_path=store_root / "runs" / "run-conf" / "results.json",
        ),
    )
    monkeypatch.setattr(
        research_module,
        "score_experiment_round",
        lambda **_: ExperimentScoreRoundResult(
            experiment_id=EXPERIMENT_ID, round="r001", stage="post_training_core", run_ids=("run-conf",)
        ),
    )

    run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=1)

    state = json.loads(
        (experiment.manifest_path.parent / "agentic_research" / "state.json").read_text(encoding="utf-8")
    )
    # Only one entry — under the parent, not under the generated config_001.json
    assert list(state["confirmations"].keys()) == ["config_005.json"]
    assert state["confirmations"]["config_005.json"]["seeds_completed"] == [17]
    trace = _trace_events(experiment.manifest_path.parent / "agentic_research" / "trace.jsonl")
    assert not any(e["event"] == "discovery_seed_auto_credited" for e in trace)


def test_auto_credit_plus_confirmations_promotes_champion(tmp_path: Path) -> None:
    """End-to-end: discovery(seed=42) auto-credits, then seed-17 and seed-99 confirmations
    complete the trio and `_maybe_promote_confirmation` populates `confirmed_champion`."""
    state: dict[str, object] = {"phase": "shallow"}
    # Simulate discovery auto-credit
    research_module._record_confirmation_attempt(
        state=state,
        parent_config="config_021.json",
        seed=42,
        run_id="run-disc",
        metric_value=0.0025,
        round_number=21,
    )
    # Simulate seed-17 confirmation
    research_module._record_confirmation_attempt(
        state=state,
        parent_config="config_021.json",
        seed=17,
        run_id="run-c17",
        metric_value=0.0027,
        round_number=22,
    )
    promo_after_17 = research_module._maybe_promote_confirmation(
        state=state, parent_config="config_021.json", round_number=22
    )
    assert promo_after_17 is None  # trio incomplete
    # Simulate seed-99 confirmation
    research_module._record_confirmation_attempt(
        state=state,
        parent_config="config_021.json",
        seed=99,
        run_id="run-c99",
        metric_value=0.0030,
        round_number=23,
    )
    promo_after_99 = research_module._maybe_promote_confirmation(
        state=state, parent_config="config_021.json", round_number=23
    )
    assert promo_after_99 is not None
    assert promo_after_99["parent_config"] == "config_021.json"
    assert promo_after_99["seed_trio_primary_mean"] == pytest.approx((0.0025 + 0.0027 + 0.0030) / 3)
    champion = cast(dict[str, object], state["confirmed_champion"])
    assert champion["parent_config"] == "config_021.json"
    assert set(cast(dict[str, str], champion["runs"]).keys()) == {"42", "17", "99"}


def test_latest_round_markdown_sorts_by_integer(tmp_path: Path) -> None:
    """Round numbering above 999 must be sorted numerically, not lexicographically."""
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")
    rounds_dir = experiment.manifest_path.parent / "agentic_research" / "rounds"
    rounds_dir.mkdir(parents=True)
    (rounds_dir / "r999.md").write_text("nine-nine-nine", encoding="utf-8")
    (rounds_dir / "r1000.md").write_text("thousand", encoding="utf-8")
    assert research_module._latest_round_markdown(experiment) == "thousand"


def test_write_experiment_markdown_is_atomic(tmp_path: Path) -> None:
    """Writing creates no leftover .tmp file on success and lands fully or not at all."""
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")
    research_module._write_experiment_markdown(experiment, "# Curated\nBody.\n")
    path = experiment.manifest_path.parent / "EXPERIMENT.md"
    assert path.read_text(encoding="utf-8") == "# Curated\nBody.\n"
    leftover = list(path.parent.glob(".EXPERIMENT.md.tmp"))
    assert leftover == []


def test_build_context_surfaces_canonical_seed_trio(tmp_path: Path) -> None:
    """Context dict includes canonical_seed_trio so the LLM picks the right seeds."""
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")
    context = research_module._build_context(
        root=store_root,
        experiment=experiment,
        report=None,
        state={"phase": "shallow"},
        eligible_ensemble_rows=[],
    )
    assert context["canonical_seed_trio"] == list(research_module.CANONICAL_SEED_TRIO)


def test_state_context_drops_unbounded_collections_keeps_program_fields() -> None:
    """The prompt's `state` must not carry the per-round-growing collections.

    `confirmations` grows one entry per config forever and was the unbounded term that
    blew the codex prompt on long runs; `tried_signatures` is the 100-entry dedup
    window. Both are surfaced curated as their own top-level context keys, so the raw
    `state` copy is pure duplication. Programs read only scalar fields + phase_history,
    which must survive."""
    state = {
        "phase": "deep",
        "next_round_number": 522,
        "total_rounds_completed": 521,
        "phase_history": [{"phase": "shallow"}],
        "confirmations": {f"config_{i:03d}.json": {"runs": {}} for i in range(481)},
        "tried_signatures": [f"sig-{i}" for i in range(100)],
    }
    curated = research_module._state_context(state)
    for field in ("phase", "next_round_number", "total_rounds_completed", "phase_history"):
        assert field in curated, field
    assert "confirmations" not in curated
    assert "tried_signatures" not in curated


def test_build_context_prompt_size_bounded_as_confirmations_grow(tmp_path: Path) -> None:
    """Rendered prompt size must stay flat as confirmations accumulate over a long run.

    Regression for the r521 codex stream-disconnect: the raw `state` dump shipped every
    confirmation entry, so the prompt grew without bound. With the curated `state`, a
    state holding 1000 confirmations must render no larger than one holding 10."""
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")

    def _state(n: int) -> dict[str, object]:
        return {
            "phase": "deep",
            "next_round_number": n + 1,
            "total_rounds_completed": n,
            "confirmations": {
                f"config_{i:03d}.json": {"runs": {"42": {"run_id": f"run{i}"}}, "last_attempt_at_round": i}
                for i in range(n)
            },
        }

    def _prompt_len(n: int) -> int:
        context = research_module._build_context(
            root=store_root, experiment=experiment, report=None, state=_state(n), eligible_ensemble_rows=[]
        )
        return len(research_module._render_prompt(context))

    small, large = _prompt_len(10), _prompt_len(1000)
    # Curated confirmations is capped at 30 entries; 1000 vs 10 must not blow the prompt.
    assert large - small < 10_000, (small, large)


def test_config_context_bounds_recent_plus_seed_and_champion(tmp_path: Path) -> None:
    """The configs menu must stay bounded as configs accumulate over a long run.

    Regression for the r521 codex stream-disconnect: the configs list dumped every
    generated config (~500 = ~500 KB, the prompt's single largest term). It must now
    surface only the seed, the confirmed champion (however old), and the most recent
    CONFIG_CONTEXT_RECENT generated configs."""
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")
    config_dir = experiment.manifest_path.parent / "configs"
    _write_training_config(config_dir / "seed.json")
    total = research_module.CONFIG_CONTEXT_RECENT + 25
    for i in range(1, total + 1):
        _write_training_config(config_dir / f"config_{i:03d}.json", learning_rate=0.001 * i)
    champion_name = "config_002.json"  # old champion, far outside the recent window
    state = {"confirmed_champion": {"parent_config": champion_name}}

    context = research_module._config_context(experiment, state=state)
    names = {item["filename"] for item in context}

    # Bounded: recent window + seed + champion, never all `total` generated configs.
    assert len(context) <= research_module.CONFIG_CONTEXT_RECENT + 2
    assert "seed.json" in names  # seeds always kept
    assert champion_name in names  # champion kept even though it is old
    assert f"config_{total:03d}.json" in names  # newest kept
    assert "config_010.json" not in names  # an old non-champion generated config is dropped


def test_experiment_markdown_not_updated_when_round_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If materialization fails, EXPERIMENT.md must not be overwritten with the unrun hypothesis."""
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")
    _set_experiment_metadata(experiment.manifest_path, {"agentic_research_phases": _SHALLOW_PHASES_CONFIG})
    _write_training_config(experiment.manifest_path.parent / "configs" / "seed.json")
    md_path = experiment.manifest_path.parent / "EXPERIMENT.md"
    prior = "# Prior\nUntouched.\n"
    md_path.write_text(prior, encoding="utf-8")

    monkeypatch.setattr(
        research_module,
        "_safe_report",
        lambda **_: _report(_row("run-0", 0.10)),
    )
    monkeypatch.setattr(
        research_module,
        "_call_research_llm",
        lambda **_: (
            _llm_response(
                {
                    "action": "run",
                    "learning": "Caps-violating probe.",
                    "belief_update": "x",
                    "next_hypothesis": "x",
                    "parent_config": "seed.json",
                    "changes": [{"path": "model.params.n_estimators", "value": 5000, "reason": "out of range"}],
                    "stop_reason": None,
                },
                experiment_markdown="# Hypothetical\nShould NOT be persisted.\n",
            ),
            "test",
        ),
    )

    result = run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=1)
    assert result.rounds[0].status == "failed"
    assert md_path.read_text(encoding="utf-8") == prior


def test_artifact_rotation_does_not_touch_other_experiments(tmp_path: Path) -> None:
    """Rotation enabled on experiment A must not delete run dirs owned by experiment B."""
    store_root = tmp_path / ".numereng"
    exp_a_id = "2026-02-22_exp-a"
    exp_b_id = "2026-02-22_exp-b"
    experiment_a = create_experiment(store_root=store_root, experiment_id=exp_a_id, name="A")
    experiment_b = create_experiment(store_root=store_root, experiment_id=exp_b_id, name="B")
    _set_experiment_metadata(experiment_a.manifest_path, {"agentic_research_artifact_rotation": "enabled"})

    runs_root = store_root / "runs"
    a_run = "run-a"
    b_run = "run-b"
    for run_id in (a_run, b_run):
        d = runs_root / run_id
        d.mkdir(parents=True)
        (d / "predictions.parquet").write_bytes(b"x" * 2_000_000)
        (d / "metrics.json").write_text("{}", encoding="utf-8")
    _set_experiment_runs(experiment_a.manifest_path, [a_run])
    _set_experiment_runs(experiment_b.manifest_path, [b_run])

    experiment_a_record = research_module.get_experiment(store_root=store_root, experiment_id=exp_a_id)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(research_module, "_safe_report", lambda **_: None)
        payload = research_module._rotate_run_artifacts(
            root=store_root,
            experiment=experiment_a_record,
            state={},
            last_round_number=50,
        )

    assert payload is not None
    assert a_run in payload["rotated_run_ids"]
    assert b_run not in payload["rotated_run_ids"]
    assert (runs_root / a_run / "predictions.parquet").exists() is False
    assert (runs_root / b_run / "predictions.parquet").exists() is True


def test_artifact_rotation_grace_window_protects_recent_signatures(tmp_path: Path) -> None:
    """tried_signatures within the 10-round grace are essential; older ones are eligible to rotate.

    Long-burn correctness: as the loop runs, signatures pile up. Once a signature's round falls
    outside `ARTIFACT_ROTATION_RECENT_ROUND_GRACE` of the latest round AND its run is not
    otherwise referenced (no leaderboard row, no confirmation, no champion), its heavy artifacts
    are eligible for rotation. This is the only path that actually frees space during a long burn.
    """
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")
    _set_experiment_metadata(experiment.manifest_path, {"agentic_research_artifact_rotation": "enabled"})

    runs_root = store_root / "runs"
    recent_run = "run-recent-r048"
    old_run = "run-old-r010"
    for run_id in (recent_run, old_run):
        d = runs_root / run_id
        d.mkdir(parents=True)
        (d / "predictions.parquet").write_bytes(b"x" * 2_000_000)
        (d / "metrics.json").write_text("{}", encoding="utf-8")
    _set_experiment_runs(experiment.manifest_path, [recent_run, old_run])

    last_round = 50
    state: dict[str, object] = {
        "tried_signatures": [
            {"r": "r010", "run_id": old_run, "primary": 0.1},
            {"r": "r048", "run_id": recent_run, "primary": 0.2},
        ],
    }
    experiment_record = research_module.get_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(research_module, "_safe_report", lambda **_: None)
        payload = research_module._rotate_run_artifacts(
            root=store_root,
            experiment=experiment_record,
            state=state,
            last_round_number=last_round,
        )

    grace_start = last_round - research_module.ARTIFACT_ROTATION_RECENT_ROUND_GRACE + 1
    assert grace_start == 41  # sanity-check the grace math

    assert payload is not None
    assert old_run in payload["rotated_run_ids"]
    assert recent_run not in payload["rotated_run_ids"]
    assert (runs_root / old_run / "predictions.parquet").exists() is False
    assert (runs_root / recent_run / "predictions.parquet").exists() is True
    assert (runs_root / old_run / "metrics.json").exists() is True  # preserve sentinel kept


def test_load_secondary_metrics_flattens_canonical_keys(tmp_path: Path) -> None:
    run_id = "run-secondary"
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "bmc_last_200_eras": {"mean": 0.0035},
                "bmc": {"mean": 0.0046},
                "corr": {"mean": 0.0158},
                "mmc": {"mean": -0.0001},
                "cwmm": {"mean": 0.2658},
            }
        ),
        encoding="utf-8",
    )
    result = research_module._load_secondary_metrics_from_disk(root=tmp_path, run_id=run_id)
    assert result == {
        "bmc_mean": 0.0046,
        "corr_mean": 0.0158,
        "mmc_mean": -0.0001,
        "cwmm_mean": 0.2658,
    }


def test_load_secondary_metrics_returns_empty_when_metrics_missing(tmp_path: Path) -> None:
    assert research_module._load_secondary_metrics_from_disk(root=tmp_path, run_id="ghost") == {}
    assert research_module._load_secondary_metrics_from_disk(root=tmp_path, run_id=None) == {}


def test_round_md_includes_wall_time_and_secondary_metrics_block(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "rounds"
    artifact_dir.mkdir()
    research_module._write_llm_round_markdown(
        artifact_dir=artifact_dir,
        round_label="r042",
        round_markdown="# r042 Research State\n\nBody.",
        round_payload={
            "action": "run",
            "status": "completed",
            "run_id": "abc123",
            "config_path": "configs/config_042.json",
            "metric_value": 0.00374,
            "completed_at": "2026-05-19T00:00:00+00:00",
            "wall_time_seconds": 138.34,
            "secondary_metrics": {
                "bmc_mean": 0.0046,
                "corr_mean": 0.0158,
                "mmc_mean": -0.0001,
                "cwmm_mean": 0.2658,
            },
        },
    )
    md = (artifact_dir / "r042.md").read_text(encoding="utf-8")
    assert "- Wall time: 138.3s" in md
    assert "## Secondary Metrics" in md
    assert "- bmc_mean: 0.0046" in md
    assert "- corr_mean: 0.0158" in md
    assert "- mmc_mean: -0.0001" in md
    assert "- cwmm_mean: 0.2658" in md


def test_render_diff_vs_parent_shows_old_and_new_values(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")
    parent_path = experiment.manifest_path.parent / "configs" / "parent.json"
    _write_training_config(parent_path, learning_rate=0.1)

    decision_payload = {
        "action": "run",
        "parent_config": "parent.json",
        "changes": [{"path": "model.params.learning_rate", "value": 0.05, "reason": "x"}],
    }
    text = research_module._render_diff_vs_parent(experiment=experiment, decision_payload=decision_payload)
    assert "## Diff vs parent" in text
    assert "- parent: parent.json" in text
    assert "model.params.learning_rate: 0.1 → 0.05" in text


def test_render_diff_vs_parent_baseline_action_returns_full_copy_note(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")
    decision_payload = {"action": "baseline", "parent_config": "base.json", "changes": []}
    text = research_module._render_diff_vs_parent(experiment=experiment, decision_payload=decision_payload)
    assert "- parent: base.json (full copy; no mutation)" in text


def test_baseline_round_markdown_uses_five_section_template() -> None:
    md = research_module._baseline_round_markdown(
        round_label="r001", parent_name="base.json", config_name="config_001.json"
    )
    assert "## Phase" in md
    assert "## What this decision tests" in md
    assert "## Evidence cited" in md
    assert "## What changed and why" in md
    assert "## Open questions and caveats" in md
    # And the legacy headers are gone
    assert "## Current Best" not in md
    assert "## Tried Inside The Cost Envelope" not in md
    assert "## Next Open Question" not in md


def test_outcome_footer_trigger_cleared_when_metric_above_champion_seed42(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "rounds"
    artifact_dir.mkdir()
    research_module._write_llm_round_markdown(
        artifact_dir=artifact_dir,
        round_label="r042",
        round_markdown="# r042 Research State\n\nBody.",
        round_payload={
            "action": "run",
            "round_type": "discovery",
            "status": "completed",
            "run_id": "abc",
            "config_path": "configs/config_042.json",
            "metric_value": 0.0037,
            "completed_at": "2026-05-19T00:00:00+00:00",
            "wall_time_seconds": 138.0,
            "secondary_metrics": {},
            "confirmation_round": False,
            "champion_seed42_before": 0.0032,
            "promotion": None,
            "phase_snapshot": {
                "phase": "shallow",
                "plateau_counter": 14,
                "plateau_threshold": 25,
                "successful_rounds": 31,
                "min_rounds_in_phase": 30,
            },
        },
    )
    md = (artifact_dir / "r042.md").read_text(encoding="utf-8")
    assert "## Outcome" in md
    assert "- Trigger cleared: yes (metric 0.0037 above champion seed-42 0.0032)" in md
    # The legacy "Confirmation round: no" line has been replaced by the round_type-aware
    # footer; discovery rounds no longer carry that line.
    assert "- Confirmation round:" not in md
    assert "- Promoted: no" in md
    assert "- Phase: shallow plateau 14/25, successful 31/30" in md


def test_outcome_footer_baseline_round_shows_status_line(tmp_path: Path) -> None:
    """Baseline rounds have no champion to compare against and no promotion to
    track; the footer states the baseline-establishment role explicitly."""
    artifact_dir = tmp_path / "rounds"
    artifact_dir.mkdir()
    research_module._write_llm_round_markdown(
        artifact_dir=artifact_dir,
        round_label="r001",
        round_markdown="# r001 Research State\n\nBaseline.",
        round_payload={
            "action": "baseline",
            "round_type": "baseline",
            "status": "completed",
            "run_id": "abc",
            "config_path": "configs/config_001.json",
            "metric_value": 0.0004,
            "completed_at": "2026-05-19T00:00:00+00:00",
            "wall_time_seconds": 111.0,
            "secondary_metrics": {},
            "confirmation_round": False,
            "champion_seed42_before": None,
            "promotion": None,
            "phase_snapshot": {
                "phase": "shallow",
                "plateau_counter": 0,
                "plateau_threshold": 25,
                "successful_rounds": 1,
                "min_rounds_in_phase": 30,
            },
        },
    )
    md = (artifact_dir / "r001.md").read_text(encoding="utf-8")
    assert "- Status: baseline establishment (no LLM mutation)" in md
    # Baseline rounds don't emit Trigger cleared or Promoted lines.
    assert "- Trigger cleared:" not in md
    assert "- Promoted:" not in md


def test_outcome_footer_discovery_with_no_champion_shows_na(tmp_path: Path) -> None:
    """Early discovery rounds (before any champion exists) keep the legacy
    n/a phrasing — there is no champion seed-42 score to compare against."""
    artifact_dir = tmp_path / "rounds"
    artifact_dir.mkdir()
    research_module._write_llm_round_markdown(
        artifact_dir=artifact_dir,
        round_label="r002",
        round_markdown="# r002 Research State\n\nFirst discovery.",
        round_payload={
            "action": "run",
            "round_type": "discovery",
            "status": "completed",
            "run_id": "abc",
            "config_path": "configs/config_002.json",
            "metric_value": 0.0008,
            "completed_at": "2026-05-19T00:00:00+00:00",
            "wall_time_seconds": 111.0,
            "secondary_metrics": {},
            "confirmation_round": False,
            "champion_seed42_before": None,
            "promotion": None,
            "phase_snapshot": {
                "phase": "shallow",
                "plateau_counter": 1,
                "plateau_threshold": 25,
                "successful_rounds": 2,
                "min_rounds_in_phase": 30,
            },
        },
    )
    md = (artifact_dir / "r002.md").read_text(encoding="utf-8")
    assert "- Trigger cleared: n/a (no champion yet)" in md


def test_outcome_footer_confirmation_round_shows_seed_and_trio_progress(tmp_path: Path) -> None:
    """Confirmation rounds drop the "Trigger cleared" line (which was misleadingly
    comparing the confirmation seed against the OLD champion's seed-42) and emit
    a confirmation-specific block with the seed being confirmed, the candidate's
    own seed-42 score, and the trio progress."""
    artifact_dir = tmp_path / "rounds"
    artifact_dir.mkdir()
    research_module._write_llm_round_markdown(
        artifact_dir=artifact_dir,
        round_label="r040",
        round_markdown="# r040 Research State\n\nSeed 99.",
        round_payload={
            "action": "run",
            "round_type": "confirmation",
            "status": "completed",
            "run_id": "abc",
            "config_path": "configs/config_040.json",
            "metric_value": 0.00365,
            "completed_at": "2026-05-19T00:00:00+00:00",
            "wall_time_seconds": 138.0,
            "secondary_metrics": {},
            "confirmation_round": True,
            "champion_seed42_before": 0.00278,
            "confirmation_context": {
                "seed": 99,
                "candidate_seed42_metric": 0.0034,
                "seeds_completed": 3,
                "total_seeds": 3,
            },
            "promotion": {
                "parent_config": "config_038.json",
                "seed_trio_primary_mean": 0.003471,
                "phase": "shallow",
            },
            "phase_snapshot": {
                "phase": "shallow",
                "plateau_counter": 0,
                "plateau_threshold": 25,
                "successful_rounds": 16,
                "min_rounds_in_phase": 30,
            },
        },
    )
    md = (artifact_dir / "r040.md").read_text(encoding="utf-8")
    assert "- Confirmation seed: 99" in md
    assert "- Candidate seed-42 score: 0.0034" in md
    assert "- Trio progress: 3/3 seeds completed" in md
    assert "- Promoted: yes (trio mean 0.003471)" in md
    assert "- Trigger cleared:" not in md


def test_render_diff_vs_parent_renders_missing_old_as_question_mark(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")
    parent_path = experiment.manifest_path.parent / "configs" / "parent.json"
    _write_training_config(parent_path, learning_rate=0.1)

    decision_payload = {
        "action": "run",
        "parent_config": "parent.json",
        "changes": [{"path": "model.params.num_leaves", "value": 8, "reason": "x"}],
    }
    text = research_module._render_diff_vs_parent(experiment=experiment, decision_payload=decision_payload)
    assert "model.params.num_leaves: ? → 8" in text


def test_failed_round_md_includes_llm_proposal_when_decision_was_parsed(tmp_path: Path) -> None:
    """When materialization or training fails after a decision was parsed, the
    failure markdown must record what the LLM proposed."""
    artifact_dir = tmp_path / "rounds"
    artifact_dir.mkdir()
    research_module._write_failure_round_markdown(
        artifact_dir=artifact_dir,
        round_label="r042",
        round_payload={
            "action": "run",
            "status": "failed",
            "run_id": None,
            "config_path": None,
            "metric_value": None,
            "completed_at": "2026-05-19T00:00:00+00:00",
            "error": "agentic_research_candidate_duplicate:abc",
            "error_class": "AgenticResearchValidationError",
            "pending_decision": {
                "action": "run",
                "parent_config": "config_037.json",
                "changes": [{"path": "model.params.num_leaves", "value": 64, "reason": "x"}],
            },
        },
    )
    md = (artifact_dir / "r042.md").read_text(encoding="utf-8")
    assert "## What was attempted" in md
    assert "LLM produced a decision" in md
    assert "AgenticResearchValidationError" in md
    assert "## LLM proposal" in md
    assert "- parent: config_037.json" in md
    assert "- model.params.num_leaves: → 64" in md
    assert "Error: agentic_research_candidate_duplicate:abc" in md


def test_failed_round_md_pre_llm_error_omits_llm_proposal(tmp_path: Path) -> None:
    """Failures before the LLM is consulted (or before parse succeeds) get a
    minimal failure markdown without an LLM proposal section."""
    artifact_dir = tmp_path / "rounds"
    artifact_dir.mkdir()
    research_module._write_failure_round_markdown(
        artifact_dir=artifact_dir,
        round_label="r004",
        round_payload={
            "action": "run",
            "status": "failed",
            "run_id": None,
            "config_path": None,
            "metric_value": None,
            "completed_at": "2026-05-19T00:00:00+00:00",
            "error": "experiment_round_not_found:exp:r004",
            "error_class": "ExperimentValidationError",
        },
    )
    md = (artifact_dir / "r004.md").read_text(encoding="utf-8")
    assert "## What was attempted" in md
    assert "failed before the LLM produced a usable decision" in md
    assert "## LLM proposal" not in md
    assert "ExperimentValidationError" in md


def test_failed_round_md_debug_pointers_only_when_files_exist(tmp_path: Path) -> None:
    """The Debug artifacts section must list only files that actually exist on disk."""
    artifact_dir = tmp_path / "rounds"
    artifact_dir.mkdir()
    (artifact_dir / "r007.debug.error.txt").write_text("boom", encoding="utf-8")
    (artifact_dir / "r007.debug.llm_response.txt").write_text("...", encoding="utf-8")
    research_module._write_failure_round_markdown(
        artifact_dir=artifact_dir,
        round_label="r007",
        round_payload={
            "action": "run",
            "status": "failed",
            "run_id": None,
            "config_path": None,
            "metric_value": None,
            "completed_at": "2026-05-19T00:00:00+00:00",
            "error": "agentic_research_decision_parse_failed",
            "error_class": "AgenticResearchValidationError",
        },
    )
    md = (artifact_dir / "r007.md").read_text(encoding="utf-8")
    assert "## Debug artifacts" in md
    assert "- r007.debug.error.txt" in md
    assert "- r007.debug.llm_response.txt" in md
    assert "- r007.debug.codex_stdout.jsonl" not in md  # never created
    assert "- r007.debug.prompt.md" not in md  # never created


def test_round_md_omits_secondary_metrics_block_when_empty(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "rounds"
    artifact_dir.mkdir()
    research_module._write_llm_round_markdown(
        artifact_dir=artifact_dir,
        round_label="r042",
        round_markdown="# r042 Research State\n\nBody.",
        round_payload={
            "action": "run",
            "status": "completed",
            "run_id": "abc123",
            "config_path": "configs/config_042.json",
            "metric_value": 0.00374,
            "completed_at": "2026-05-19T00:00:00+00:00",
            "wall_time_seconds": 138.34,
            "secondary_metrics": {},
        },
    )
    md = (artifact_dir / "r042.md").read_text(encoding="utf-8")
    assert "## Secondary Metrics" not in md
    assert "- Wall time: 138.3s" in md


def test_reuse_finished_run_on_hash_collision_returns_existing_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When training fails because the run dir already holds a FINISHED run, reuse it
    and link the run into this experiment's manifest."""
    root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=root, experiment_id="2026-05-21_reuse-test", name="Reuse Test")
    run_id = "abcdef012345"
    run_dir = root / "runs" / run_id
    (run_dir / "artifacts" / "predictions").mkdir(parents=True)
    run_dir.joinpath("run.json").write_text(
        json.dumps({"status": "FINISHED", "output": {"predictions_name": "p"}}),
        encoding="utf-8",
    )
    run_dir.joinpath("artifacts/predictions/p.parquet").write_bytes(b"")
    run_dir.joinpath("results.json").write_text("{}", encoding="utf-8")
    exc = TrainingError(f"training_run_dir_not_fresh:{run_id}:preexisting=run.json:reset_required")
    monkeypatch.setattr(research_module, "index_run", lambda **_: None)

    result = research_module._reuse_finished_run_on_hash_collision(root=root, experiment=experiment, exc=exc)

    assert result is not None
    assert result.run_id == run_id
    assert result.experiment_id == "2026-05-21_reuse-test"
    assert result.predictions_path == run_dir / "artifacts" / "predictions" / "p.parquet"
    manifest_after = json.loads(experiment.manifest_path.read_text(encoding="utf-8"))
    assert run_id in manifest_after["runs"]
    assert manifest_after["status"] == "active"


def test_reuse_finished_run_on_hash_collision_skips_non_finished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Do not reuse a run whose status is not FINISHED (e.g. FAILED, RUNNING)."""
    root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=root, experiment_id="2026-05-21_reuse-test", name="Reuse Test")
    run_id = "deadbeef0000"
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True)
    run_dir.joinpath("run.json").write_text(json.dumps({"status": "FAILED"}), encoding="utf-8")
    exc = TrainingError(f"training_run_dir_not_fresh:{run_id}:preexisting=run.json:reset_required")
    monkeypatch.setattr(research_module, "index_run", lambda **_: None)

    result = research_module._reuse_finished_run_on_hash_collision(root=root, experiment=experiment, exc=exc)

    assert result is None
    manifest_after = json.loads(experiment.manifest_path.read_text(encoding="utf-8"))
    assert run_id not in manifest_after.get("runs", [])


def test_reuse_finished_run_on_hash_collision_skips_unrelated_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Errors that aren't the freshness check must not trigger reuse."""
    root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=root, experiment_id="2026-05-21_reuse-test", name="Reuse Test")
    monkeypatch.setattr(research_module, "index_run", lambda **_: None)
    result = research_module._reuse_finished_run_on_hash_collision(
        root=root, experiment=experiment, exc=TrainingError("training_config_invalid:something_else")
    )
    assert result is None


def test_normalize_decision_changes_drops_noop_against_parent() -> None:
    """A change whose new value equals the parent's existing value at the same path is
    dropped. `data.target_horizon` is now stripped earlier as a controller-managed path
    (it lands in `dropped_managed_paths`, not `dropped_no_op_paths`), so this uses an
    ordinary no-op (`data.feature_set: 'small' -> 'small'`) to exercise no-op dropping."""
    parent_payload: dict[str, object] = {
        "data": {"target_col": "target_ender_60", "target_horizon": "60d", "feature_set": "small"},
        "model": {"params": {"learning_rate": 0.1}},
    }
    decision = research_module.ResearchDecision(
        action="run",
        learning="x",
        belief_update="x",
        next_hypothesis="x",
        parent_config="config_017.json",
        changes=(
            research_module.ResearchChange(path="data.target_col", value="target_alpha_60", reason="swap"),
            research_module.ResearchChange(path="data.feature_set", value="small", reason="unchanged no-op"),
        ),
        stop_reason=None,
    )
    normalized, summary = research_module._normalize_decision_changes(decision=decision, parent_payload=parent_payload)
    assert [c.path for c in normalized.changes] == ["data.target_col"]
    assert summary["dropped_no_op_paths"] == ["data.feature_set"]
    assert summary["kept_count"] == 1
    assert summary["raw_count"] == 2


def test_normalize_decision_changes_dedupes_path_last_write_wins() -> None:
    """If the LLM emits the same path twice, the last value wins and the
    duplicate is recorded in the summary."""
    parent_payload: dict[str, object] = {"model": {"params": {"learning_rate": 0.1}}}
    decision = research_module.ResearchDecision(
        action="run",
        learning="x",
        belief_update="x",
        next_hypothesis="x",
        parent_config="config_001.json",
        changes=(
            research_module.ResearchChange(path="model.params.learning_rate", value=0.05, reason="first"),
            research_module.ResearchChange(path="model.params.learning_rate", value=0.03, reason="oops"),
        ),
        stop_reason=None,
    )
    normalized, summary = research_module._normalize_decision_changes(decision=decision, parent_payload=parent_payload)
    assert len(normalized.changes) == 1
    assert normalized.changes[0].value == 0.03
    assert summary["dropped_duplicate_paths"] == ["model.params.learning_rate"]


def test_normalize_decision_changes_keeps_real_mutations() -> None:
    """A decision with no no-ops and no duplicates is returned unchanged."""
    parent_payload: dict[str, object] = {"model": {"params": {"learning_rate": 0.1, "max_depth": 3}}}
    decision = research_module.ResearchDecision(
        action="run",
        learning="x",
        belief_update="x",
        next_hypothesis="x",
        parent_config="config_001.json",
        changes=(
            research_module.ResearchChange(path="model.params.learning_rate", value=0.05, reason="x"),
            research_module.ResearchChange(path="model.params.max_depth", value=4, reason="x"),
        ),
        stop_reason=None,
    )
    normalized, summary = research_module._normalize_decision_changes(decision=decision, parent_payload=parent_payload)
    assert len(normalized.changes) == 2
    assert summary["dropped_no_op_paths"] == []
    assert summary["dropped_duplicate_paths"] == []


def test_normalize_decision_changes_keeps_missing_path_as_real_change() -> None:
    """When the parent payload doesn't have the path (e.g. adding a new key
    like `model.module_path` during a family switch), the change must be kept
    even though the parent value is "missing" rather than equal."""
    parent_payload: dict[str, object] = {"model": {"type": "LGBMRegressor", "params": {}}}
    decision = research_module.ResearchDecision(
        action="run",
        learning="x",
        belief_update="x",
        next_hypothesis="x",
        parent_config="config_001.json",
        changes=(research_module.ResearchChange(path="model.module_path", value="xgboost_model.py", reason="add"),),
        stop_reason=None,
    )
    normalized, summary = research_module._normalize_decision_changes(decision=decision, parent_payload=parent_payload)
    assert [c.path for c in normalized.changes] == ["model.module_path"]
    assert summary["dropped_no_op_paths"] == []


def test_strip_python_owned_sections_removes_llm_authored_diff_block() -> None:
    """An LLM-authored `## Diff vs parent` block must be stripped before the
    composer appends its own canonical version (otherwise the final round.md
    carries two consecutive diff blocks — one with ASCII -> arrows and stale
    `?` placeholders, one with Unicode → arrows from the controller)."""
    text = (
        "# r017 Research State\n\n"
        "## Phase\n\nshallow.\n\n"
        "## Diff vs parent\n\n"
        "- parent: config_014.json\n"
        "- model.type: 'LGBMRegressor' -> 'XGBoostRegressor'\n"
        "- model.module_path: ? -> 'xgboost_model.py'\n\n"
        "## Open questions and caveats\n\nseed variance.\n"
    )
    cleaned, stripped = research_module._strip_python_owned_sections(text)
    assert stripped == ["Diff vs parent"]
    assert "## Diff vs parent" not in cleaned
    assert "## Phase" in cleaned
    assert "## Open questions and caveats" in cleaned


def test_strip_python_owned_sections_no_change_when_llm_does_not_author_them() -> None:
    text = (
        "# r014 Research State\n\n"
        "## Phase\n\nshallow.\n\n"
        "## What this decision tests\n\nA target swap.\n\n"
        "## Open questions and caveats\n\nNone.\n"
    )
    cleaned, stripped = research_module._strip_python_owned_sections(text)
    assert stripped == []
    assert cleaned.rstrip() == text.rstrip()


def test_strip_python_owned_sections_strips_multiple_owned_sections() -> None:
    """An aggressively-over-helpful LLM may try to author Execution Result,
    Secondary Metrics, and Outcome too. All must be dropped."""
    text = (
        "# r042 Research State\n\n"
        "## Phase\n\nshallow.\n\n"
        "## Execution Result\n\n- Action: run\n\n"
        "## Secondary Metrics\n\n- bmc_mean: 0.005\n\n"
        "## Outcome\n\n- Promoted: maybe\n"
    )
    cleaned, stripped = research_module._strip_python_owned_sections(text)
    assert set(stripped) == {"Execution Result", "Secondary Metrics", "Outcome"}
    assert "## Execution Result" not in cleaned
    assert "## Secondary Metrics" not in cleaned
    assert "## Outcome" not in cleaned
    assert "## Phase" in cleaned


def test_apply_state_defaults_backfills_missing_fields() -> None:
    """A state file written by an older controller version is upgraded on load
    to include all canonical fields with zero-values."""
    legacy_state: dict[str, object] = {
        "schema_version": 1,
        "experiment_id": "old-exp",
        "status": "running",
        "next_round_number": 5,
        "total_rounds_completed": 4,
    }
    research_module._apply_state_defaults(legacy_state, has_phase=False)
    assert legacy_state["diversification_dry_run_count"] == 0
    assert legacy_state["failed_rounds_counter"] == 0
    assert legacy_state["confirmations"] == {}
    assert legacy_state["confirmed_champion"] is None
    assert legacy_state["tried_signatures"] == []
    # Existing values must be preserved.
    assert legacy_state["next_round_number"] == 5
    assert legacy_state["total_rounds_completed"] == 4


def test_apply_state_defaults_phase_fields_only_when_has_phase() -> None:
    """Phase defaults are opt-in per experiment; non-phase experiments stay slim."""
    state_without_phase: dict[str, object] = {}
    research_module._apply_state_defaults(state_without_phase, has_phase=False)
    assert "phase_plateau_counter" not in state_without_phase

    state_with_phase: dict[str, object] = {}
    research_module._apply_state_defaults(state_with_phase, has_phase=True)
    assert state_with_phase["phase_plateau_counter"] == 0
    assert state_with_phase["phase_successful_rounds"] == 0
    assert state_with_phase["phase_history"] == []
    assert state_with_phase["phase_best_metric"] is None


# --- Ensemble action -------------------------------------------------------


def _ensemble_form(run_ids: list[str], *, weights: list[float] | None = None) -> dict[str, object]:
    return {
        "action": "ensemble",
        "learning": "Single-model search plateaued.",
        "belief_update": "A blend of the two strongest runs should beat either alone.",
        "next_hypothesis": "Blend the top two complementary runs.",
        "parent_config": None,
        "changes": [],
        "stop_reason": None,
        "ensemble_run_ids": run_ids,
        "ensemble_weights": weights,
    }


def test_parse_ensemble_decision_accepts_valid() -> None:
    decision = research_module._parse_decision_object(_ensemble_form(["run-a", "run-b"]))
    assert decision.action == "ensemble"
    assert decision.ensemble_run_ids == ("run-a", "run-b")
    assert decision.ensemble_weights is None
    assert decision.parent_config is None
    assert decision.changes == ()


def test_parse_ensemble_decision_accepts_weights() -> None:
    decision = research_module._parse_decision_object(_ensemble_form(["run-a", "run-b"], weights=[0.6, 0.4]))
    assert decision.ensemble_weights == (0.6, 0.4)


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda f: f.update(ensemble_run_ids=["only-one"]), "member_count_invalid"),
        (lambda f: f.update(ensemble_run_ids=["a", "a"]), "duplicate_member"),
        (lambda f: f.update(ensemble_run_ids=["ok", "bad id!"]), "run_id_invalid"),
        (lambda f: f.update(ensemble_weights=[0.5]), "weight_count_mismatch"),
        (lambda f: f.update(parent_config="config_001.json"), "parent_config_forbidden"),
        (
            lambda f: f.update(changes=[{"path": "model.params.x", "value": 1, "reason": "no"}]),
            "changes_forbidden",
        ),
    ],
)
def test_parse_ensemble_decision_rejects_malformed(mutate, code: str) -> None:
    form = _ensemble_form(["run-a", "run-b"])
    mutate(form)
    with pytest.raises(AgenticResearchValidationError) as excinfo:
        research_module._parse_decision_object(form)
    assert code in str(excinfo.value)


def test_parse_run_decision_still_requires_parent_and_changes() -> None:
    form: dict[str, object] = {
        "action": "run",
        "learning": "x",
        "belief_update": "y",
        "next_hypothesis": "z",
        "parent_config": None,
        "changes": [],
        "stop_reason": None,
        "ensemble_run_ids": [],
        "ensemble_weights": None,
    }
    with pytest.raises(AgenticResearchValidationError) as excinfo:
        research_module._parse_decision_object(form)
    assert "parent_config_missing" in str(excinfo.value)


def test_llm_response_schema_gates_ensemble_action() -> None:
    without_ensemble = research_module._llm_response_schema(allow_ensemble=False)
    with_ensemble = research_module._llm_response_schema(allow_ensemble=True)

    def _enum(schema: dict[str, object]) -> list[str]:
        decision = cast(dict[str, object], schema["properties"])["decision_form"]
        props = cast(dict[str, object], cast(dict[str, object], decision)["properties"])
        return cast(list[str], cast(dict[str, object], props["action"])["enum"])

    assert _enum(without_ensemble) == ["run"]
    assert _enum(with_ensemble) == ["run", "ensemble"]
    # Ensemble fields are always present and required so the schema stays stable.
    decision = cast(dict[str, object], cast(dict[str, object], without_ensemble["properties"])["decision_form"])
    assert "ensemble_run_ids" in cast(dict[str, object], decision["properties"])
    assert "ensemble_run_ids" in cast(list[str], decision["required"])


def test_eligible_ensemble_rows_filters_and_gates_availability(tmp_path: Path) -> None:
    """The ensemble action is a structural precondition: it is offered iff at least
    ENSEMBLE_MIN_MEMBERS blendable runs exist (FINISHED + scored + predictions on
    disk). No plateau threshold — WHEN to ensemble is the LLM's call."""
    store_root = tmp_path / ".numereng"
    create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")
    _make_member_run(store_root, "run-a", target="target_alpha_60")
    _make_member_run(store_root, "run-b", target="target_alpha_20")
    # run-c is scored in the report but its predictions were rotated off disk.
    report = _report(_row("run-a", 0.004), _row("run-b", 0.0041), _row("run-c", 0.0039))

    rows = research_module._eligible_ensemble_rows(root=store_root, report=report)
    assert [r["run_id"] for r in rows] == ["run-a", "run-b"]  # run-c excluded: no predictions
    assert rows[0]["target"] == "target_alpha_60"

    # Availability is purely a member-count precondition, not a plateau gate.
    assert research_module._ensemble_context(state={}, eligible_rows=rows)["available"] is True
    assert research_module._ensemble_context(state={}, eligible_rows=rows[:1])["available"] is False


def test_update_best_ensemble_only_on_improvement() -> None:
    state: dict[str, object] = {}
    assert research_module._update_best_ensemble(
        state, ensemble_id="e1", member_ids=["a", "b"], weights=None, metric_value=0.003, round_label="r050"
    )
    assert cast(dict[str, object], state["best_ensemble"])["metric_value"] == 0.003
    # Lower score does not displace the incumbent.
    assert not research_module._update_best_ensemble(
        state, ensemble_id="e2", member_ids=["a", "c"], weights=None, metric_value=0.002, round_label="r051"
    )
    assert cast(dict[str, object], state["best_ensemble"])["ensemble_id"] == "e1"
    # Higher score does.
    assert research_module._update_best_ensemble(
        state, ensemble_id="e3", member_ids=["a", "d"], weights=None, metric_value=0.004, round_label="r052"
    )
    assert cast(dict[str, object], state["best_ensemble"])["ensemble_id"] == "e3"


def _make_member_run(store_root: Path, run_id: str, *, target: str) -> None:
    run_dir = store_root / "runs" / run_id
    (run_dir / "artifacts" / "predictions").mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts" / "predictions" / "p.parquet").write_bytes(b"PAR1")
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "status": "FINISHED",
                "data": {
                    "data_version": "v5.2",
                    "feature_set": "small",
                    "dataset_scope": "train_plus_validation",
                    "target_col": target,
                },
            }
        ),
        encoding="utf-8",
    )


def _seed_plateaued_state(experiment_dir: Path, *, plateau: int, next_round: int) -> None:
    state_path = experiment_dir / "agentic_research" / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "running",
                "next_round_number": next_round,
                "total_rounds_completed": next_round - 1,
                "phase": "deep",
                "phase_plateau_counter": plateau,
                "phase_successful_rounds": plateau,
                "confirmed_champion": None,
            }
        ),
        encoding="utf-8",
    )


def test_run_research_builds_and_scores_ensemble_round(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")
    experiment_dir = experiment.manifest_path.parent
    # No plateau-threshold metadata: the ensemble action is available purely because
    # two blendable member runs exist (run-a, run-b made below). The seeded plateau
    # is here only to assert it stays untouched by ensemble rounds.
    _seed_plateaued_state(experiment_dir, plateau=50, next_round=10)
    _make_member_run(store_root, "run-a", target="target_alpha_60")
    _make_member_run(store_root, "run-b", target="target_alpha_20")

    monkeypatch.setattr(
        research_module,
        "_safe_report",
        lambda **_: _report(_row("run-a", 0.0039), _row("run-b", 0.0040)),
    )
    monkeypatch.setattr(
        research_module,
        "_call_research_llm",
        lambda **_: (_llm_response(_ensemble_form(["run-a", "run-b"])), "test"),
    )

    build_calls: list[tuple[str, ...]] = []

    def _fake_build(*, store_root: Path, request: object) -> object:  # noqa: ARG001
        run_ids = cast(tuple[str, ...], getattr(request, "run_ids"))
        build_calls.append(run_ids)
        artifacts = store_root / "ensembles" / "ens-1"
        artifacts.mkdir(parents=True, exist_ok=True)
        (artifacts / "predictions.parquet").write_bytes(b"PAR1")
        return SimpleNamespace(ensemble_id="ens-1", artifacts_path=artifacts)

    monkeypatch.setattr(research_module, "build_ensemble", _fake_build)
    monkeypatch.setattr(research_module, "_score_ensemble_predictions", lambda **_: 0.0042)

    def _fail_score_round(**_: object) -> None:
        raise AssertionError("score_experiment_round must not run for ensemble rounds")

    def _fail_run_plan(**_: object) -> None:
        raise AssertionError("_record_round_config_in_run_plan must not run for ensemble rounds")

    monkeypatch.setattr(research_module, "score_experiment_round", _fail_score_round)
    monkeypatch.setattr(research_module, "_record_round_config_in_run_plan", _fail_run_plan)

    # Two rounds: the second proposes the identical blend and must dedup-skip.
    result = run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=2)

    assert [r.action for r in result.rounds] == ["ensemble", "ensemble"]
    assert build_calls == [("run-a", "run-b")]  # second round reused the cached blend

    state = json.loads((experiment_dir / "agentic_research" / "state.json").read_text(encoding="utf-8"))
    best = state["best_ensemble"]
    assert best["metric_value"] == 0.0042
    assert best["run_ids"] == ["run-a", "run-b"]
    assert state["confirmed_champion"] is None  # single-model track untouched
    assert state["phase_plateau_counter"] == 50  # ensemble rounds neither tick nor reset plateau
    assert state["failed_rounds_counter"] == 0
    # r1 set the first best_ensemble (improved -> 0); r2 dedup-skip did not improve (-> 1).
    assert state["ensemble_rounds_since_improved"] == 1

    decisions = (
        (experiment_dir / "agentic_research" / "rounds" / "decision.json").read_text(encoding="utf-8").splitlines()
    )
    assert json.loads(decisions[-1])["decision"]["action"] == "ensemble"
    assert json.loads(decisions[-1])["round_type"] == "ensemble"

    events = {e["event"] for e in _trace_events(experiment_dir / "agentic_research" / "trace.jsonl")}
    assert "ensemble_round_completed" in events
    assert "ensemble_duplicate_skipped" in events


def test_attach_target_to_blend_merges_target_from_member(tmp_path: Path) -> None:
    """build_ensemble drops the trained target; the scorer needs it inline. The
    controller must re-attach it from a member's prediction file before scoring."""
    import pandas as pd

    blend_path = tmp_path / "predictions.parquet"
    pd.DataFrame({"id": ["i1", "i2", "i3"], "era": ["e1", "e1", "e2"], "prediction": [0.1, 0.2, 0.3]}).to_parquet(
        blend_path, index=False
    )
    member_dir = tmp_path / "runs" / "run-a"
    (member_dir / "artifacts" / "predictions").mkdir(parents=True)
    pd.DataFrame(
        {
            "id": ["i1", "i2", "i3"],
            "era": ["e1", "e1", "e2"],
            "target_ender_20": [0.5, 0.75, 0.25],
            "prediction": [0.9, 0.8, 0.7],
        }
    ).to_parquet(member_dir / "artifacts" / "predictions" / "pred_x.parquet", index=False)

    out = research_module._attach_target_to_blend(
        blend_path=blend_path, target_col="target_ender_20", member_run_dir=member_dir
    )
    assert out.name == "predictions_scored.parquet"
    scored = pd.read_parquet(out)
    assert list(scored.columns) == ["id", "era", "prediction", "target_ender_20"]
    assert scored["target_ender_20"].tolist() == [0.5, 0.75, 0.25]


def test_attach_target_to_blend_passes_through_when_present(tmp_path: Path) -> None:
    import pandas as pd

    blend_path = tmp_path / "predictions.parquet"
    pd.DataFrame({"id": ["i1"], "era": ["e1"], "prediction": [0.1], "target_ender_20": [0.5]}).to_parquet(
        blend_path, index=False
    )
    out = research_module._attach_target_to_blend(
        blend_path=blend_path, target_col="target_ender_20", member_run_dir=tmp_path / "runs" / "missing"
    )
    assert out == blend_path  # already inline → no rewrite, source dir never consulted


def test_attach_target_to_blend_raises_when_source_missing(tmp_path: Path) -> None:
    import pandas as pd

    blend_path = tmp_path / "predictions.parquet"
    pd.DataFrame({"id": ["i1"], "era": ["e1"], "prediction": [0.1]}).to_parquet(blend_path, index=False)
    with pytest.raises(research_module.AgenticResearchValidationError, match="target_source_missing"):
        research_module._attach_target_to_blend(
            blend_path=blend_path, target_col="target_ender_20", member_run_dir=tmp_path / "runs" / "absent"
        )


def test_resolve_member_prediction_file_matches_config_derived_name(tmp_path: Path) -> None:
    """Regression (live r522 bail `ensemble_target_source_missing`): prediction parquets are
    named from the run's config (e.g. `xgb_small_charlie60_champ_s42.parquet`), NOT a fixed
    `pred_*` prefix. The resolver must use the same `*.parquet` glob as the eligibility gate
    (`_member_predictions_exists`) — otherwise a run passes eligibility but fails target-attach."""
    run_dir = tmp_path / "runs" / "run-a"
    pred_dir = run_dir / "artifacts" / "predictions"
    pred_dir.mkdir(parents=True)
    config_named = pred_dir / "xgb_small_charlie60_champ_s42.parquet"  # no `pred_` prefix
    config_named.write_bytes(b"")

    # Both the eligibility gate and the resolver must agree this run is blendable.
    assert research_module._member_predictions_exists(run_dir) is True
    assert research_module._resolve_member_prediction_file(run_dir) == config_named
