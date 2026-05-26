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
    assert "Confirmation round:" in round_notes
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
    )

    assert context["latest_round_markdown"] == "latest memo"
    assert "old memo" not in json.dumps(context)
    assert "debug prompt" not in json.dumps(context)


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
                    "action": "stop",
                    "learning": "done",
                    "belief_update": "done",
                    "next_hypothesis": None,
                    "parent_config": None,
                    "changes": [],
                    "stop_reason": "done",
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
    assert captured["timeout"] == research_module.CODEX_TIMEOUT_SECONDS
    assert captured["encoding"] == "utf-8"
    assert captured["errors"] == "replace"
    schema = cast(dict[str, object], captured["schema"])
    assert "decision_form" in cast(dict[str, object], schema["properties"])
    assert json.loads(response)["decision_form"]["action"] == "stop"
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


def test_materialize_rejects_value_outside_program_cap(tmp_path: Path) -> None:
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
    with pytest.raises(AgenticResearchValidationError, match="agentic_research_change_value_out_of_range"):
        research_module._materialize_decision_config(experiment=experiment, round_label="r002", decision=decision)


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
                        "action": "stop",
                        "learning": "x",
                        "belief_update": "x",
                        "next_hypothesis": None,
                        "parent_config": None,
                        "changes": [],
                        "stop_reason": "x",
                    }
                }
            )
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


def test_terminal_phase_stop_emits_all_phases_done() -> None:
    """Predicate satisfied in a phase marked is_terminal → state stops with all_phases_done reason."""
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
    assert payload == {"transition": "terminal", "from": "medium", "to": None}
    assert state["status"] == "stopped"
    assert state["stop_reason"] == "all_phases_done:medium_plateau"
    history = state["phase_history"]
    assert isinstance(history, list) and len(history) == 1
    assert history[0]["exit_reason"] == "all_phases_done"


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
    """Complete seeds 42/17/99 with mean above champion+3e-4 → confirmed_champion updated."""
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
    """New trio mean only beats champion by less than 3e-4 → no promotion."""
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
        "target": "target_alpha_60",
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
    """phase_successful_rounds increments only on a real metric; plateau ticks on no improvement."""
    state: dict[str, object] = {"phase": "shallow"}
    research_module._update_phase_progress(state=state, metric_value=0.5)
    research_module._update_phase_progress(state=state, metric_value=0.5)
    assert state["phase_successful_rounds"] == 2
    assert state["phase_plateau_counter"] == 1
    assert state["phase_best_metric"] == 0.5


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

    result = run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=2)
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
    )
    assert context["canonical_seed_trio"] == list(research_module.CANONICAL_SEED_TRIO)


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
    assert "- Confirmation round: no" in md
    assert "- Promoted: no" in md
    assert "- Phase: shallow plateau 14/25, successful 31/30" in md


def test_outcome_footer_no_champion_yet_shows_na(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "rounds"
    artifact_dir.mkdir()
    research_module._write_llm_round_markdown(
        artifact_dir=artifact_dir,
        round_label="r001",
        round_markdown="# r001 Research State\n\nBaseline.",
        round_payload={
            "action": "baseline",
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
    assert "- Trigger cleared: n/a (no champion yet)" in md


def test_outcome_footer_promoted_yes_with_trio_mean(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "rounds"
    artifact_dir.mkdir()
    research_module._write_llm_round_markdown(
        artifact_dir=artifact_dir,
        round_label="r040",
        round_markdown="# r040 Research State\n\nSeed 99.",
        round_payload={
            "action": "run",
            "status": "completed",
            "run_id": "abc",
            "config_path": "configs/config_040.json",
            "metric_value": 0.00365,
            "completed_at": "2026-05-19T00:00:00+00:00",
            "wall_time_seconds": 138.0,
            "secondary_metrics": {},
            "confirmation_round": True,
            "champion_seed42_before": 0.00278,
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
    assert "- Confirmation round: yes" in md
    assert "- Promoted: yes (trio mean 0.003471)" in md


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
