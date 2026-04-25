"""Tests for the minimal agentic config-research loop."""

from __future__ import annotations

import json
from pathlib import Path

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


def test_status_synthesizes_blank_state(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")

    status = get_research_status(store_root=store_root, experiment_id=EXPERIMENT_ID)

    assert status.status == "initialized"
    assert status.next_round_number == 1
    assert status.program_path.name == "PROGRAM.md"


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
            stage="post_training_full",
            run_ids=("run-1",),
        ),
    )

    result = run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=1)

    assert result.rounds[0].action == "baseline"
    assert result.rounds[0].run_id == "run-1"
    assert (experiment.manifest_path.parent / "configs" / "r001_baseline_seed.json").is_file()
    assert (experiment.manifest_path.parent / "agentic_research" / "ledger.jsonl").is_file()


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
            json.dumps(
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
                }
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
            stage="post_training_full",
            run_ids=("run-1",),
        ),
    )

    result = run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=1)

    config_path = result.rounds[0].config_path
    assert config_path is not None
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert payload["model"]["params"]["learning_rate"] == 0.02
    assert result.rounds[0].metric_value == 0.13
    assert (experiment.manifest_path.parent / "agentic_research" / "rounds" / "r001" / "prompt.md").is_file()


def test_parse_decision_rejects_disallowed_change_path() -> None:
    with pytest.raises(AgenticResearchValidationError, match="agentic_research_change_path_not_allowed"):
        research_module._parse_decision(
            json.dumps(
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
