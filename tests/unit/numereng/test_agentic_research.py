"""Tests for the minimal agentic config-research loop."""

from __future__ import annotations

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


def test_status_synthesizes_blank_state(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")

    status = get_research_status(store_root=store_root, experiment_id=EXPERIMENT_ID)

    assert status.status == "initialized"
    assert status.next_round_number == 1
    assert status.program_path.name == "PROGRAM.md"


def test_status_uses_experiment_metadata_program(tmp_path: Path) -> None:
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Research")
    manifest = json.loads(experiment.manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"] = {"agentic_research_program": "TEST-PROGRAM.md"}
    experiment.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    status = get_research_status(store_root=store_root, experiment_id=EXPERIMENT_ID)

    assert status.program_path.name == "TEST-PROGRAM.md"


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
    artifact_dir = experiment.manifest_path.parent / "agentic_research" / "rounds"
    assert (experiment.manifest_path.parent / "agentic_research" / "ledger.jsonl").is_file()
    assert result.rounds[0].artifact_dir == artifact_dir
    assert (artifact_dir / "decision.json").is_file()
    assert (artifact_dir / "r001.md").is_file()
    assert not (artifact_dir / "r001").exists()
    assert not (artifact_dir / "context.json").exists()
    assert not (artifact_dir / "round.json").exists()
    assert not (artifact_dir / "learning.md").exists()


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
    artifact_dir = experiment.manifest_path.parent / "agentic_research" / "rounds"
    assert (artifact_dir / "decision.json").is_file()
    assert (artifact_dir / "r001.md").is_file()
    assert "A slightly larger learning rate is worth testing." in (artifact_dir / "r001.md").read_text(encoding="utf-8")
    assert not (artifact_dir / "r001").exists()
    assert not (artifact_dir / "context.json").exists()
    assert not (artifact_dir / "prompt.md").exists()
    assert not (artifact_dir / "llm_response.txt").exists()
    assert not (artifact_dir / "codex_stdout.jsonl").exists()
    assert not (artifact_dir / "codex_stderr.txt").exists()
    assert not (artifact_dir / "round.json").exists()
    assert not (artifact_dir / "learning.md").exists()


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
    ) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        captured["input"] = input
        _ = (text, capture_output, check)
        output_path = Path(cmd[cmd.index("-o") + 1])
        output_path.write_text('{"action": "stop"}', encoding="utf-8")
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
    assert response == '{"action": "stop"}'
    assert captured["input"] == "choose next run"
    assert not (tmp_path / "codex_stdout.jsonl").exists()
    assert not (tmp_path / "codex_stderr.txt").exists()


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
    ) -> subprocess.CompletedProcess[str]:
        _ = (cmd, input, text, capture_output, check)
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
