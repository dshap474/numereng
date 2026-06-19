"""Contract tests for the agentic-research harness keep-list.

These tests characterize the behaviors that must survive the 7-invariants rebuild
(.project/plans/2026-06-10-agentic-research-7-invariants-rebuild.md). They are written
ONLY against the public API (`run_research`, `get_research_status`), the durable
artifacts (state.json, rounds/rNNN.md, configs/, run_plan.csv, EXPERIMENT.md), and the
approved mock seams. They must pass today AND after run.py is replaced.

All monkeypatching is centralized in `_install_seams` / `_Seams` below — the rebuild
only needs to update that helper layer, never the tests themselves.
"""

from __future__ import annotations

import csv
import inspect
import json
from dataclasses import dataclass, field, fields
from pathlib import Path

import pytest

from numereng.features.agentic_research import (
    AgenticResearchError,
    ResearchBestRun,
    ResearchRoundResult,
    ResearchRunResult,
    ResearchStatusResult,
    get_research_status,
    run_research,
)
from numereng.features.agentic_research import loop as research_module
from numereng.features.experiments import (
    ExperimentReport,
    ExperimentReportRow,
    ExperimentScoreRoundResult,
    ExperimentTrainResult,
    create_experiment,
)
from numereng.features.training.errors import TrainingError

EXPERIMENT_ID = "2026-06-10_contract-exp"


# ---------------------------------------------------------------------------
# Durable-artifact helpers (filesystem only — no internals)
# ---------------------------------------------------------------------------


def _write_training_config(path: Path, *, learning_rate: float = 0.01) -> None:
    payload: dict[str, object] = {
        "data": {"data_version": "v5.2", "dataset_variant": "non_downsampled"},
        "model": {"type": "LGBMRegressor", "params": {"learning_rate": learning_rate}},
        "training": {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _setup_experiment(tmp_path: Path) -> tuple[Path, Path]:
    """Create an experiment with a seed config; return (store_root, experiment_dir)."""
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Contract")
    experiment_dir = experiment.manifest_path.parent
    _write_training_config(experiment_dir / "configs" / "seed.json")
    return store_root, experiment_dir


def _read_state(experiment_dir: Path) -> dict[str, object]:
    return json.loads((experiment_dir / "agentic_research" / "state.json").read_text(encoding="utf-8"))


def _rounds_dir(experiment_dir: Path) -> Path:
    return experiment_dir / "agentic_research" / "rounds"


def _config_files(experiment_dir: Path) -> set[str]:
    return {path.name for path in (experiment_dir / "configs").glob("*.json")}


def _run_plan_rows(experiment_dir: Path) -> list[dict[str, str]]:
    path = experiment_dir / "run_plan.csv"
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _set_experiment_metadata(experiment_dir: Path, metadata: dict[str, object]) -> None:
    manifest_path = experiment_dir / "experiment.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["metadata"] = {**payload.get("metadata", {}), **metadata}
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_run_manifest(store_root: Path, run_id: str, *, experiment_id: str) -> None:
    run_dir = store_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "FINISHED",
        "experiment_id": experiment_id,
        "output": {"predictions_name": "predictions"},
    }
    (run_dir / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
    # Reusable runs are *already scored*: persist the primary metric so the
    # reuse path sees a scored run (an unscored reused run is rescored instead).
    metrics = {"bmc_last_200_eras": {"mean": 0.12}}
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


# ---------------------------------------------------------------------------
# LLM response builders
# ---------------------------------------------------------------------------


def _llm_response(
    decision_form: dict[str, object],
    *,
    round_markdown: str = "# rNNN Research State\n\nMemo.",
    experiment_markdown: str | None = None,
) -> str:
    return json.dumps(
        {
            "decision_form": decision_form,
            "round_markdown": round_markdown,
            "experiment_markdown": experiment_markdown,
        }
    )


def _mutation_response(
    *,
    parent: str = "seed.json",
    path: str = "model.params.learning_rate",
    value: object = 0.02,
    round_markdown: str = "# rNNN Research State\n\nMemo.",
    experiment_markdown: str | None = None,
) -> str:
    return _llm_response(
        {
            "action": "run",
            "learning": "Prior round informed this probe.",
            "belief_update": "The probed axis matters.",
            "next_hypothesis": "This mutation improves the primary metric.",
            "parent_config": parent,
            "changes": [{"path": path, "value": value, "reason": "probe"}],
            "stop_reason": None,
        },
        round_markdown=round_markdown,
        experiment_markdown=experiment_markdown,
    )


# ---------------------------------------------------------------------------
# Mock-seam layer. The ONLY patched attributes are the approved seams:
# train_experiment, score_experiment_round, _safe_report, _call_research_llm,
# index_run. The rebuild updates this class, not the tests.
# ---------------------------------------------------------------------------


@dataclass
class _Seams:
    store_root: Path
    experiment_dir: Path
    # Leaderboard rows surfaced by the report seam (kept sorted best-first, like
    # the real report). Seed a row here to skip the baseline round.
    rows: list[ExperimentReportRow] = field(default_factory=list)
    # One entry per expected train call: (run_id, metric) tuples, or an Exception
    # to raise from train_experiment.
    train_queue: list[object] = field(default_factory=list)
    # One entry per expected LLM call: a raw-response string, or an exception
    # (BaseException allowed: KeyboardInterrupt) to raise.
    llm_queue: list[object] = field(default_factory=list)
    # (round_label, requested_stage, run_plan_had_row_for_round) per
    # score_experiment_round call.
    score_calls: list[tuple[str, str, bool]] = field(default_factory=list)

    def add_row(self, run_id: str, metric: float) -> None:
        self.rows.append(
            ExperimentReportRow(
                run_id=run_id,
                status="FINISHED",
                created_at="2026-06-10T00:00:00+00:00",
                metric_value=metric,
                corr_mean=0.01,
                mmc_mean=0.02,
                cwmm_mean=0.03,
                bmc_mean=0.04,
                bmc_last_200_eras_mean=metric,
                is_champion=False,
            )
        )
        self.rows.sort(key=lambda row: row.bmc_last_200_eras_mean or 0.0, reverse=True)


def _install_seams(monkeypatch: pytest.MonkeyPatch, store_root: Path, experiment_dir: Path) -> _Seams:
    seams = _Seams(store_root=store_root, experiment_dir=experiment_dir)

    def fake_report(**_: object) -> ExperimentReport | None:
        if not seams.rows:
            return None
        return ExperimentReport(
            experiment_id=EXPERIMENT_ID,
            metric="bmc_last_200_eras.mean",
            total_runs=len(seams.rows),
            champion_run_id=None,
            rows=tuple(seams.rows),
        )

    def fake_train(**_: object) -> ExperimentTrainResult:
        item = seams.train_queue.pop(0)
        if isinstance(item, Exception):
            raise item
        run_id, metric = item  # type: ignore[misc]
        seams.add_row(str(run_id), float(metric))  # type: ignore[arg-type]
        return ExperimentTrainResult(
            experiment_id=EXPERIMENT_ID,
            run_id=str(run_id),
            predictions_path=store_root / "runs" / str(run_id) / "predictions.parquet",
            results_path=store_root / "runs" / str(run_id) / "results.json",
        )

    def fake_score(**kwargs: object) -> ExperimentScoreRoundResult:
        round_label = str(kwargs.get("round"))
        stage = str(kwargs.get("stage"))
        plan_rows = _run_plan_rows(experiment_dir)
        seams.score_calls.append((round_label, stage, any(row.get("round") == round_label for row in plan_rows)))
        return ExperimentScoreRoundResult(
            experiment_id=EXPERIMENT_ID,
            round=round_label,
            stage="post_training_full",
            run_ids=("scored-run",),
        )

    def fake_llm(**_: object) -> tuple[str, str]:
        item = seams.llm_queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return str(item), "test"

    monkeypatch.setattr(research_module, "_safe_report", fake_report)
    monkeypatch.setattr(research_module, "train_experiment", fake_train)
    monkeypatch.setattr(research_module, "score_experiment_round", fake_score)
    monkeypatch.setattr(research_module, "_call_research_llm", fake_llm)
    monkeypatch.setattr(research_module, "index_run", lambda **_: None)
    return seams


# ---------------------------------------------------------------------------
# Public surface: signatures, result dataclasses, blank status
# ---------------------------------------------------------------------------


def test_public_api_signatures_and_result_fields() -> None:
    run_params = inspect.signature(run_research).parameters
    assert list(run_params) == ["store_root", "experiment_id", "max_rounds"]
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in run_params.values())
    assert run_params["max_rounds"].default == 1

    status_params = inspect.signature(get_research_status).parameters
    assert list(status_params) == ["store_root", "experiment_id"]
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in status_params.values())

    assert {f.name for f in fields(ResearchRunResult)} == {
        "experiment_id",
        "status",
        "next_round_number",
        "total_rounds_completed",
        "last_checkpoint",
        "stop_reason",
        "best_overall",
        "rounds",
        "interrupted",
    }
    assert {f.name for f in fields(ResearchRoundResult)} == {
        "round_number",
        "round_label",
        "action",
        "status",
        "config_path",
        "run_id",
        "metric_value",
        "learning",
        "artifact_dir",
    }
    assert {f.name for f in fields(ResearchBestRun)} == {
        "experiment_id",
        "run_id",
        "bmc_last_200_eras_mean",
        "bmc_mean",
        "corr_mean",
        "mmc_mean",
        "cwmm_mean",
        "updated_at",
    }
    # Subset only: log-file path fields (trace/decision) are expected to change
    # when the memory artifacts consolidate to journal.jsonl.
    status_fields = {f.name for f in fields(ResearchStatusResult)}
    assert {
        "experiment_id",
        "status",
        "next_round_number",
        "total_rounds_completed",
        "last_round_label",
        "last_run_id",
        "stop_reason",
        "best_overall",
        "agentic_research_dir",
        "state_path",
        "program_path",
    } <= status_fields


def test_status_synthesizes_blank_state(tmp_path: Path) -> None:
    store_root, experiment_dir = _setup_experiment(tmp_path)

    status = get_research_status(store_root=store_root, experiment_id=EXPERIMENT_ID)

    assert status.experiment_id == EXPERIMENT_ID
    assert status.status == "initialized"
    assert status.next_round_number == 1
    assert status.total_rounds_completed == 0
    assert status.stop_reason is None
    assert status.best_overall == ResearchBestRun()
    assert status.state_path == experiment_dir / "agentic_research" / "state.json"
    # Synthesized status must not create state on disk.
    assert not status.state_path.exists()


# ---------------------------------------------------------------------------
# Baseline round
# ---------------------------------------------------------------------------


def test_first_round_is_baseline_copy_of_seed_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store_root, experiment_dir = _setup_experiment(tmp_path)
    seams = _install_seams(monkeypatch, store_root, experiment_dir)
    seams.train_queue = [("run-1", 0.12)]  # no scored rows yet -> baseline, no LLM call

    result = run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=1)

    assert result.rounds[0].action == "baseline"
    assert result.rounds[0].status == "completed"
    assert result.rounds[0].run_id == "run-1"
    assert seams.llm_queue == []  # never consulted (queue untouched because empty)
    baseline_config = experiment_dir / "configs" / "config_001.json"
    assert baseline_config.is_file()
    assert json.loads(baseline_config.read_text(encoding="utf-8"))["model"]["params"]["learning_rate"] == 0.01
    assert (_rounds_dir(experiment_dir) / "r001.md").is_file()
    # Frozen evaluator: the loop requested the post_training_full stage, after
    # the run_plan row for the round existed.
    assert seams.score_calls == [("r001", "post_training_full", True)]
    state = _read_state(experiment_dir)
    assert state["total_rounds_completed"] == 1
    assert state["next_round_number"] == 2
    assert state["failed_rounds_counter"] == 0


# ---------------------------------------------------------------------------
# LLM mutation round
# ---------------------------------------------------------------------------


def test_llm_mutation_round_materializes_config_and_records_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_root, experiment_dir = _setup_experiment(tmp_path)
    seams = _install_seams(monkeypatch, store_root, experiment_dir)
    seams.add_row("run-0", 0.10)  # scored row -> LLM round
    memo = "# r001 Research State\n\nThe baseline underfit; lift the learning rate."
    seams.llm_queue = [_mutation_response(value=0.02, round_markdown=memo)]
    seams.train_queue = [("run-1", 0.13)]

    result = run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=1)

    round_result = result.rounds[0]
    assert round_result.status == "completed"
    assert round_result.run_id == "run-1"
    assert round_result.metric_value == pytest.approx(0.13)
    config_path = round_result.config_path
    assert config_path is not None and config_path.name == "config_001.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert payload["model"]["params"]["learning_rate"] == 0.02
    round_md = (_rounds_dir(experiment_dir) / "r001.md").read_text(encoding="utf-8")
    assert "lift the learning rate" in round_md
    plan_rows = _run_plan_rows(experiment_dir)
    assert [(row["round"], row["config_path"]) for row in plan_rows] == [("r001", "configs/config_001.json")]
    # Frozen evaluator, durable artifact: the run_plan row pins the default stage.
    assert plan_rows[0]["score_stage_default"] == "post_training_full"
    # score_experiment_round ran AFTER the run_plan row existed, at the frozen stage.
    assert seams.score_calls == [("r001", "post_training_full", True)]
    state = _read_state(experiment_dir)
    assert state["status"] == "stopped"
    assert state["total_rounds_completed"] == 1


# ---------------------------------------------------------------------------
# Champion / best tracking
# ---------------------------------------------------------------------------


def test_best_overall_tracks_highest_metric_across_rounds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store_root, experiment_dir = _setup_experiment(tmp_path)
    seams = _install_seams(monkeypatch, store_root, experiment_dir)
    seams.add_row("run-0", 0.10)
    seams.llm_queue = [_mutation_response(value=0.02), _mutation_response(value=0.03)]
    seams.train_queue = [("run-1", 0.15), ("run-2", 0.05)]  # second round regresses

    result = run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=2)

    assert [round_.run_id for round_ in result.rounds] == ["run-1", "run-2"]
    assert result.best_overall.run_id == "run-1"
    assert result.best_overall.bmc_last_200_eras_mean == pytest.approx(0.15)
    status = get_research_status(store_root=store_root, experiment_id=EXPERIMENT_ID)
    assert status.best_overall.run_id == "run-1"


# ---------------------------------------------------------------------------
# Allowed-path rejection
# ---------------------------------------------------------------------------


def test_disallowed_change_path_fails_round_and_loop_continues(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store_root, experiment_dir = _setup_experiment(tmp_path)
    _set_experiment_metadata(experiment_dir, {"agentic_research_allowed_change_paths": ["model.params.*"]})
    seams = _install_seams(monkeypatch, store_root, experiment_dir)
    seams.add_row("run-0", 0.10)
    seams.llm_queue = [
        _mutation_response(path="data.feature_set", value="medium"),  # outside the allowlist
        _mutation_response(path="model.params.learning_rate", value=0.02),
    ]
    seams.train_queue = [("run-1", 0.11)]  # only round 2 trains

    result = run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=2)

    assert [round_.status for round_ in result.rounds] == ["failed", "completed"]
    assert "agentic_research_change_path_not_allowed:data.feature_set" in result.rounds[0].learning
    assert result.rounds[0].config_path is None
    assert "config_001.json" not in _config_files(experiment_dir)  # rejected round wrote nothing
    assert "config_002.json" in _config_files(experiment_dir)
    state = _read_state(experiment_dir)
    assert state["failed_rounds_counter"] == 0  # round-2 success reset the counter
    assert state["total_rounds_completed"] == 1  # failed rounds do not count


# ---------------------------------------------------------------------------
# Dedup soft-skip
# ---------------------------------------------------------------------------


def test_duplicate_config_soft_skips_and_resets_failure_counter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_root, experiment_dir = _setup_experiment(tmp_path)
    seams = _install_seams(monkeypatch, store_root, experiment_dir)
    seams.add_row("run-0", 0.10)
    same_mutation = _mutation_response(value=0.02)
    seams.llm_queue = [
        same_mutation,  # r001: trains config_001
        AgenticResearchError("agentic_research_llm_down"),  # r002: failure, counter -> 1
        same_mutation,  # r003: identical hash -> duplicate soft skip, counter -> 0
    ]
    seams.train_queue = [("run-1", 0.11)]

    result = run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=3)

    assert [round_.status for round_ in result.rounds] == ["completed", "failed", "skipped"]
    assert result.rounds[2].run_id is None
    assert result.rounds[2].config_path is None
    assert _config_files(experiment_dir) == {"seed.json", "config_001.json"}  # no dup file
    state = _read_state(experiment_dir)
    assert state["failed_rounds_counter"] == 0  # dup skip reset the counter from 1
    # A duplicate skip counts as a completed (no-progress) round.
    assert state["total_rounds_completed"] == 2
    assert state["next_round_number"] == 4
    assert (_rounds_dir(experiment_dir) / "r003.md").is_file()


# ---------------------------------------------------------------------------
# Consecutive-failure bail + resumability
# ---------------------------------------------------------------------------


def test_five_consecutive_failures_bail_stops_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store_root, experiment_dir = _setup_experiment(tmp_path)
    seams = _install_seams(monkeypatch, store_root, experiment_dir)
    seams.add_row("run-0", 0.10)
    seams.llm_queue = [AgenticResearchError(f"boom_{i}") for i in range(10)]

    result = run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=10)

    assert len(result.rounds) == 5  # bailed before exhausting max_rounds
    assert all(round_.status == "failed" for round_ in result.rounds)
    assert result.status == "stopped"
    assert result.stop_reason == "consecutive_failures:5"
    state = _read_state(experiment_dir)
    assert state["status"] == "stopped"
    assert state["stop_reason"] == "consecutive_failures:5"
    assert state["failed_rounds_counter"] == 5
    assert state["total_rounds_completed"] == 0


def test_session_resumes_after_failure_bail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store_root, experiment_dir = _setup_experiment(tmp_path)
    seams = _install_seams(monkeypatch, store_root, experiment_dir)
    seams.add_row("run-0", 0.10)
    seams.llm_queue = [AgenticResearchError("boom") for _ in range(5)]
    bailed = run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=10)
    assert bailed.stop_reason == "consecutive_failures:5"

    seams.llm_queue = [_mutation_response(value=0.02)]
    seams.train_queue = [("run-1", 0.11)]
    resumed = run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=1)

    assert resumed.rounds[0].status == "completed"
    assert resumed.rounds[0].round_number == 6  # numbering continued past the bail
    assert resumed.status == "stopped"
    assert resumed.stop_reason == "max_rounds_reached"
    state = _read_state(experiment_dir)
    assert state["failed_rounds_counter"] == 0  # success reset the counter


# ---------------------------------------------------------------------------
# Stale-run-reuse guard
# ---------------------------------------------------------------------------


def test_stale_run_reuse_same_experiment_succeeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store_root, experiment_dir = _setup_experiment(tmp_path)
    seams = _install_seams(monkeypatch, store_root, experiment_dir)
    _write_run_manifest(store_root, "run-reuse", experiment_id=EXPERIMENT_ID)
    seams.train_queue = [TrainingError("training_run_dir_not_fresh:run-reuse:cfghash")]

    result = run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=1)

    assert result.rounds[0].status == "completed"
    assert result.rounds[0].run_id == "run-reuse"
    # Reused runs are already scored: the scoring seam must not be invoked again.
    assert seams.score_calls == []
    manifest = json.loads((experiment_dir / "experiment.json").read_text(encoding="utf-8"))
    assert "run-reuse" in manifest["runs"]


def test_stale_run_reuse_cross_experiment_blocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store_root, experiment_dir = _setup_experiment(tmp_path)
    seams = _install_seams(monkeypatch, store_root, experiment_dir)
    _write_run_manifest(store_root, "run-reuse", experiment_id="some-other-experiment")
    seams.train_queue = [TrainingError("training_run_dir_not_fresh:run-reuse:cfghash")]

    result = run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=1)

    assert result.rounds[0].status == "failed"
    assert "agentic_research_stale_run_reuse_blocked:run-reuse" in result.rounds[0].learning
    assert "some-other-experiment" in result.rounds[0].learning
    assert result.rounds[0].run_id is None
    state = _read_state(experiment_dir)
    assert state["failed_rounds_counter"] == 1
    manifest = json.loads((experiment_dir / "experiment.json").read_text(encoding="utf-8"))
    assert "run-reuse" not in (manifest.get("runs") or [])


# ---------------------------------------------------------------------------
# State persistence / resumability across invocations
# ---------------------------------------------------------------------------


def test_state_persists_across_invocations_with_continuing_round_numbers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_root, experiment_dir = _setup_experiment(tmp_path)
    seams = _install_seams(monkeypatch, store_root, experiment_dir)
    seams.add_row("run-0", 0.10)

    seams.llm_queue = [_mutation_response(value=0.02)]
    seams.train_queue = [("run-1", 0.11)]
    first = run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=1)
    assert first.status == "stopped"
    assert first.stop_reason == "max_rounds_reached"
    assert first.rounds[0].round_label == "r001"

    seams.llm_queue = [_mutation_response(value=0.03)]
    seams.train_queue = [("run-2", 0.12)]
    second = run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=1)

    assert second.rounds[0].round_number == 2
    assert second.rounds[0].round_label == "r002"
    assert second.total_rounds_completed == 2
    state = _read_state(experiment_dir)
    assert state["next_round_number"] == 3
    assert state["total_rounds_completed"] == 2
    assert state["status"] == "stopped"
    assert state["stop_reason"] == "max_rounds_reached"
    assert (_rounds_dir(experiment_dir) / "r001.md").is_file()
    assert (_rounds_dir(experiment_dir) / "r002.md").is_file()


# ---------------------------------------------------------------------------
# EXPERIMENT.md passthrough
# ---------------------------------------------------------------------------


def test_experiment_markdown_overwritten_when_llm_returns_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store_root, experiment_dir = _setup_experiment(tmp_path)
    experiment_md = experiment_dir / "EXPERIMENT.md"
    experiment_md.write_text("# Stale\nPrior content.\n", encoding="utf-8")
    seams = _install_seams(monkeypatch, store_root, experiment_dir)
    seams.add_row("run-0", 0.10)
    curated = "# Active Beliefs\n- LR 0.02 beats 0.01.\n"
    seams.llm_queue = [_mutation_response(value=0.02, experiment_markdown=curated)]
    seams.train_queue = [("run-1", 0.11)]

    run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=1)

    assert experiment_md.read_text(encoding="utf-8") == curated


def test_experiment_markdown_preserved_when_llm_returns_null(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store_root, experiment_dir = _setup_experiment(tmp_path)
    experiment_md = experiment_dir / "EXPERIMENT.md"
    prior = "# Prior\nKept across rounds.\n"
    experiment_md.write_text(prior, encoding="utf-8")
    seams = _install_seams(monkeypatch, store_root, experiment_dir)
    seams.add_row("run-0", 0.10)
    seams.llm_queue = [_mutation_response(value=0.02, experiment_markdown=None)]
    seams.train_queue = [("run-1", 0.11)]

    run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=1)

    assert experiment_md.read_text(encoding="utf-8") == prior


# ---------------------------------------------------------------------------
# LLM failure handling + clean interrupt
# ---------------------------------------------------------------------------


def test_llm_failure_records_failed_round_and_returns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store_root, experiment_dir = _setup_experiment(tmp_path)
    seams = _install_seams(monkeypatch, store_root, experiment_dir)
    seams.add_row("run-0", 0.10)
    seams.llm_queue = [AgenticResearchError("agentic_research_codex_failed:1:stream disconnected")]

    result = run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=1)

    assert result.rounds[0].status == "failed"
    assert "agentic_research_codex_failed" in result.rounds[0].learning
    assert result.status == "stopped"  # session ended at max_rounds, not crashed
    assert _config_files(experiment_dir) == {"seed.json"}  # no config written
    assert seams.score_calls == []
    state = _read_state(experiment_dir)
    assert state["failed_rounds_counter"] == 1
    assert state["total_rounds_completed"] == 0
    assert (_rounds_dir(experiment_dir) / "r001.md").is_file()  # failure still memorialized


def test_keyboard_interrupt_marks_state_interrupted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store_root, experiment_dir = _setup_experiment(tmp_path)
    seams = _install_seams(monkeypatch, store_root, experiment_dir)
    seams.add_row("run-0", 0.10)
    seams.llm_queue = [KeyboardInterrupt()]

    result = run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=3)

    assert result.interrupted is True
    assert result.status == "interrupted"
    state = _read_state(experiment_dir)
    assert state["status"] == "interrupted"
    assert state["stop_reason"] == "keyboard_interrupt"
