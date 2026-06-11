"""Unit tests for the rebuilt agentic-research modules (boundary/memory/context/loop).

These import the NEW modules directly (not run.py) and cover the rebuild's risk list:
boundary rejection tokens + reject-never-edit + dedup-vs-orphan, memory append-only +
corrupt-state + round-md, context boundedness (long session), and a happy-path loop round
with the five seams patched ON loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from numereng.features.agentic_research import boundary, context, memory
from numereng.features.agentic_research import loop as research_module
from numereng.features.agentic_research.types import (
    AgenticResearchDuplicateCandidate,
    AgenticResearchValidationError,
    ResearchChange,
    ResearchDecision,
)
from numereng.features.experiments import (
    ExperimentReport,
    ExperimentReportRow,
    ExperimentScoreRoundResult,
    ExperimentTrainResult,
    create_experiment,
    get_experiment,
)

EXPERIMENT_ID = "2026-06-10_rebuild-exp"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _write_training_config(path: Path, *, learning_rate: float = 0.01, target_col: str = "target") -> None:
    payload: dict[str, object] = {
        "data": {"data_version": "v5.2", "dataset_variant": "non_downsampled", "target_col": target_col},
        "model": {"type": "LGBMRegressor", "params": {"learning_rate": learning_rate}},
        "training": {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _setup_experiment(tmp_path: Path) -> tuple[Path, Path]:
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Rebuild")
    experiment_dir = experiment.manifest_path.parent
    _write_training_config(experiment_dir / "configs" / "seed.json")
    return store_root, experiment_dir


def _experiment(store_root: Path):
    return get_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID)


def _decision(*, path: str = "model.params.learning_rate", value: object = 0.02, parent: str = "seed.json"):
    return ResearchDecision(
        action="run",
        learning="probe",
        belief_update="belief",
        next_hypothesis="hypothesis",
        parent_config=parent,
        changes=(ResearchChange(path=path, value=value, reason="why"),),
        stop_reason=None,
    )


def _rounds_dir(experiment_dir: Path) -> Path:
    return experiment_dir / "agentic_research" / "rounds"


def _state_path(experiment_dir: Path) -> Path:
    return experiment_dir / "agentic_research" / "state.json"


def _config_files(experiment_dir: Path) -> set[str]:
    return {path.name for path in (experiment_dir / "configs").glob("*.json")}


# ---------------------------------------------------------------------------
# boundary
# ---------------------------------------------------------------------------


def test_boundary_rejects_path_not_in_narrowed_allowlist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store_root, _ = _setup_experiment(tmp_path)
    experiment = _experiment(store_root)
    monkeypatch.setattr(boundary, "program_allowed_paths", lambda _exp: ("model.params.*",))
    with pytest.raises(AgenticResearchValidationError) as exc:
        boundary.materialize_config(
            experiment=experiment, round_label="r001", decision=_decision(path="data.feature_set", value="medium")
        )
    assert "agentic_research_change_path_not_allowed:data.feature_set" in str(exc.value)


def test_boundary_rejects_out_of_cap_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store_root, _ = _setup_experiment(tmp_path)
    experiment = _experiment(store_root)
    monkeypatch.setattr(boundary, "program_value_caps", lambda _exp: {"model.params.learning_rate": (0.0, 0.05)})
    with pytest.raises(AgenticResearchValidationError) as exc:
        boundary.materialize_config(experiment=experiment, round_label="r001", decision=_decision(value=0.9))
    assert "agentic_research_change_value_out_of_cap:model.params.learning_rate" in str(exc.value)


def test_boundary_rejects_horizon_target_mismatch(tmp_path: Path) -> None:
    store_root, experiment_dir = _setup_experiment(tmp_path)
    # Parent pins target_horizon=60d; the model switches target_col to an _20 target (an
    # allowed change), so the materialized config carries a contradicting horizon -> reject.
    seed_payload = {
        "data": {
            "data_version": "v5.2",
            "dataset_variant": "non_downsampled",
            "target_col": "target_alpha_60",
            "target_horizon": "60d",
        },
        "model": {"type": "LGBMRegressor", "params": {"learning_rate": 0.01}},
        "training": {},
    }
    (experiment_dir / "configs" / "seed.json").write_text(json.dumps(seed_payload), encoding="utf-8")
    experiment = _experiment(store_root)
    decision = ResearchDecision(
        action="run",
        learning="probe",
        belief_update="belief",
        next_hypothesis="hyp",
        parent_config="seed.json",
        changes=(ResearchChange(path="data.target_col", value="target_ender_20", reason="why"),),
        stop_reason=None,
    )
    with pytest.raises(AgenticResearchValidationError) as exc:
        boundary.materialize_config(experiment=experiment, round_label="r001", decision=decision)
    assert "agentic_research_horizon_target_mismatch:target_ender_20:60d" in str(exc.value)


def test_boundary_reject_never_edits_the_decision_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store_root, _ = _setup_experiment(tmp_path)
    experiment = _experiment(store_root)
    monkeypatch.setattr(boundary, "program_value_caps", lambda _exp: {"model.params.learning_rate": (0.0, 0.05)})
    decision = _decision(value=0.9)
    before = [{"path": c.path, "value": c.value, "reason": c.reason} for c in decision.changes]
    with pytest.raises(AgenticResearchValidationError):
        boundary.materialize_config(experiment=experiment, round_label="r001", decision=decision)
    after = [{"path": c.path, "value": c.value, "reason": c.reason} for c in decision.changes]
    assert after == before  # the harness never mutates the proposal


def test_boundary_duplicate_with_recorded_run_raises(tmp_path: Path) -> None:
    store_root, experiment_dir = _setup_experiment(tmp_path)
    experiment = _experiment(store_root)
    config_path = boundary.materialize_config(experiment=experiment, round_label="r001", decision=_decision(value=0.02))
    # Record a run for that config in the journal, then re-propose the identical mutation.
    memory.append_journal(experiment, {"config": config_path.name, "run_id": "run-1"})
    with pytest.raises(AgenticResearchDuplicateCandidate) as exc:
        boundary.materialize_config(experiment=experiment, round_label="r002", decision=_decision(value=0.02))
    assert "agentic_research_candidate_duplicate" in str(exc.value)


def test_boundary_orphan_config_is_adopted_not_skipped(tmp_path: Path) -> None:
    store_root, experiment_dir = _setup_experiment(tmp_path)
    experiment = _experiment(store_root)
    # First write produces config_001 but NO journal run is recorded (crash orphan).
    boundary.materialize_config(experiment=experiment, round_label="r001", decision=_decision(value=0.02))
    # Re-proposing the same mutation must adopt/overwrite and run, not soft-skip.
    path = boundary.materialize_config(experiment=experiment, round_label="r002", decision=_decision(value=0.02))
    assert path.name == "config_002.json"
    assert path.is_file()


# ---------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------


def test_journal_is_append_only_across_rounds(tmp_path: Path) -> None:
    store_root, experiment_dir = _setup_experiment(tmp_path)
    experiment = _experiment(store_root)
    journal = experiment_dir / "agentic_research" / "journal.jsonl"
    memory.append_journal(experiment, {"round": 1, "status": "completed"})
    first_line = journal.read_bytes().splitlines()[0]
    memory.append_journal(experiment, {"round": 2, "status": "failed"})
    memory.append_journal(experiment, {"round": 3, "status": "skipped"})
    lines = journal.read_bytes().splitlines()
    assert len(lines) == 3
    assert lines[0] == first_line  # the first line is byte-identical after later appends


def test_corrupt_state_raises_stable_token(tmp_path: Path) -> None:
    store_root, experiment_dir = _setup_experiment(tmp_path)
    state_path = _state_path(experiment_dir)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text('{"status": "run', encoding="utf-8")
    with pytest.raises(AgenticResearchValidationError) as exc:
        memory.load_state(state_path)
    assert "agentic_research_state_invalid" in str(exc.value)


def test_v1_shaped_state_loads_via_defaults(tmp_path: Path) -> None:
    store_root, experiment_dir = _setup_experiment(tmp_path)
    state_path = _state_path(experiment_dir)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    # A v1-era state dict with stale keys must load without crashing and gain v2 shape.
    state_path.write_text(
        json.dumps({"schema_version": 1, "status": "running", "confirmations": {}, "phase": "deep"}),
        encoding="utf-8",
    )
    loaded = memory.load_state(state_path)
    assert loaded is not None
    assert loaded["schema_version"] == 2
    assert loaded["champion"] is None  # backfilled
    assert loaded["failed_rounds_counter"] == 0


def test_round_markdown_has_memo_verbatim_and_machine_block(tmp_path: Path) -> None:
    store_root, experiment_dir = _setup_experiment(tmp_path)
    experiment = _experiment(store_root)
    entry = {
        "round": 1,
        "round_label": "r001",
        "action": "run",
        "status": "completed",
        "parent_config": "seed.json",
        "config": "config_001.json",
        "run_id": "run-1",
        "metric": 0.0123,
        "is_champion": True,
        "wall_seconds": 12.3,
    }
    memory.write_round_markdown(experiment, entry, memo="# r001 Research State\n\nLift the learning rate.")
    text = (_rounds_dir(experiment_dir) / "r001.md").read_text(encoding="utf-8")
    assert "Lift the learning rate." in text
    assert "## Machine Result" in text
    assert "- run_id: run-1" in text
    assert "- champion: yes" in text


# ---------------------------------------------------------------------------
# context boundedness (the deferred new-contract test)
# ---------------------------------------------------------------------------


def _seed_long_session(experiment_dir: Path, *, rounds: int) -> None:
    journal = experiment_dir / "agentic_research" / "journal.jsonl"
    journal.parent.mkdir(parents=True, exist_ok=True)
    long_memo = "x" * 50_000  # would blow the cap if surfaced verbatim
    with journal.open("w", encoding="utf-8") as handle:
        for n in range(1, rounds + 1):
            handle.write(
                json.dumps(
                    {
                        "round": n,
                        "round_label": f"r{n:03d}",
                        "status": "completed",
                        "config": f"config_{n:03d}.json",
                        "run_id": f"run-{n}",
                        "metric": 0.001 * n,
                        "learning": long_memo,
                    }
                )
                + "\n"
            )
    config_dir = experiment_dir / "configs"
    for n in range(1, rounds + 1):
        _write_training_config(config_dir / f"config_{n:03d}.json", learning_rate=0.001 * n)
    rounds_dir = _rounds_dir(experiment_dir)
    rounds_dir.mkdir(parents=True, exist_ok=True)
    (rounds_dir / f"r{rounds:03d}.md").write_text("# memo\n" + long_memo, encoding="utf-8")
    (experiment_dir / "EXPERIMENT.md").write_text(long_memo, encoding="utf-8")


def _build_context_at(store_root: Path, experiment_dir: Path, *, rounds: int) -> dict[str, object]:
    _seed_long_session(experiment_dir, rounds=rounds)
    experiment = _experiment(store_root)
    # _safe_report caps at REPORT_LIMIT (25); mirror that here since we hand-build the report.
    report = ExperimentReport(
        experiment_id=EXPERIMENT_ID,
        metric="bmc_last_200_eras.mean",
        total_runs=rounds,
        champion_run_id=None,
        rows=tuple(
            ExperimentReportRow(
                run_id=f"run-{n}",
                status="FINISHED",
                created_at="2026-06-10T00:00:00+00:00",
                metric_value=0.001 * n,
                corr_mean=0.01,
                mmc_mean=0.02,
                cwmm_mean=0.03,
                bmc_mean=0.04,
                bmc_last_200_eras_mean=0.001 * n,
                is_champion=False,
            )
            for n in range(rounds, max(0, rounds - 25), -1)
        ),
    )
    state = memory.initial_state(experiment)
    return context.build_context(root=store_root, experiment=experiment, report=report, state=state)


def test_context_is_bounded_and_does_not_grow_with_round_count(tmp_path: Path) -> None:
    store_root, experiment_dir = _setup_experiment(tmp_path)
    ctx_100 = _build_context_at(store_root, experiment_dir, rounds=100)
    size_100 = len(json.dumps(ctx_100, default=str))
    ctx_200 = _build_context_at(store_root, experiment_dir, rounds=200)
    size_200 = len(json.dumps(ctx_200, default=str))

    # No term grows with round count: the two sizes match within a small tolerance.
    assert abs(size_200 - size_100) <= max(1024, size_100 // 50)
    assert len(ctx_200["configs"]) <= 41  # champion + last 40
    assert len(ctx_200["recent_journal"]) <= 12
    assert len(ctx_200["report"]["rows"]) <= 25
    for field_name in ("last_round_memo", "experiment_notes", "research_memory"):
        value = ctx_200.get(field_name)
        if value is not None:
            assert len(value) <= 12_000 + 64  # cap + truncation marker


# ---------------------------------------------------------------------------
# loop happy-path (five seams patched ON loop)
# ---------------------------------------------------------------------------


@dataclass
class _Seams:
    store_root: Path
    experiment_dir: Path
    rows: list[ExperimentReportRow] = field(default_factory=list)
    train_queue: list[object] = field(default_factory=list)
    llm_queue: list[object] = field(default_factory=list)

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


def _mutation_response(*, value: object = 0.02) -> str:
    return json.dumps(
        {
            "decision_form": {
                "action": "run",
                "learning": "probe",
                "belief_update": "belief",
                "next_hypothesis": "hyp",
                "parent_config": "seed.json",
                "changes": [{"path": "model.params.learning_rate", "value": value, "reason": "probe"}],
                "stop_reason": None,
            },
            "round_markdown": "# rNNN Research State\n\nMemo.",
            "experiment_markdown": None,
        }
    )


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
        run_id, metric = seams.train_queue.pop(0)  # type: ignore[misc]
        seams.add_row(str(run_id), float(metric))
        return ExperimentTrainResult(
            experiment_id=EXPERIMENT_ID,
            run_id=str(run_id),
            predictions_path=store_root / "runs" / str(run_id) / "predictions.parquet",
            results_path=store_root / "runs" / str(run_id) / "results.json",
        )

    def fake_score(**kwargs: object) -> ExperimentScoreRoundResult:
        return ExperimentScoreRoundResult(
            experiment_id=EXPERIMENT_ID, round=str(kwargs.get("round")), stage="post_training_core", run_ids=("x",)
        )

    def fake_llm(**_: object) -> tuple[str, str]:
        return str(seams.llm_queue.pop(0)), "test"

    monkeypatch.setattr(research_module, "_safe_report", fake_report)
    monkeypatch.setattr(research_module, "train_experiment", fake_train)
    monkeypatch.setattr(research_module, "score_experiment_round", fake_score)
    monkeypatch.setattr(research_module, "_call_research_llm", fake_llm)
    monkeypatch.setattr(research_module, "index_run", lambda **_: None)
    return seams


def test_loop_happy_path_baseline_then_mutation_round(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store_root, experiment_dir = _setup_experiment(tmp_path)
    seams = _install_seams(monkeypatch, store_root, experiment_dir)
    # Round 1: no scored rows -> baseline (no LLM). Round 2: LLM mutation improves the metric.
    seams.train_queue = [("run-1", 0.10), ("run-2", 0.15)]
    seams.llm_queue = [_mutation_response(value=0.02)]

    result = research_module.run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=2)

    assert [r.action for r in result.rounds] == ["baseline", "run"]
    assert [r.status for r in result.rounds] == ["completed", "completed"]
    assert result.best_overall.run_id == "run-2"
    state = json.loads(_state_path(experiment_dir).read_text(encoding="utf-8"))
    assert state["champion"]["run_id"] == "run-2"
    assert state["champion"]["metric"] == pytest.approx(0.15)


def test_loop_champion_advances_only_on_strict_improvement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store_root, experiment_dir = _setup_experiment(tmp_path)
    seams = _install_seams(monkeypatch, store_root, experiment_dir)
    # Baseline 0.15, then a mutation that ties exactly at 0.15 must NOT advance the champion.
    seams.train_queue = [("run-1", 0.15), ("run-2", 0.15)]
    seams.llm_queue = [_mutation_response(value=0.02)]

    research_module.run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=2)

    state = json.loads(_state_path(experiment_dir).read_text(encoding="utf-8"))
    # Equal metric does not advance: the champion remains the first (baseline) run.
    assert state["champion"]["run_id"] == "run-1"
