"""Tests for the single round path: decide -> execute -> finalize, plus the one in-round retry.

Every round shape (baseline, single seed, seed trio, rejection, duplicate, terminal failure) now
flows through one execution path and one journal-line builder. These tests pin the journal field
set and the `## Machine Result` block those rounds produce, pin the retry semantics (rebuild the
context with the rejection token, ask once more, record the second failure exactly as one failure),
and prove that a journal line written by the OLD multi-path code still aggregates and still builds
closeout evidence alongside lines from the new builder.

USAGE:
    uv run pytest tests/unit/numereng/test_agentic_research_round_path.py -q
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from numereng.agentic_research.engine import aggregate
from numereng.agentic_research.engine import loop as research_module
from numereng.agentic_research.engine.closeout import evidence as closeout_evidence
from numereng.features.experiments import (
    ExperimentReport,
    ExperimentReportRow,
    ExperimentScoreRoundResult,
    ExperimentTrainResult,
    create_experiment,
    get_experiment,
)

EXPERIMENT_ID = "2026-09-01_round-path-exp"

# One journal line emitted by the PRE-refactor code, copied verbatim out of
# .numereng/experiments/2026-08-31_nn-e60-agentic-hillclimb/agentic_research/journal.jsonl.
# Downstream readers must keep consuming it unchanged.
OLD_CODE_JOURNAL_LINE = (
    '{"action": "baseline", "benchmark_corr": 0.3038210831628037, "changes": [], "completed_at": '
    '"2026-09-01T16:56:43.917270+00:00", "config": "config_001.json", "created_at": '
    '"2026-09-01T16:56:43.917270+00:00", "fnc": 0.020484462266418844, "is_champion": true, '
    '"learning": "Baseline round (copy of seed `config_001.json`) before asking the LLM for '
    'mutations.", "llm": null, "metric": 0.0061576761850895565, "next_hypothesis": null, '
    '"parent_config": "config_001.json", "round": 1, "round_label": "r001", "run_id": '
    '"999384e1a461", "seed": 42, "status": "completed", "wall_seconds": 971.6}'
)

# The field set every journal line carries. `error` is the one conditional field.
JOURNAL_FIELDS = {
    "round",
    "round_label",
    "action",
    "status",
    "config",
    "parent_config",
    "run_id",
    "seed",
    "metric",
    "fnc",
    "benchmark_corr",
    "is_champion",
    "learning",
    "next_hypothesis",
    "changes",
    "wall_seconds",
    "llm",
    "created_at",
    "completed_at",
}


# --------------------------------------------------------------------------- #
# Filesystem helpers
# --------------------------------------------------------------------------- #
def _write_training_config(path: Path, *, learning_rate: float = 0.01, random_state: int | None = None) -> None:
    params: dict[str, object] = {"learning_rate": learning_rate}
    if random_state is not None:
        params["random_state"] = random_state
    payload: dict[str, object] = {
        "data": {"data_version": "v5.2", "dataset_variant": "non_downsampled", "target_col": "target"},
        "model": {"type": "LGBMRegressor", "params": params},
        "training": {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _setup_experiment(tmp_path: Path, *, random_state: int | None = None) -> tuple[Path, Path]:
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Round path")
    experiment_dir = experiment.manifest_path.parent
    _write_training_config(experiment_dir / "configs" / "seed.json", random_state=random_state)
    return store_root, experiment_dir


def _agentic_dir(experiment_dir: Path) -> Path:
    return experiment_dir / "agentic_research"


def _entries(experiment_dir: Path) -> list[dict[str, object]]:
    text = (_agentic_dir(experiment_dir) / "journal.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _state(experiment_dir: Path) -> dict[str, object]:
    return json.loads((_agentic_dir(experiment_dir) / "state.json").read_text(encoding="utf-8"))


def _memo(experiment_dir: Path, round_label: str) -> str:
    return (_agentic_dir(experiment_dir) / "rounds" / f"{round_label}.md").read_text(encoding="utf-8")


def _config_files(experiment_dir: Path) -> set[str]:
    return {path.name for path in (experiment_dir / "configs").glob("*.json")}


# --------------------------------------------------------------------------- #
# LLM response builders
# --------------------------------------------------------------------------- #
def _response(
    *,
    path: str = "model.params.learning_rate",
    value: object = 0.02,
    seeds: list[int] | None = None,
    memo: str = "# rNNN Research State\n\nMemo.",
) -> str:
    decision_form: dict[str, object] = {
        "action": "run",
        "learning": "probe",
        "belief_update": "belief",
        "next_hypothesis": "hypothesis",
        "parent_config": "seed.json",
        "changes": [{"path": path, "value": value, "reason": "probe"}],
        "stop_reason": None,
    }
    if seeds is not None:
        decision_form["seeds"] = seeds
    payload = {"decision_form": decision_form, "round_markdown": memo, "experiment_markdown": None}
    return json.dumps(payload)


def _rejected_response(*, value: object = "downsampled") -> str:
    """A proposal the boundary rejects: `data.dataset_variant` is deliberately off the allowlist."""
    return _response(path="data.dataset_variant", value=value)


# --------------------------------------------------------------------------- #
# Seams (the five approved monkeypatch points on loop)
# --------------------------------------------------------------------------- #
@dataclass
class _Seams:
    store_root: Path
    experiment_dir: Path
    rows: list[ExperimentReportRow] = field(default_factory=list)
    train_queue: list[object] = field(default_factory=list)
    llm_queue: list[object] = field(default_factory=list)
    llm_prompts: list[str] = field(default_factory=list)

    def add_row(self, run_id: str, metric: float) -> None:
        self.rows.append(
            ExperimentReportRow(
                run_id=run_id,
                status="FINISHED",
                created_at="2026-09-01T00:00:00+00:00",
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
        seams.add_row(str(run_id), float(metric))
        return ExperimentTrainResult(
            experiment_id=EXPERIMENT_ID,
            run_id=str(run_id),
            predictions_path=store_root / "runs" / str(run_id) / "predictions.parquet",
            results_path=store_root / "runs" / str(run_id) / "results.json",
        )

    def fake_score(**kwargs: object) -> ExperimentScoreRoundResult:
        return ExperimentScoreRoundResult(
            experiment_id=EXPERIMENT_ID,
            round=str(kwargs.get("round")),
            stage="post_training_full",
            run_ids=("scored-run",),
        )

    def fake_llm(**kwargs: object) -> tuple[str, str]:
        seams.llm_prompts.append(str(kwargs.get("prompt") or ""))
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


def _run(store_root: Path, *, max_rounds: int) -> object:
    return research_module.run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=max_rounds)


# --------------------------------------------------------------------------- #
# One path per round shape: baseline, single seed, seed trio
# --------------------------------------------------------------------------- #
def test_baseline_round_writes_one_journal_line_through_the_shared_builder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_root, experiment_dir = _setup_experiment(tmp_path, random_state=42)
    seams = _install_seams(monkeypatch, store_root, experiment_dir)
    seams.train_queue = [("run-1", 0.10)]

    result = _run(store_root, max_rounds=1)

    # The baseline is a synthetic decision down the same path: no LLM call, one journal line.
    assert seams.llm_queue == [] and seams.llm_prompts == []
    entries = _entries(experiment_dir)
    assert len(entries) == 1
    entry = entries[0]
    assert set(entry) == JOURNAL_FIELDS
    assert entry["round"] == 1
    assert entry["round_label"] == "r001"
    assert entry["action"] == "baseline"
    assert entry["status"] == "completed"
    assert entry["config"] == "config_001.json"
    assert entry["parent_config"] == "config_001.json"
    assert entry["run_id"] == "run-1"
    assert entry["seed"] == 42
    assert entry["metric"] == 0.10
    assert entry["is_champion"] is True
    assert entry["changes"] == []
    assert entry["llm"] is None
    assert entry["created_at"] == entry["completed_at"]
    assert [r.status for r in result.rounds] == ["completed"]  # type: ignore[attr-defined]

    memo = _memo(experiment_dir, "r001")
    assert "## Machine Result" in memo
    assert "- action: baseline" in memo
    assert "- status: completed" in memo
    assert "- run_id: run-1" in memo
    assert "- retry:" not in memo
    assert "- per-seed results:" not in memo


def test_single_seed_round_writes_one_journal_line_with_the_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_root, experiment_dir = _setup_experiment(tmp_path)
    seams = _install_seams(monkeypatch, store_root, experiment_dir)
    seams.train_queue = [("run-1", 0.10), ("run-2", 0.15)]
    seams.llm_queue = [_response(value=0.02)]

    _run(store_root, max_rounds=2)

    entries = _entries(experiment_dir)
    assert [entry["round"] for entry in entries] == [1, 2]
    entry = entries[1]
    assert set(entry) == JOURNAL_FIELDS
    assert entry["action"] == "run"
    assert entry["status"] == "completed"
    assert entry["config"] == "config_002.json"
    assert entry["run_id"] == "run-2"
    assert entry["changes"] == [{"path": "model.params.learning_rate", "value": 0.02}]
    assert entry["next_hypothesis"] == "hypothesis"
    assert entry["learning"] == "probe"
    assert entry["llm"] == "test"
    # Un-suffixed config name, and no per-seed block for a round the model gave no seeds for.
    assert not any(name.startswith("config_002_s") for name in _config_files(experiment_dir))
    assert "- per-seed results:" not in _memo(experiment_dir, "r002")


def test_three_seed_round_writes_three_journal_lines_for_one_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_root, experiment_dir = _setup_experiment(tmp_path, random_state=42)
    seams = _install_seams(monkeypatch, store_root, experiment_dir)
    seams.train_queue = [("run-1", 0.10), ("run-42", 0.12), ("run-17", 0.20), ("run-99", 0.15)]
    seams.llm_queue = [_response(value=0.02, seeds=[42, 17, 99])]

    result = _run(store_root, max_rounds=2)

    # One decision is one round even though it produced three runs.
    assert [r.round_number for r in result.rounds] == [1, 2]  # type: ignore[attr-defined]
    trio = [entry for entry in _entries(experiment_dir) if entry["round"] == 2]
    assert len(trio) == 3
    assert all(set(entry) == JOURNAL_FIELDS for entry in trio)
    assert all(entry["round_label"] == "r002" for entry in trio)
    assert all(entry["action"] == "run" for entry in trio)
    assert all(entry["status"] == "completed" for entry in trio)
    assert [entry["seed"] for entry in trio] == [42, 17, 99]
    assert [entry["config"] for entry in trio] == ["config_002_s42.json", "config_002_s17.json", "config_002_s99.json"]
    assert [entry["run_id"] for entry in trio] == ["run-42", "run-17", "run-99"]
    # Champion advances per run, in seed order: run-42 beats the baseline, run-17 then beats it.
    assert [entry["is_champion"] for entry in trio] == [True, True, False]

    memo = _memo(experiment_dir, "r002")
    assert "- run_id: run-17" in memo  # the primary outcome speaks for the round
    assert "- per-seed results:" in memo
    for seed, run_id in (("42", "run-42"), ("17", "run-17"), ("99", "run-99")):
        assert f"  - seed {seed}: status=completed run_id={run_id} " in memo


# --------------------------------------------------------------------------- #
# The one in-round retry
# --------------------------------------------------------------------------- #
def test_rejected_proposal_then_valid_one_records_a_single_completed_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_root, experiment_dir = _setup_experiment(tmp_path)
    seams = _install_seams(monkeypatch, store_root, experiment_dir)
    seams.train_queue = [("run-1", 0.10), ("run-2", 0.15)]
    seams.llm_queue = [_rejected_response(), _response(value=0.02)]

    result = _run(store_root, max_rounds=2)

    # Two LLM calls, one round: the rejection never reaches the journal.
    assert seams.llm_queue == []
    assert len(seams.llm_prompts) == 2
    assert "agentic_research_change_path_not_allowed:data.dataset_variant" in seams.llm_prompts[1]
    assert [r.round_number for r in result.rounds] == [1, 2]  # type: ignore[attr-defined]
    assert [r.status for r in result.rounds] == ["completed", "completed"]  # type: ignore[attr-defined]
    entries = _entries(experiment_dir)
    assert [entry["round"] for entry in entries] == [1, 2]
    assert entries[1]["status"] == "completed"
    assert entries[1]["run_id"] == "run-2"
    assert "error" not in entries[1]

    # The first token is the memo's record that a retry happened.
    memo = _memo(experiment_dir, "r002")
    assert "- retry: agentic_research_change_path_not_allowed:data.dataset_variant" in memo
    assert "- status: completed" in memo

    state = _state(experiment_dir)
    assert state["failed_rounds_counter"] == 0
    assert state["total_rounds_completed"] == 2
    assert state["next_round_number"] == 3


def test_two_rejections_record_one_failed_round_and_increment_the_counter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_root, experiment_dir = _setup_experiment(tmp_path)
    seams = _install_seams(monkeypatch, store_root, experiment_dir)
    seams.train_queue = [("run-1", 0.10)]
    seams.llm_queue = [_rejected_response(), _rejected_response(value="non_downsampled")]

    result = _run(store_root, max_rounds=2)

    assert seams.llm_queue == []
    assert [r.status for r in result.rounds] == ["completed", "failed"]  # type: ignore[attr-defined]
    entries = _entries(experiment_dir)
    # One failed line for the round, not one per attempt.
    assert [entry["round"] for entry in entries] == [1, 2]
    failed = entries[1]
    assert set(failed) == JOURNAL_FIELDS | {"error"}
    assert failed["status"] == "failed"
    assert failed["config"] is None
    assert failed["run_id"] is None
    assert "agentic_research_change_path_not_allowed:data.dataset_variant" in str(failed["error"])
    assert str(failed["learning"]).startswith("Round skipped: ")

    # A second failure counts exactly as one failure did before the retry existed.
    state = _state(experiment_dir)
    assert state["failed_rounds_counter"] == 1
    assert state["total_rounds_completed"] == 1
    assert state["next_round_number"] == 3
    assert "agentic_research_change_path_not_allowed" in str(state["last_error"])

    memo = _memo(experiment_dir, "r002")
    assert "- status: failed" in memo
    assert "- retry: agentic_research_change_path_not_allowed:data.dataset_variant" in memo


def test_duplicate_on_both_attempts_soft_skips_without_incrementing_the_counter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_root, experiment_dir = _setup_experiment(tmp_path)
    seams = _install_seams(monkeypatch, store_root, experiment_dir)
    seams.train_queue = [("run-1", 0.10), ("run-2", 0.15)]
    # r002 materializes lr=0.02; r003 proposes it twice more, so both attempts are duplicates.
    seams.llm_queue = [_response(value=0.02), _response(value=0.02), _response(value=0.02)]

    result = _run(store_root, max_rounds=3)

    assert seams.llm_queue == []
    assert [r.status for r in result.rounds] == ["completed", "completed", "skipped"]  # type: ignore[attr-defined]
    entries = _entries(experiment_dir)
    assert [entry["round"] for entry in entries] == [1, 2, 3]
    skipped = entries[2]
    assert skipped["status"] == "skipped"
    assert skipped["config"] is None
    assert "agentic_research_candidate_duplicate" in str(skipped["error"])

    # A duplicate is not a harness failure: the bail counter stays put.
    state = _state(experiment_dir)
    assert state["failed_rounds_counter"] == 0
    assert state["total_rounds_completed"] == 3
    assert state["believed_best"] is not None
    assert "- retry: agentic_research_candidate_duplicate" in _memo(experiment_dir, "r003")


# --------------------------------------------------------------------------- #
# Backward compatibility with journals written by the old code
# --------------------------------------------------------------------------- #
def test_old_and_new_journal_lines_aggregate_and_build_evidence_together(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_root, experiment_dir = _setup_experiment(tmp_path, random_state=42)
    seams = _install_seams(monkeypatch, store_root, experiment_dir)
    seams.train_queue = [("run-1", 0.10), ("run-2", 0.15)]
    seams.llm_queue = [_response(value=0.02)]
    _run(store_root, max_rounds=2)

    # Append one line written by the PRE-refactor code. It names `config_001.json` and seed 42 —
    # the slot the new baseline line also holds — so the aggregator's "latest entry for a
    # (recipe, seed) wins" rule must hand that slot to the old line.
    journal_path = _agentic_dir(experiment_dir) / "journal.jsonl"
    text = journal_path.read_text(encoding="utf-8")
    journal_path.write_text(text + OLD_CODE_JOURNAL_LINE + "\n", encoding="utf-8")

    entries = _entries(experiment_dir)
    new_entry, old_entry = entries[0], entries[-1]
    # Same field set from both eras of the code: the old line needs no migration.
    assert set(old_entry) == JOURNAL_FIELDS
    assert set(new_entry) == JOURNAL_FIELDS

    configs = {
        path.name: json.loads(path.read_text(encoding="utf-8")) for path in (experiment_dir / "configs").glob("*.json")
    }
    groups = aggregate.aggregate_recipes(entries, configs=configs)
    by_config = {name: aggregate.group_for_config(groups, name, configs) for name in configs}
    # The old line is a first-class entry: it takes the baseline recipe's seed-42 slot outright.
    baseline_group = by_config["config_001.json"]
    assert baseline_group is not None
    assert baseline_group.trio_mean == pytest.approx(0.0061576761850895565)
    assert baseline_group.run_ids == ("999384e1a461",)
    assert baseline_group.seeds == (42,)
    # ...and the new builder's lines aggregate beside it, unchanged.
    mutation_group = by_config["config_002.json"]
    assert mutation_group is not None
    assert mutation_group.trio_mean == pytest.approx(0.15)
    assert mutation_group.run_ids == ("run-2",)

    experiment = get_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID)
    bundle = closeout_evidence.build_evidence(
        experiment=experiment, state=_state(experiment_dir), store_root=store_root
    )
    # Evidence counts every completed line, old and new, and still resolves the believed best.
    assert bundle["totals"]["journal_entries"] == len(entries)
    assert bundle["totals"]["completed"] == len(entries)
    assert bundle["believed_best"]["config"] == "config_002.json"
    leaderboard = {row["representative_config"]: row["run_ids"] for row in bundle["leaderboard"]}
    assert leaderboard == {"config_001.json": ["999384e1a461"], "config_002.json": ["run-2"]}
