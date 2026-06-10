"""Reliability characterization for the agentic-research harness (rebuild step 2 probe).

Each test pins the ACTUAL behavior of one reliability path from the 7-invariants
rebuild spec (crash mid-round, partial-state recovery, transport failures). Tests
asserting rebuild-surviving contracts are unmarked; tests that characterize a HOLE
deferred to the rebuild carry a `# HOLE:` comment and pass against current behavior.

Fixed (no longer a hole): reuse-without-scoring — a same-experiment reused run with
no primary metric on disk is now rescored (or the round fails honestly) instead of
silently completing unscored; see the two baseline-score-failure tests below.

Same seam-layer pattern as test_agentic_research_contract.py: all monkeypatching is
centralized in `_install_seams`, touching only the approved seams
(train_experiment, score_experiment_round, _safe_report, _call_research_llm,
index_run). This file's seams additionally support score-failure injection and
unscored (metric=None) report rows, which the reliability probes require.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from numereng.features.agentic_research import (
    AgenticResearchError,
    get_research_status,
    run_research,
)
from numereng.features.agentic_research import run as research_module
from numereng.features.experiments import (
    ExperimentReport,
    ExperimentReportRow,
    ExperimentScoreRoundResult,
    ExperimentTrainResult,
    create_experiment,
)
from numereng.features.training.errors import TrainingError

EXPERIMENT_ID = "2026-06-10_hardening-exp"


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
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Hardening")
    experiment_dir = experiment.manifest_path.parent
    _write_training_config(experiment_dir / "configs" / "seed.json")
    return store_root, experiment_dir


def _state_path(experiment_dir: Path) -> Path:
    return experiment_dir / "agentic_research" / "state.json"


def _read_state(experiment_dir: Path) -> dict[str, object]:
    return json.loads(_state_path(experiment_dir).read_text(encoding="utf-8"))


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


def _write_run_manifest(store_root: Path, run_id: str, *, experiment_id: str) -> None:
    run_dir = store_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "FINISHED",
        "experiment_id": experiment_id,
        "output": {"predictions_name": "predictions"},
    }
    (run_dir / "run.json").write_text(json.dumps(manifest), encoding="utf-8")


def _mutation_response(*, value: object = 0.02) -> str:
    return json.dumps(
        {
            "decision_form": {
                "action": "run",
                "learning": "Prior round informed this probe.",
                "belief_update": "The probed axis matters.",
                "next_hypothesis": "This mutation improves the primary metric.",
                "parent_config": "seed.json",
                "changes": [{"path": "model.params.learning_rate", "value": value, "reason": "probe"}],
                "stop_reason": None,
            },
            "round_markdown": "# rNNN Research State\n\nMemo.",
            "experiment_markdown": None,
        }
    )


# ---------------------------------------------------------------------------
# Mock-seam layer (approved seams only). Extends the contract-file pattern with
# a score_queue (inject scoring failures) and metric=None rows (unscored runs).
# ---------------------------------------------------------------------------


@dataclass
class _Seams:
    store_root: Path
    experiment_dir: Path
    rows: list[ExperimentReportRow] = field(default_factory=list)
    # (run_id, metric_or_None) tuples, or a BaseException to raise (SystemExit allowed).
    train_queue: list[object] = field(default_factory=list)
    # Raw-response strings, or exceptions to raise from the LLM transport.
    llm_queue: list[object] = field(default_factory=list)
    # One entry consumed per score call: an Exception to raise, a (run_id, metric)
    # tuple to materialize that run's metric then succeed, or None to succeed.
    score_queue: list[object] = field(default_factory=list)
    score_calls: list[tuple[str, str]] = field(default_factory=list)

    def set_metric(self, run_id: str, metric: float | None) -> None:
        self.rows = [row for row in self.rows if row.run_id != run_id]
        self.add_row(run_id, metric)

    def add_row(self, run_id: str, metric: float | None) -> None:
        self.rows.append(
            ExperimentReportRow(
                run_id=run_id,
                status="FINISHED",
                created_at="2026-06-10T00:00:00+00:00",
                metric_value=metric,
                corr_mean=0.01 if metric is not None else None,
                mmc_mean=None,
                cwmm_mean=None,
                bmc_mean=None,
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
        if isinstance(item, BaseException):
            raise item
        run_id, metric = item  # type: ignore[misc]
        seams.add_row(str(run_id), metric)  # type: ignore[arg-type]
        return ExperimentTrainResult(
            experiment_id=EXPERIMENT_ID,
            run_id=str(run_id),
            predictions_path=store_root / "runs" / str(run_id) / "predictions.parquet",
            results_path=store_root / "runs" / str(run_id) / "results.json",
        )

    def fake_score(**kwargs: object) -> ExperimentScoreRoundResult:
        round_label = str(kwargs.get("round"))
        seams.score_calls.append((round_label, str(kwargs.get("stage"))))
        if seams.score_queue:
            item = seams.score_queue.pop(0)
            if isinstance(item, Exception):
                raise item
            if isinstance(item, tuple):
                run_id, metric = item
                seams.set_metric(str(run_id), float(metric))
        return ExperimentScoreRoundResult(
            experiment_id=EXPERIMENT_ID,
            round=round_label,
            stage="post_training_core",
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
# Path 1 — score failure after a successful train
# ---------------------------------------------------------------------------


def test_score_failure_after_train_fails_round_but_run_plan_row_survives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract: the run_plan row is written BEFORE scoring, so a score failure
    leaves a durable recovery path (`experiment score-round` can rescore later)."""
    store_root, experiment_dir = _setup_experiment(tmp_path)
    seams = _install_seams(monkeypatch, store_root, experiment_dir)
    seams.add_row("run-0", 0.10)
    seams.llm_queue = [_mutation_response(value=0.02)]
    seams.train_queue = [("run-1", 0.13)]
    seams.score_queue = [RuntimeError("score_exploded")]

    result = run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=1)

    assert result.rounds[0].status == "failed"
    assert "score_exploded" in result.rounds[0].learning
    # Contract: config + run_plan row survive the score failure (manual rescore possible).
    assert "config_001.json" in _config_files(experiment_dir)
    plan_rows = _run_plan_rows(experiment_dir)
    assert [(row["round"], row["config_path"]) for row in plan_rows] == [("r001", "configs/config_001.json")]
    assert seams.score_calls == [("r001", "post_training_core")]
    # HOLE: the round trained run-1 (compute spent, run linked by train_experiment)
    # but the failed-round record drops the run_id entirely — the memory says no
    # run happened. state counts it as a failure with no run.
    assert result.rounds[0].run_id is None
    state = _read_state(experiment_dir)
    assert state["failed_rounds_counter"] == 1
    assert state["total_rounds_completed"] == 0
    assert state["last_run_id"] is None


def test_score_failure_then_identical_proposal_dedup_skips_without_rescore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # HOLE (deferred to rebuild): after a score failure the config file is already
    # on disk, so when the model re-proposes the same mutation the duplicate-by-hash
    # gate soft-skips it BEFORE training — the reuse-rescore path never gets a
    # chance, so the loop never retries scoring. The trained run stays unscored
    # (invisible to best-tracking), and the dup-skip memo claims "the existing
    # config's score already stands in the report" — which is false here.
    store_root, experiment_dir = _setup_experiment(tmp_path)
    seams = _install_seams(monkeypatch, store_root, experiment_dir)
    seams.add_row("run-0", 0.10)
    same = _mutation_response(value=0.02)
    seams.llm_queue = [same, same]
    seams.train_queue = [("run-1", 0.13)]
    seams.score_queue = [RuntimeError("score_exploded")]

    result = run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=2)

    assert [round_.status for round_ in result.rounds] == ["failed", "skipped"]
    assert "agentic_research_candidate_duplicate" in result.rounds[1].learning
    # Scoring was attempted exactly once (the failed attempt) — never retried.
    assert seams.score_calls == [("r001", "post_training_core")]
    state = _read_state(experiment_dir)
    # The dup skip counts as a completed round and resets the failure counter,
    # so this state can repeat indefinitely without tripping the bail.
    assert state["failed_rounds_counter"] == 0
    assert state["total_rounds_completed"] == 1


# ---------------------------------------------------------------------------
# Path 2 — hard crash mid-round (SystemExit between config write and train)
# ---------------------------------------------------------------------------


def test_systemexit_mid_round_escapes_without_state_save_then_resume_dedup_skips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_root, experiment_dir = _setup_experiment(tmp_path)
    seams = _install_seams(monkeypatch, store_root, experiment_dir)
    seams.add_row("run-0", 0.10)
    seams.llm_queue = [_mutation_response(value=0.02)]
    seams.train_queue = [SystemExit(1)]  # crash after config write, before training

    with pytest.raises(SystemExit):
        run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=1)

    # Contract: round numbering survives the crash — next_round_number was never
    # advanced, so the resumed session reuses r001 (no gaps, no double-count).
    state = _read_state(experiment_dir)
    assert state["next_round_number"] == 1
    assert state["total_rounds_completed"] == 0
    # HOLE: state.json still says "running" after the process died — a status-only
    # watcher cannot tell a live session from a dead one.
    assert state["status"] == "running"
    assert get_research_status(store_root=store_root, experiment_id=EXPERIMENT_ID).status == "running"
    # The crash left an orphaned config (written before train) with no round record.
    assert "config_001.json" in _config_files(experiment_dir)
    assert not (_rounds_dir(experiment_dir) / "r001.md").exists()
    assert _run_plan_rows(experiment_dir) == []

    # Resume: the model retries the SAME mutation (the likely move after a crash).
    seams.llm_queue = [_mutation_response(value=0.02)]
    resumed = run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=1)

    assert resumed.rounds[0].round_label == "r001"  # numbering consistent
    # HOLE: the crash orphan hash-collides with the retry, so the mutation is
    # soft-skipped and NEVER trained — the candidate is permanently untestable
    # until a human deletes the orphaned config file.
    assert resumed.rounds[0].status == "skipped"
    assert "agentic_research_candidate_duplicate" in resumed.rounds[0].learning
    assert seams.train_queue == []  # train was never attempted on resume
    state = _read_state(experiment_dir)
    assert state["next_round_number"] == 2
    assert state["total_rounds_completed"] == 1


# ---------------------------------------------------------------------------
# Path 3 — state.json corruption
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("corruption", ['{"status": "run', "[]", "null"], ids=["truncated", "list", "null"])
def test_corrupt_state_fails_loudly_and_never_reinitializes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    """Contract: a corrupt state.json must fail loudly with a stable error token.
    It must NOT be silently reinitialized (which would erase champion/round
    counters) and must NOT be overwritten by the failed call."""
    store_root, experiment_dir = _setup_experiment(tmp_path)
    seams = _install_seams(monkeypatch, store_root, experiment_dir)
    seams.add_row("run-0", 0.10)
    seams.llm_queue = [_mutation_response(value=0.02)]
    seams.train_queue = [("run-1", 0.11)]
    run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=1)

    _state_path(experiment_dir).write_text(corruption, encoding="utf-8")

    with pytest.raises(AgenticResearchError) as run_exc:
        run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=1)
    assert "agentic_research_state_invalid" in str(run_exc.value)
    with pytest.raises(AgenticResearchError) as status_exc:
        get_research_status(store_root=store_root, experiment_id=EXPERIMENT_ID)
    assert "agentic_research_state_invalid" in str(status_exc.value)
    # The corrupt file is preserved byte-for-byte for forensics — no blind rewrite.
    assert _state_path(experiment_dir).read_text(encoding="utf-8") == corruption


# ---------------------------------------------------------------------------
# Path 4 — LLM transport / parse failures
# ---------------------------------------------------------------------------


def test_llm_transport_failure_writes_debug_dumps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Contract: a transport failure dumps the prompt + error beside the round
    artifacts (these debug files saved the last postmortem)."""
    store_root, experiment_dir = _setup_experiment(tmp_path)
    seams = _install_seams(monkeypatch, store_root, experiment_dir)
    seams.add_row("run-0", 0.10)
    seams.llm_queue = [AgenticResearchError("agentic_research_codex_failed:1:stream disconnected")]

    result = run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=1)

    assert result.rounds[0].status == "failed"
    rounds_dir = _rounds_dir(experiment_dir)
    assert (rounds_dir / "r001.debug.prompt.md").is_file()
    assert "stream disconnected" in (rounds_dir / "r001.debug.error.txt").read_text(encoding="utf-8")
    state = _read_state(experiment_dir)
    assert state["failed_rounds_counter"] == 1


def test_llm_broken_json_fails_round_and_preserves_raw_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract: a parse failure is a failed round (counts toward the bail), writes
    no config, and preserves the raw response durably for postmortem."""
    store_root, experiment_dir = _setup_experiment(tmp_path)
    seams = _install_seams(monkeypatch, store_root, experiment_dir)
    seams.add_row("run-0", 0.10)
    raw = "```json\n{ this is not json"
    seams.llm_queue = [raw]

    result = run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=1)

    assert result.rounds[0].status == "failed"
    assert "agentic_research_json" in result.rounds[0].learning
    assert _config_files(experiment_dir) == {"seed.json"}  # no partial config artifact
    rounds_dir = _rounds_dir(experiment_dir)
    assert (rounds_dir / "r001.debug.llm_response.txt").read_text(encoding="utf-8") == raw
    assert (rounds_dir / "r001.debug.prompt.md").is_file()
    assert (rounds_dir / "r001.md").is_file()  # failure still memorialized
    state = _read_state(experiment_dir)
    assert state["failed_rounds_counter"] == 1
    assert state["total_rounds_completed"] == 0


def test_llm_schema_violations_fail_rounds_and_accumulate_toward_bail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract: structurally-valid JSON violating the decision schema (missing
    decision_form; forbidden `stop` action) fails the round with a stable token,
    writes no config, and counts toward the consecutive-failure bail."""
    store_root, experiment_dir = _setup_experiment(tmp_path)
    seams = _install_seams(monkeypatch, store_root, experiment_dir)
    seams.add_row("run-0", 0.10)
    missing_form = json.dumps({"round_markdown": "# r Memo", "decision_form": None})
    stop_action = json.dumps(
        {
            "decision_form": {"action": "stop", "learning": "x", "belief_update": "y", "stop_reason": "done"},
            "round_markdown": "# r Memo",
        }
    )
    seams.llm_queue = [missing_form, stop_action]

    result = run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=2)

    assert [round_.status for round_ in result.rounds] == ["failed", "failed"]
    assert "agentic_research_decision_form_missing" in result.rounds[0].learning
    assert "agentic_research_action_invalid" in result.rounds[1].learning
    assert _config_files(experiment_dir) == {"seed.json"}
    state = _read_state(experiment_dir)
    assert state["failed_rounds_counter"] == 2  # accumulates: parse failures can trip the bail


# ---------------------------------------------------------------------------
# Path 5 — baseline-round failures
# ---------------------------------------------------------------------------


def test_baseline_train_failure_retries_then_bails_resumably(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Contract: repeated baseline train failures hit the consecutive-failure bail
    instead of retrying forever; the bail is resumable."""
    store_root, experiment_dir = _setup_experiment(tmp_path)
    seams = _install_seams(monkeypatch, store_root, experiment_dir)
    seams.train_queue = [TrainingError(f"gpu_oom_{i}") for i in range(10)]

    result = run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=10)

    assert len(result.rounds) == 5  # bailed, did not burn all 10 rounds
    assert all(round_.status == "failed" for round_ in result.rounds)
    assert result.stop_reason == "consecutive_failures:5"
    # HOLE (minor): every baseline attempt writes a fresh copy of the seed config
    # BEFORE training, so the 5 failures leave 5 identical orphan configs whose
    # hashes also pre-seed the duplicate gate.
    assert _config_files(experiment_dir) == {
        "seed.json",
        "config_001.json",
        "config_002.json",
        "config_003.json",
        "config_004.json",
        "config_005.json",
    }


def test_baseline_score_failure_reused_unscored_run_is_rescored_on_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract (reuse-without-scoring fix): baseline trains OK but scoring fails.
    The retry round hash-collides with the FINISHED run dir and takes the
    same-experiment reuse path — because the reused run has no primary metric on
    disk, the round must SCORE it (not skip), so the round ends with the linked
    run scored and the session leaves the baseline branch."""
    store_root, experiment_dir = _setup_experiment(tmp_path)
    seams = _install_seams(monkeypatch, store_root, experiment_dir)
    _write_run_manifest(store_root, "run-1", experiment_id=EXPERIMENT_ID)
    seams.train_queue = [("run-1", None), TrainingError("training_run_dir_not_fresh:run-1:cfghash")]
    seams.score_queue = [RuntimeError("score_exploded"), ("run-1", 0.12)]

    result = run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=2)

    assert [round_.status for round_ in result.rounds] == ["failed", "completed"]
    # The failed-round record hardcodes action="run" even though the round was a
    # baseline (run.py:1539) — minor misleading-state wart, pinned as-is.
    assert [round_.action for round_ in result.rounds] == ["run", "baseline"]
    # Scoring ran on BOTH rounds: the original failure AND the reuse retry.
    assert seams.score_calls == [("r001", "post_training_core"), ("r002", "post_training_core")]
    # The reuse round ends scored — the linked run carries its metric.
    assert result.rounds[1].run_id == "run-1"
    assert result.rounds[1].metric_value == pytest.approx(0.12)
    state = _read_state(experiment_dir)
    assert state["failed_rounds_counter"] == 0
    assert state["total_rounds_completed"] == 1
    manifest = json.loads((experiment_dir / "experiment.json").read_text(encoding="utf-8"))
    assert "run-1" in manifest["runs"]


def test_baseline_score_failure_with_persistent_scoring_outage_bails_honestly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract (reuse-without-scoring fix): if scoring keeps failing, reuse rounds
    FAIL instead of completing unscored — no infinite metric=None "completed"
    loop. The consecutive-failure bail trips and the session stops resumably."""
    store_root, experiment_dir = _setup_experiment(tmp_path)
    seams = _install_seams(monkeypatch, store_root, experiment_dir)
    _write_run_manifest(store_root, "run-1", experiment_id=EXPERIMENT_ID)
    seams.train_queue = [("run-1", None)] + [
        TrainingError("training_run_dir_not_fresh:run-1:cfghash") for _ in range(9)
    ]
    seams.score_queue = [RuntimeError(f"score_exploded_{i}") for i in range(10)]

    result = run_research(store_root=store_root, experiment_id=EXPERIMENT_ID, max_rounds=10)

    assert len(result.rounds) == 5  # bailed, did not burn all 10 rounds
    assert all(round_.status == "failed" for round_ in result.rounds)
    assert result.stop_reason == "consecutive_failures:5"
    # Every round honestly attempted scoring before failing.
    assert [call[0] for call in seams.score_calls] == ["r001", "r002", "r003", "r004", "r005"]
    state = _read_state(experiment_dir)
    assert state["failed_rounds_counter"] == 5
    assert state["total_rounds_completed"] == 0
