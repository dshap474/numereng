"""Shared fixtures for closeout unit tests.

Builds a completed agentic experiment on disk (manifest, agentic_research/state.json + non-empty
journal.jsonl, valid configs, round memos), so gate/evidence/runner tests can exercise the real
readers without any network or training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from numereng.features.experiments import ExperimentRecord, create_experiment, get_experiment

EXPERIMENT_ID = "2026-07-13_closeout-exp"

MEMO_HEADINGS = (
    "Verdict",
    "Evidence And Gaps",
    "Candidates",
    "Metric Conflicts",
    "Search Audit",
    "Design-Space Roles",
    "Implications",
    "Memory Notes",
)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def write_training_config(path: Path, *, learning_rate: float = 0.01, target_col: str = "target") -> None:
    payload = {
        "data": {"data_version": "v5.2", "dataset_variant": "non_downsampled", "target_col": target_col},
        "model": {"type": "LGBMRegressor", "params": {"learning_rate": learning_rate, "random_state": 42}},
        "training": {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def build_finalize_memo(*, experiment_id: str, believed_best_config: str) -> str:
    """A memo shaped like a real finalize response: plain markdown, `## Verdict` first."""
    body = (
        f"Experiment {experiment_id}; the believed-best recipe is {believed_best_config}. "
        "This is candidate-tier evidence; the within-lane BMC200 metric is a ranker, not a deploy "
        "signal, and scout outputs are candidates. " * 12
    )
    return "\n\n".join(f"## {heading}\n\n{body}" for heading in MEMO_HEADINGS)


def install_fake_llm(
    monkeypatch: pytest.MonkeyPatch, fixture: CloseoutFixture, *, raw: str | None = None
) -> dict[str, object]:
    """Stub the research LLM with a valid memo (or a fixed ``raw`` payload); record the calls."""
    from numereng.agentic_research.engine import llm

    calls: dict[str, object] = {"n": 0, "prompts": [], "kwargs": []}

    def fake(**kwargs: object) -> tuple[str, str]:
        calls["n"] = int(calls["n"]) + 1
        calls["prompts"].append(kwargs.get("prompt"))  # type: ignore[union-attr]
        calls["kwargs"].append(dict(kwargs))  # type: ignore[union-attr]
        return (raw if raw is not None else fixture.memo()), "codex-exec"

    monkeypatch.setattr(llm, "call_research_llm", fake)
    return calls


# --------------------------------------------------------------------------- #
# Fixture object
# --------------------------------------------------------------------------- #
@dataclass
class CloseoutFixture:
    store_root: Path
    experiment_dir: Path
    experiment_id: str
    believed_best_config: str

    def experiment(self) -> ExperimentRecord:
        return get_experiment(store_root=self.store_root, experiment_id=self.experiment_id)

    def agentic_dir(self) -> Path:
        return self.experiment_dir / "agentic_research"

    def closeout_dir(self) -> Path:
        return self.agentic_dir() / "closeout"

    def journal_path(self) -> Path:
        return self.agentic_dir() / "journal.jsonl"

    def state_path(self) -> Path:
        return self.agentic_dir() / "state.json"

    def memo(self) -> str:
        return build_finalize_memo(experiment_id=self.experiment_id, believed_best_config=self.believed_best_config)

    def set_manifest_status(self, status: str) -> None:
        manifest_path = self.experiment_dir / "experiment.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = status
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def set_budget(self, rounds: int) -> None:
        manifest_path = self.experiment_dir / "experiment.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault("metadata", {})["agentic_research_budget_rounds"] = rounds
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def set_run_status(self, status: str) -> None:
        state = json.loads(self.state_path().read_text(encoding="utf-8"))
        state["status"] = status
        self.state_path().write_text(json.dumps(state), encoding="utf-8")


@pytest.fixture
def closeout_fixture(tmp_path: Path) -> CloseoutFixture:
    store_root = tmp_path / ".numereng"
    experiment = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Closeout")
    experiment_dir = experiment.manifest_path.parent

    configs_dir = experiment_dir / "configs"
    write_training_config(configs_dir / "config_001.json", learning_rate=0.01)
    write_training_config(configs_dir / "config_002.json", learning_rate=0.02)

    agentic = experiment_dir / "agentic_research"
    agentic.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "round": 1,
            "config": "config_001.json",
            "parent_config": "config_001.json",
            "seed": 42,
            "metric": 0.0041,
            "fnc": 0.02,
            "benchmark_corr": 0.3,
            "status": "completed",
            "run_id": "run-1",
            "wall_seconds": 120.0,
            "changes": [{"path": "model.params.learning_rate", "value": 0.01}],
        },
        {
            "round": 2,
            "config": "config_002.json",
            "parent_config": "config_001.json",
            "seed": 42,
            "metric": 0.0035,
            "fnc": 0.01,
            "benchmark_corr": 0.4,
            "status": "completed",
            "run_id": "run-2",
            "wall_seconds": 130.0,
            "changes": [{"path": "model.params.learning_rate", "value": 0.02}],
        },
        {
            "round": 3,
            "config": "config_002.json",
            "parent_config": "config_001.json",
            "seed": 17,
            "metric": 0.0037,
            "fnc": 0.015,
            "status": "completed",
            "run_id": "run-3",
            "wall_seconds": 125.0,
            "changes": [{"path": "model.params.learning_rate", "value": 0.02}],
        },
    ]
    with (agentic / "journal.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    state = {
        "schema_version": 2,
        "status": "stopped",
        "next_round_number": 4,
        "total_rounds_completed": 3,
        "believed_best": {"config": "config_001.json"},
        "champion": {"config": "config_001.json", "metric": 0.0041, "round": 1, "run_id": "run-1"},
    }
    (agentic / "state.json").write_text(json.dumps(state), encoding="utf-8")

    rounds_dir = agentic / "rounds"
    rounds_dir.mkdir(exist_ok=True)
    (rounds_dir / "r001.md").write_text("# r001\n\nBaseline round.\n", encoding="utf-8")
    (rounds_dir / "r002.md").write_text("# r002\n\nSWEEP ABANDONED because the axis was inert.\n", encoding="utf-8")

    (experiment_dir / "EXPERIMENT.md").write_text("# Champion State\n\nWorking set.\n", encoding="utf-8")

    return CloseoutFixture(
        store_root=store_root,
        experiment_dir=experiment_dir,
        experiment_id=EXPERIMENT_ID,
        believed_best_config="config_001.json",
    )
