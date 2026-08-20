"""Shared fixtures for closeout-chain unit tests.

Builds a completed agentic experiment on disk (manifest, agentic_research/state.json + non-empty
journal.jsonl, valid configs, round memos) plus a valid research-memory root, so gate/evidence/runner
tests can exercise the real readers without any network or training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from numereng.features.experiments import ExperimentRecord, create_experiment, get_experiment

EXPERIMENT_ID = "2026-07-13_closeout-exp"


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


def write_memory_root(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "CURRENT.md").write_text("# CURRENT\n", encoding="utf-8")
    topics = root / "topics"
    topics.mkdir(exist_ok=True)
    for name in ("ensembling", "features", "hyperparameters", "models", "neutralization-exposure", "targets"):
        (topics / f"{name}.md").write_text(f"# {name}\n", encoding="utf-8")
    return root


def build_finalize_memo(*, experiment_id: str, believed_best_config: str) -> str:
    headings = (
        "Verdict",
        "Evidence Status And Caveats",
        "Candidate Hierarchy",
        "Metric Conflicts",
        "Sweep Discipline Audit",
        "Design-Space Roles",
        "Implications For Future Work",
        "Master-Ledger Update",
    )
    body = (
        f"Experiment {experiment_id}; the believed-best recipe is {believed_best_config}. "
        "This is candidate-tier evidence; the within-lane BMC200 metric is a ranker, not a deploy "
        "signal, and scout outputs are candidates. " * 12
    )
    return "\n\n".join([f"## {heading}\n\n{body}" for heading in headings])


def valid_envelope(*, experiment_id: str, believed_best_config: str) -> str:
    memo = build_finalize_memo(experiment_id=experiment_id, believed_best_config=believed_best_config)
    return json.dumps(
        {"files": [{"path": "EXPERIMENT.closeout.md", "content": memo}], "notes": "candidate hierarchy confirmed"}
    )


TOPIC_NAMES = ("ensembling", "features", "hyperparameters", "models", "neutralization-exposure", "targets")

_EXTRACT_HEADINGS = (
    "Experiment-Specific Takeaway",
    "Evidence Snapshot",
    "Evidence Level",
    "Design-Space Role",
    "Confounds",
    "What Not To Infer",
    "Not Established",
    "Scope And Caveats",
    "Future Implication",
    "Master Ledger Update",
)


def extract_topic_file(topic: str) -> str:
    """A topic file carrying all ten required headings, a valid evidence level, and a design role."""
    values = {"Evidence Level": "computed metric", "Design-Space Role": "varied"}
    parts = []
    for heading in _EXTRACT_HEADINGS:
        body = values.get(heading, f"The {topic} axis: BMC200 mean was 0.0041 for the believed-best recipe.")
        parts.append(f"## {heading}\n\n{body}")
    return "\n\n".join(parts) + "\n"


def extract_readme(experiment_id: str) -> str:
    links = "\n".join(f"- [{topic}]({topic}.md)" for topic in TOPIC_NAMES)
    return f"# Research Memory Branch — {experiment_id}\n\nExperiment {experiment_id} branch.\n\n{links}\n"


def valid_extract_envelope(*, experiment_id: str) -> str:
    files = [{"path": "README.md", "content": extract_readme(experiment_id)}]
    files += [{"path": f"{topic}.md", "content": extract_topic_file(topic)} for topic in TOPIC_NAMES]
    return json.dumps({"files": files, "notes": "branch extracted"})


def ledger_text(topic: str, *, prior_entry: bool = True) -> str:
    """A master ledger mirroring the real three-section anatomy (Overview, Best Understanding, Learnings)."""
    head = (
        f"# {topic}\n\n**Updated** 2026-07-13\n\n"
        f"## Current Overview\n\nPrior overview for {topic}.\n\n"
        f"## Current Best Understanding\n\nPrior best understanding for {topic}.\n\n"
    )
    learnings = "## Append-Only Experiment Learnings\n"
    if prior_entry:
        learnings += (
            "\n### 2026-06-01_prior-exp\n"
            f"- Source: [branch](../experiments/2026-06-01_prior-exp/{topic}.md)\n"
            "- Learning: prior computed-metric result.\n"
        )
    return head + learnings


def write_ledger_memory_root(root: Path) -> Path:
    """A memory root whose topic ledgers have the real overview/learnings anchor structure."""
    write_memory_root(root)
    for topic in TOPIC_NAMES:
        (root / "topics" / f"{topic}.md").write_text(ledger_text(topic), encoding="utf-8")
    return root


def synthesize_entry(topic: str, experiment_id: str) -> str:
    return (
        f"### {experiment_id}\n"
        f"- Source: [branch](../experiments/{experiment_id}/{topic}.md)\n"
        "- Learning: computed metric — lr 0.01 led BMC200 within the seed-noise floor.\n"
    )


def synthesize_current_md(experiment_id: str) -> str:
    body = (
        f"Experiment {experiment_id} folded into the frontier. Within-lane BMC200 is a candidate "
        "ranker, not a deploy signal; scout outputs stay candidates until confirmed on full data. "
    ) * 12
    return (
        f"# CURRENT\n\n## Compressed Frontier\n\n{body}\n\n"
        f"Full record: experiments/{experiment_id}/README.md\n\n"
        f"## Comparison Anchors\n\n{body}\n\n"
        f"## Current Constraints\n\n{body}\n"
    )


def valid_synthesize_envelope(
    *,
    experiment_id: str,
    topics: tuple[str, ...] = TOPIC_NAMES,
    overview_topics: tuple[str, ...] = (),
    best_understanding_topics: tuple[str, ...] = (),
) -> str:
    deltas = []
    for topic in topics:
        overview = f"New overview for {topic}." if topic in overview_topics else None
        best_understanding = f"New best understanding for {topic}." if topic in best_understanding_topics else None
        deltas.append(
            {
                "topic": topic,
                "new_entry_markdown": synthesize_entry(topic, experiment_id),
                "overview_replacement_markdown": overview,
                "best_understanding_replacement_markdown": best_understanding,
            }
        )
    return json.dumps(
        {"deltas": deltas, "current_md": synthesize_current_md(experiment_id), "notes": "frontier updated"}
    )


# --------------------------------------------------------------------------- #
# CLASSIFY builder
# --------------------------------------------------------------------------- #
def valid_classification(
    *,
    disposition: str = "master",
    relevant_topics: tuple[str, ...] = TOPIC_NAMES,
) -> str:
    return json.dumps(
        {
            "disposition": disposition,
            "relevant_topics": list(relevant_topics),
            "rationale": "The evidence has durable research-memory value.",
        }
    )


# --------------------------------------------------------------------------- #
# Fixture object
# --------------------------------------------------------------------------- #
@dataclass
class CloseoutFixture:
    store_root: Path
    experiment_dir: Path
    experiment_id: str
    believed_best_config: str
    memory_root: Path

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

    memory_root = write_memory_root(store_root / "notes" / "__RESEARCH_MEMORY__")

    return CloseoutFixture(
        store_root=store_root,
        experiment_dir=experiment_dir,
        experiment_id=EXPERIMENT_ID,
        believed_best_config="config_001.json",
        memory_root=memory_root,
    )
