"""Strategy-brief resolution, the round-prompt composition, and the tracked-prompt lint.

The round prompt is ``programs/PROGRAM.md`` with ``{{STRATEGY}}`` replaced by the experiment's
brief and ``{{CONTEXT_JSON}}`` replaced by the bounded context, so these tests pin the resolution
order (experiment-local brief wins), the two substitutions, and the placeholder invariant the
session-start pre-flight enforces.

USAGE:
    uv run pytest tests/unit/numereng/test_agentic_research_strategy_resolution.py -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

from numereng.agentic_research.engine import llm, memory
from numereng.agentic_research.engine import loop as research_loop
from numereng.agentic_research.engine import types as ar_types
from numereng.features.experiments import ExperimentRecord, create_experiment, get_experiment

EXPERIMENT_ID = "2026-07-05_strategy-resolution"


def _experiment(tmp_path: Path) -> ExperimentRecord:
    store_root = tmp_path / ".numereng"
    create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Resolution")
    return get_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID)


def _write_brief(experiment: ExperimentRecord, text: str) -> Path:
    path = memory.agentic_dir(experiment) / ar_types.STRATEGY_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Resolution order
# --------------------------------------------------------------------------- #


def test_missing_brief_resolves_to_the_generic_one(tmp_path: Path) -> None:
    assert memory.strategy_path(_experiment(tmp_path)) == ar_types.DEFAULT_STRATEGY_PATH


def test_experiment_brief_wins_over_the_generic_one(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    brief = _write_brief(experiment, "## This Experiment\n\nlocal brief\n")

    resolved = memory.strategy_path(experiment)
    assert resolved == brief
    assert resolved.read_text(encoding="utf-8") == "## This Experiment\n\nlocal brief\n"


def test_a_directory_at_the_brief_path_falls_back_to_the_generic_one(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    (memory.agentic_dir(experiment) / ar_types.STRATEGY_FILENAME).mkdir(parents=True)

    assert memory.strategy_path(experiment) == ar_types.DEFAULT_STRATEGY_PATH


# --------------------------------------------------------------------------- #
# Composition
# --------------------------------------------------------------------------- #


def test_render_prompt_substitutes_the_brief_and_the_context_once_each() -> None:
    prompt = llm.render_prompt({"champion": "r007"}, strategy_text="## This Experiment\n\nlocal brief\n")

    assert prompt.count("local brief") == 1
    assert prompt.count('"champion": "r007"') == 1
    assert ar_types.STRATEGY_PLACEHOLDER not in prompt
    assert ar_types.CONTEXT_PLACEHOLDER not in prompt
    assert prompt.index("local brief") < prompt.index('"champion": "r007"')


def test_preflight_accepts_the_generic_brief(tmp_path: Path) -> None:
    research_loop._prevalidate_prompt_placeholders(_experiment(tmp_path))


def test_preflight_rejects_a_brief_carrying_a_placeholder(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    _write_brief(experiment, f"## This Experiment\n\n{ar_types.CONTEXT_PLACEHOLDER}\n")

    with pytest.raises(ar_types.AgenticResearchValidationError, match="agentic_research_program_placeholder_invalid"):
        research_loop._prevalidate_prompt_placeholders(experiment)


# --------------------------------------------------------------------------- #
# Tracked-prompt lint
# --------------------------------------------------------------------------- #


def test_tracked_program_carries_each_placeholder_exactly_once() -> None:
    program_text = ar_types.PROGRAM_PATH.read_text(encoding="utf-8")

    assert program_text.count(ar_types.STRATEGY_PLACEHOLDER) == 1
    assert program_text.count(ar_types.CONTEXT_PLACEHOLDER) == 1


def test_tracked_generic_brief_exists_and_carries_no_placeholder() -> None:
    assert ar_types.DEFAULT_STRATEGY_PATH.is_file()
    brief_text = ar_types.DEFAULT_STRATEGY_PATH.read_text(encoding="utf-8")

    assert ar_types.STRATEGY_PLACEHOLDER not in brief_text
    assert ar_types.CONTEXT_PLACEHOLDER not in brief_text
