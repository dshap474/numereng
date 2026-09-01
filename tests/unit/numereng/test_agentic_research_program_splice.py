"""CORE splice helper plus the `research program check|resplice` engine, API, and CLI contract.

A CORE edit in PROGRAM.md invalidates every custom program's byte-verbatim CORE copy; these tests
pin the mechanical fix: only CORE section bodies move, strategy sections and the preamble survive
untouched, a backup is kept, and the CLI exit code reports drift.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from numereng.agentic_research.engine import boundary, program
from numereng.agentic_research.engine import types as ar_types
from numereng.cli.commands.research import handle_research_command
from numereng.features.experiments import ExperimentRecord, create_experiment, get_experiment

EXPERIMENT_ID = "2026-09-01_program-splice"
STRATEGY_KEYS = ("0.", "4.", "6.")
BASE = ar_types.PROGRAM_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _custom_program(base: str = BASE) -> str:
    """A pristine self-contained custom program: base CORE plus rewritten strategy sections."""
    out: list[str] = []
    for key, lines in ar_types._split_program_sections(base):
        if key in STRATEGY_KEYS:
            out.extend([lines[0], "", f"Experiment-specific strategy for section {key}.", ""])
        else:
            out.extend(lines)
    return "\n".join(out) + "\n"


def _drift(text: str, key: str = "5.") -> str:
    """Append a stale line to one CORE section body."""
    out: list[str] = []
    for section_key, lines in ar_types._split_program_sections(text):
        out.extend(lines)
        if section_key == key:
            out.append("STALE LINE THAT NO LONGER MATCHES PROGRAM.md")
    return "\n".join(out) + "\n"


def _experiment(tmp_path: Path, *, metadata: dict[str, object] | None = None) -> ExperimentRecord:
    store_root = tmp_path / ".numereng"
    created = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Splice")
    if metadata:
        payload = json.loads(created.manifest_path.read_text(encoding="utf-8"))
        payload["metadata"] = {**payload.get("metadata", {}), **metadata}
        created.manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return get_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID)


def _write_program(experiment: ExperimentRecord, name: str, text: str) -> Path:
    path = experiment.manifest_path.parent / "agentic_research" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# splice_core_sections
# ---------------------------------------------------------------------------


def test_splice_restores_drifted_core_and_keeps_strategy_sections() -> None:
    pristine = _custom_program()
    drifted = _drift(pristine, "5.")
    assert ar_types.first_diverging_core_section(drifted, BASE) == "5."

    spliced = ar_types.splice_core_sections(drifted, BASE)
    assert ar_types.first_diverging_core_section(spliced, BASE) is None
    # Byte-exact round trip: the only thing that moved was the stale CORE body.
    assert spliced == pristine
    for key in STRATEGY_KEYS:
        assert f"Experiment-specific strategy for section {key}." in spliced


def test_splice_is_a_no_op_on_an_in_sync_program() -> None:
    pristine = _custom_program()
    assert ar_types.splice_core_sections(pristine, BASE) == pristine
    assert ar_types.splice_core_sections(BASE, BASE) == BASE


def test_splice_preserves_preamble_and_trailing_newline_choice() -> None:
    pristine = "preamble line before any heading\n" + _custom_program()
    drifted = _drift(pristine, "9.")
    spliced = ar_types.splice_core_sections(drifted, BASE)
    assert spliced.startswith("preamble line before any heading\n")
    assert spliced == pristine
    assert not ar_types.splice_core_sections(drifted.rstrip("\n"), BASE).endswith("\n")


def test_splice_refuses_to_invent_a_missing_core_section() -> None:
    without_8 = "\n".join(
        "\n".join(lines) for key, lines in ar_types._split_program_sections(_custom_program()) if key != "8."
    )
    with pytest.raises(ar_types.AgenticResearchValidationError) as exc:
        ar_types.splice_core_sections(without_8, BASE)
    assert str(exc.value) == "agentic_research_program_core_missing:program:8."


# ---------------------------------------------------------------------------
# engine: check_program_core / resplice_program_core
# ---------------------------------------------------------------------------


def test_base_program_is_always_in_sync_and_never_written(tmp_path: Path) -> None:
    _experiment(tmp_path)
    checked = program.check_program_core(store_root=tmp_path / ".numereng", experiment_id=EXPERIMENT_ID)
    assert checked.is_base_program and checked.in_sync and checked.diverging_section is None
    resplice = program.resplice_program_core(store_root=tmp_path / ".numereng", experiment_id=EXPERIMENT_ID)
    assert resplice.written is False and resplice.backup_path is None


def test_resplice_rewrites_drifted_program_with_backup_and_is_idempotent(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path, metadata={ar_types.PROGRAM_METADATA_KEY: "focus.md"})
    pristine = _custom_program()
    drifted = _drift(pristine, "10.")
    program_path = _write_program(experiment, "focus.md", drifted)
    store_root = tmp_path / ".numereng"

    checked = program.check_program_core(store_root=store_root, experiment_id=EXPERIMENT_ID)
    assert checked.program_path == program_path
    assert not checked.is_base_program and checked.diverging_section == "10." and not checked.in_sync

    result = program.resplice_program_core(store_root=store_root, experiment_id=EXPERIMENT_ID)
    assert result.written and result.in_sync and result.backup_path is not None
    assert result.backup_path.suffix == ".bak"  # a `.md` sibling would look like an active program
    assert result.backup_path.read_text(encoding="utf-8") == drifted
    assert program_path.read_text(encoding="utf-8") == pristine

    again = program.resplice_program_core(store_root=store_root, experiment_id=EXPERIMENT_ID)
    assert again.written is False and again.in_sync
    assert len(list(program_path.parent.glob("*.bak"))) == 1
    assert program.first_core_drift(get_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID)) is None


# ---------------------------------------------------------------------------
# CLI: numereng research program check|resplice
# ---------------------------------------------------------------------------


def test_cli_check_exits_one_on_drift_then_resplice_and_check_exit_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    experiment = _experiment(tmp_path, metadata={ar_types.PROGRAM_METADATA_KEY: "focus.md"})
    _write_program(experiment, "focus.md", _drift(_custom_program(), "2.1"))
    common = ["--experiment-id", EXPERIMENT_ID, "--workspace", str(tmp_path)]

    assert handle_research_command(["program", "check", *common, "--format", "json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["in_sync"] is False and payload["diverging_section"] == "2.1" and payload["written"] is False

    assert handle_research_command(["program", "resplice", *common]) == 0
    out = capsys.readouterr().out
    assert "written: True" in out and "in_sync: True" in out and ".bak" in out

    assert handle_research_command(["program", "check", *common]) == 0
    assert "diverging_section: none" in capsys.readouterr().out


def test_cli_program_argument_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert handle_research_command(["program", "frobnicate", "--experiment-id", "x"]) == 2
    assert "unknown research program action" in capsys.readouterr().err
    assert handle_research_command(["program", "check", "--workspace", str(tmp_path)]) == 2
    assert "missing required argument: --experiment-id" in capsys.readouterr().err
    assert handle_research_command(["program", "check", "--experiment-id", "x", "--format", "yaml"]) == 2
    # Unknown experiment surfaces as a PackageError -> exit 1, not a traceback.
    assert (
        handle_research_command(["program", "check", "--experiment-id", "missing", "--workspace", str(tmp_path)]) == 1
    )


# ---------------------------------------------------------------------------
# boundary: the evaluator block is not globally mutable
# ---------------------------------------------------------------------------


def test_training_engine_paths_are_outside_the_global_allowlist(tmp_path: Path) -> None:
    assert not any(path.startswith("training.engine") for path in boundary.ALLOWED_CHANGE_PATHS)
    assert "data.dataset_variant" not in boundary.ALLOWED_CHANGE_PATHS
    experiment = _experiment(
        tmp_path,
        metadata={ar_types.ALLOWED_PATHS_METADATA_KEY: ["model.params.*", "training.engine.profile"]},
    )
    with pytest.raises(ar_types.AgenticResearchValidationError) as exc:
        boundary.program_allowed_paths(experiment)
    assert str(exc.value) == "agentic_research_allowed_path_invalid:'training.engine.profile'"
