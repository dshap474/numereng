"""Program resolution order and the session-start CORE pre-flight check."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from numereng.agentic_research.engine import loop as research_loop
from numereng.agentic_research.engine import memory
from numereng.agentic_research.engine import types as ar_types
from numereng.features.experiments import ExperimentRecord, create_experiment, get_experiment

EXPERIMENT_ID = "2026-07-05_program-resolution"


def _experiment(tmp_path: Path, *, program: str | None = None) -> ExperimentRecord:
    store_root = tmp_path / ".numereng"
    created = create_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID, name="Resolution")
    if program is not None:
        manifest_path = created.manifest_path
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["metadata"] = {**payload.get("metadata", {}), ar_types.PROGRAM_METADATA_KEY: program}
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return get_experiment(store_root=store_root, experiment_id=EXPERIMENT_ID)


def _agentic_dir(experiment: ExperimentRecord) -> Path:
    path = experiment.manifest_path.parent / "agentic_research"
    path.mkdir(parents=True, exist_ok=True)
    return path


# --- resolution order -------------------------------------------------------


def test_unset_metadata_resolves_to_base_program(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    assert memory.program_path(experiment) == ar_types.PROGRAM_PATH


def test_experiment_folder_is_preferred_over_custom_programs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    experiment = _experiment(tmp_path, program="focus.md")
    (_agentic_dir(experiment) / "focus.md").write_text("# experiment copy\n", encoding="utf-8")
    legacy = tmp_path / "custom_programs"
    legacy.mkdir()
    (legacy / "focus.md").write_text("# legacy copy\n", encoding="utf-8")
    monkeypatch.setattr(ar_types, "CUSTOM_PROGRAM_DIR", legacy)

    resolved = memory.program_path(experiment)
    assert resolved == _agentic_dir(experiment) / "focus.md"
    assert resolved.read_text(encoding="utf-8") == "# experiment copy\n"


def test_falls_back_to_custom_programs_when_absent_from_experiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = _experiment(tmp_path, program="focus.md")
    legacy = tmp_path / "custom_programs"
    legacy.mkdir()
    (legacy / "focus.md").write_text("# legacy copy\n", encoding="utf-8")
    monkeypatch.setattr(ar_types, "CUSTOM_PROGRAM_DIR", legacy)

    assert memory.program_path(experiment) == legacy / "focus.md"


def test_missing_in_both_locations_names_both_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    experiment = _experiment(tmp_path, program="focus.md")
    legacy = tmp_path / "custom_programs"
    monkeypatch.setattr(ar_types, "CUSTOM_PROGRAM_DIR", legacy)

    with pytest.raises(ar_types.AgenticResearchValidationError) as excinfo:
        memory.program_path(experiment)
    message = str(excinfo.value)
    assert "agentic_research_program_missing:focus.md" in message
    assert str(experiment.manifest_path.parent / "agentic_research" / "focus.md") in message
    assert str(legacy / "focus.md") in message


@pytest.mark.parametrize("bad", ["sub/dir.md", "/abs/focus.md", "../focus.md"])
def test_path_separators_are_rejected(tmp_path: Path, bad: str) -> None:
    experiment = _experiment(tmp_path, program=bad)
    with pytest.raises(ar_types.AgenticResearchValidationError, match="agentic_research_program_invalid"):
        memory.program_path(experiment)


# --- pre-flight CORE check --------------------------------------------------


def test_preflight_passes_for_base_program(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    research_loop._prevalidate_program_core(experiment)  # unset metadata -> base program, exempt


def test_preflight_passes_for_verbatim_copy(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path, program="focus.md")
    base_text = ar_types.PROGRAM_PATH.read_text(encoding="utf-8")
    (_agentic_dir(experiment) / "focus.md").write_text(base_text, encoding="utf-8")
    research_loop._prevalidate_program_core(experiment)


def test_preflight_fails_on_core_drift(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path, program="focus.md")
    base_text = ar_types.PROGRAM_PATH.read_text(encoding="utf-8")
    key, section_text = next(iter(ar_types.extract_core_sections(base_text).items()))
    drifted = base_text.replace(section_text, section_text + "\n\nDRIFT LINE\n", 1)
    assert drifted != base_text
    (_agentic_dir(experiment) / "focus.md").write_text(drifted, encoding="utf-8")

    with pytest.raises(ar_types.AgenticResearchValidationError) as excinfo:
        research_loop._prevalidate_program_core(experiment)
    assert f"agentic_research_program_core_drift:focus.md:section:{key}" in str(excinfo.value)
