"""Program CORE drift check and mechanical re-splice.

The runner loads exactly one self-contained program file per experiment, so every custom program
carries a byte-verbatim copy of ``programs/PROGRAM.md``'s CORE sections. Editing a CORE section in
PROGRAM.md therefore invalidates every custom program until it is re-spliced: the session-start
pre-flight (``loop._prevalidate_program_core``) refuses to run a drifted program, which also bites
a live run on re-entry (bail re-invoke, restart). This module is the mechanical fix — report the
first diverging section, or rewrite the program's CORE sections from the base while leaving its
strategy sections (§0, §4, §6) and preamble untouched.

USAGE:
    uv run numereng research program check --experiment-id <id>
    uv run numereng research program resplice --experiment-id <id>

    from numereng.agentic_research.engine import program
    result = program.resplice_program_core(store_root=root, experiment_id="2026-08-31_nn-e60")
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from numereng.agentic_research.engine import memory
from numereng.agentic_research.engine import types as ar_types
from numereng.features.experiments import ExperimentRecord, get_experiment

# --------------------------------------------------------------------------- #
# Result type
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProgramCoreResult:
    experiment_id: str
    program_path: Path
    base_program_path: Path
    is_base_program: bool
    diverging_section: str | None
    written: bool = False
    backup_path: Path | None = None

    @property
    def in_sync(self) -> bool:
        return self.diverging_section is None


# --------------------------------------------------------------------------- #
# Drift check
# --------------------------------------------------------------------------- #


def first_core_drift(experiment: ExperimentRecord) -> str | None:
    """First CORE section of the experiment's program that differs from PROGRAM.md, else ``None``.

    The base program is exempt (it is the canonical CORE).
    """
    program = memory.program_path(experiment)
    if program == ar_types.PROGRAM_PATH:
        return None
    return ar_types.first_diverging_core_section(
        program.read_text(encoding="utf-8"), ar_types.PROGRAM_PATH.read_text(encoding="utf-8")
    )


def check_program_core(*, store_root: Path, experiment_id: str) -> ProgramCoreResult:
    """Resolve the experiment's program and report whether its CORE matches PROGRAM.md."""
    experiment = get_experiment(store_root=store_root, experiment_id=experiment_id)
    program = memory.program_path(experiment)
    return ProgramCoreResult(
        experiment_id=experiment.experiment_id,
        program_path=program,
        base_program_path=ar_types.PROGRAM_PATH,
        is_base_program=program == ar_types.PROGRAM_PATH,
        diverging_section=first_core_drift(experiment),
    )


# --------------------------------------------------------------------------- #
# Re-splice
# --------------------------------------------------------------------------- #


def resplice_program_core(*, store_root: Path, experiment_id: str) -> ProgramCoreResult:
    """Rewrite the program's CORE sections from PROGRAM.md, keeping a ``.bak`` copy of the original.

    No-op (nothing written) when the program is the base itself or already in sync. The spliced text
    is verified against the base before anything touches disk, so a structural problem surfaces as
    an error rather than a half-written program.
    """
    checked = check_program_core(store_root=store_root, experiment_id=experiment_id)
    if checked.is_base_program or checked.in_sync:
        return checked
    program_text = checked.program_path.read_text(encoding="utf-8")
    base_text = checked.base_program_path.read_text(encoding="utf-8")
    spliced = ar_types.splice_core_sections(program_text, base_text)
    residual = ar_types.first_diverging_core_section(spliced, base_text)
    if residual is not None:
        raise ar_types.AgenticResearchValidationError(
            f"agentic_research_program_resplice_failed:{checked.program_path.name}:section:{residual}"
        )
    backup = _backup_path(checked.program_path)
    backup.write_text(program_text, encoding="utf-8")
    ar_types.write_text(checked.program_path, spliced)
    return ProgramCoreResult(
        experiment_id=checked.experiment_id,
        program_path=checked.program_path,
        base_program_path=checked.base_program_path,
        is_base_program=False,
        diverging_section=None,
        written=True,
        backup_path=backup,
    )


def _backup_path(program: Path) -> Path:
    # `.bak`, not `.md`: a Markdown sibling would be picked up as an active program by the drift lint
    # when the program lives in the legacy `programs/` directory.
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return program.with_name(f"{program.stem}.pre-resplice-{stamp}.bak")
