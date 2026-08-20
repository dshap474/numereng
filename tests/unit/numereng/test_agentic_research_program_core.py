"""Drift lint for the invariant CORE of agentic-research program files.

~75% of every program file is invariant boilerplate (frozen evaluator, evidence doctrine, output
contract, context glossary, ...) that each program must copy verbatim because the runner loads
exactly one self-contained file. That copy step is how a stale line (e.g. an old scoring stage)
drifts in. This test pins PROGRAM.md as the canonical CORE and asserts every active program's CORE
sections match it byte-for-byte. Strategy sections (the live-viability frame, substrate/budget, and
search discipline) are experiment-specific and exempt.

The CORE-section parser lives in ``types.extract_core_sections`` and is reused by the runtime
pre-flight check in ``loop._prevalidate_program_core`` — this test imports it rather than duplicating.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from numereng.agentic_research.engine import types as ar_types

_CUSTOM_DIR = ar_types.CUSTOM_PROGRAM_DIR
CORE_KEYS = ar_types.CORE_PROGRAM_SECTION_KEYS


def _active_programs() -> list[Path]:
    programs = [ar_types.PROGRAM_PATH]
    programs.extend(
        path for path in sorted(_CUSTOM_DIR.glob("*.md")) if path.name != "README.md" and path != ar_types.PROGRAM_PATH
    )
    return programs


_CANONICAL_CORE = ar_types.extract_core_sections(ar_types.PROGRAM_PATH.read_text(encoding="utf-8"))


def test_canonical_program_defines_every_core_section() -> None:
    # If PROGRAM.md loses or renames a CORE section, the pin itself is broken — catch it here.
    missing = [key for key in CORE_KEYS if key not in _CANONICAL_CORE]
    assert not missing, f"PROGRAM.md missing CORE sections: {missing}"
    assert all(_CANONICAL_CORE[key] for key in CORE_KEYS)


@pytest.mark.parametrize("program_path", _active_programs(), ids=lambda path: path.name)
def test_active_program_core_matches_canonical(program_path: Path) -> None:
    program_core = ar_types.extract_core_sections(program_path.read_text(encoding="utf-8"))
    for key in CORE_KEYS:
        assert key in program_core, f"{program_path.name} is missing CORE section {key}"
        assert program_core[key] == _CANONICAL_CORE[key], (
            f"{program_path.name} CORE section {key} drifted from PROGRAM.md — copy it verbatim"
        )
