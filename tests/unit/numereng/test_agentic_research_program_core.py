"""Drift lint for the invariant CORE of agentic-research program files.

~75% of every program file is invariant boilerplate (frozen evaluator, evidence doctrine, output
contract, context glossary, ...) that each program must copy verbatim because the runner loads
exactly one self-contained file. That copy step is how a stale line (e.g. an old scoring stage)
drifts in. This test pins PROGRAM.md as the canonical CORE and asserts every active program's CORE
sections match it byte-for-byte. Strategy sections (the live-viability frame, substrate/budget, and
search discipline) are experiment-specific and exempt.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from numereng.features.agentic_research import types as ar_types

_PROGRAM_DIR = ar_types.PROGRAM_PATH.parent
_CUSTOM_DIR = ar_types.CUSTOM_PROGRAM_DIR

# Section keys that are pure harness mechanism — identical across every experiment. Excludes the
# strategy sections "0." (frame), "4." (substrate/budget), and "6." (search discipline).
CORE_KEYS = ("1.", "2.", "2.1", "3.", "5.", "7.", "8.", "9.", "10.", "Context")

# A section boundary is a level-2 heading, or a numbered level-3 heading (e.g. "### 2.1 ...").
# Un-numbered level-3 headings (e.g. "### `round_markdown`") stay inside their parent section body.
_BOUNDARY = re.compile(r"^(#{2} |#{3} \d)")


def _section_key(heading: str) -> str:
    return heading.lstrip("#").strip().split()[0]


def _extract_sections(text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current_key: str | None = None
    buffer: list[str] = []
    for line in text.splitlines():
        if _BOUNDARY.match(line):
            if current_key is not None:
                sections[current_key] = "\n".join(buffer).strip()
            current_key = _section_key(line)
            buffer = [line]
        elif current_key is not None:
            buffer.append(line)
    if current_key is not None:
        sections[current_key] = "\n".join(buffer).strip()
    return sections


def _core(text: str) -> dict[str, str]:
    sections = _extract_sections(text)
    return {key: sections[key] for key in CORE_KEYS if key in sections}


def _active_programs() -> list[Path]:
    programs = [ar_types.PROGRAM_PATH]
    programs.extend(path for path in sorted(_CUSTOM_DIR.glob("*.md")) if path.name != "README.md")
    return programs


_CANONICAL_CORE = _core(ar_types.PROGRAM_PATH.read_text(encoding="utf-8"))


def test_canonical_program_defines_every_core_section() -> None:
    # If PROGRAM.md loses or renames a CORE section, the pin itself is broken — catch it here.
    missing = [key for key in CORE_KEYS if key not in _CANONICAL_CORE]
    assert not missing, f"PROGRAM.md missing CORE sections: {missing}"
    assert all(_CANONICAL_CORE[key] for key in CORE_KEYS)


@pytest.mark.parametrize("program_path", _active_programs(), ids=lambda path: path.name)
def test_active_program_core_matches_canonical(program_path: Path) -> None:
    program_core = _core(program_path.read_text(encoding="utf-8"))
    for key in CORE_KEYS:
        assert key in program_core, f"{program_path.name} is missing CORE section {key}"
        assert program_core[key] == _CANONICAL_CORE[key], (
            f"{program_path.name} CORE section {key} drifted from PROGRAM.md — copy it verbatim"
        )
