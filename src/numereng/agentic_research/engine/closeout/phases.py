"""Phase output schemas, envelope parsing, and content validators.

Each closeout phase emits exactly one JSON object as a file-oriented envelope. This module
holds the response schemas, strict parsers, and validators that must pass before anything is
staged (no partial writes). The schemas reach codex as ``--output-schema``; under the droid-exec
transport the schema is appended to the prompt instead, so it is enforced only by the parsers and
validators below.

USAGE:
    from numereng.agentic_research.engine.closeout import phases
    files, notes = phases.parse_files_envelope(raw_response)
    slots = phases.validate_finalize(files, experiment_id="x", believed_best_config="config_040.json")
"""

from __future__ import annotations

from typing import cast

from numereng.agentic_research.engine import llm
from numereng.agentic_research.engine.closeout import types as ct

# --------------------------------------------------------------------------- #
# Shared output schema (codex --output-schema; prompt-appended under droid-exec)
# --------------------------------------------------------------------------- #
FILES_ENVELOPE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "files": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
        "notes": {"type": "string"},
    },
    "required": ["files", "notes"],
    "additionalProperties": False,
}

# EXTRACT reuses the files envelope; SYNTHESIZE returns per-topic deltas plus a full CURRENT.md.
SYNTHESIZE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "deltas": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "new_entry_markdown": {"type": "string"},
                    "overview_replacement_markdown": {"type": ["string", "null"]},
                    "best_understanding_replacement_markdown": {"type": ["string", "null"]},
                },
                "required": [
                    "topic",
                    "new_entry_markdown",
                    "overview_replacement_markdown",
                    "best_understanding_replacement_markdown",
                ],
                "additionalProperties": False,
            },
        },
        "current_md": {"type": "string"},
        "notes": {"type": "string"},
    },
    "required": ["deltas", "current_md", "notes"],
    "additionalProperties": False,
}

# --------------------------------------------------------------------------- #
# FINALIZE contract (§3.1)
# --------------------------------------------------------------------------- #
FINALIZE_ALLOWED_SLOTS = (ct.CLOSEOUT_MEMO_FILENAME,)
FINALIZE_REQUIRED_HEADINGS = (
    "Verdict",
    "Evidence Status And Caveats",
    "Candidate Hierarchy",
    "Metric Conflicts",
    "Sweep Discipline Audit",
    "Design-Space Roles",
    "Implications For Future Work",
    "Master-Ledger Update",
)
FINALIZE_MIN_CHARS = 3_000


# --------------------------------------------------------------------------- #
# CLASSIFY contract
# --------------------------------------------------------------------------- #
CLASSIFY_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "disposition": {"type": "string", "enum": list(ct.CLASSIFICATION_DISPOSITIONS)},
        "relevant_topics": {
            "type": "array",
            "items": {"type": "string", "enum": list(ct.MEMORY_TOPIC_FILES)},
        },
        "rationale": {"type": "string"},
    },
    "required": ["disposition", "relevant_topics", "rationale"],
    "additionalProperties": False,
}


def parse_classification(raw_response: str) -> dict[str, object]:
    """Parse the persisted closeout routing decision."""
    payload = llm._extract_json_object(raw_response)
    if set(payload) != {"disposition", "relevant_topics", "rationale"}:
        raise ct.CloseoutError(ct.err_classification_field_invalid("fields"))
    disposition = payload.get("disposition")
    if disposition not in ct.CLASSIFICATION_DISPOSITIONS:
        raise ct.CloseoutError(ct.err_classification_field_invalid("disposition"))
    topics = payload.get("relevant_topics")
    if (
        not isinstance(topics, list)
        or any(not isinstance(topic, str) or topic not in ct.MEMORY_TOPIC_FILES for topic in topics)
        or len(topics) != len(set(topics))
    ):
        raise ct.CloseoutError(ct.err_classification_field_invalid("relevant_topics"))
    if disposition != "master" and topics:
        raise ct.CloseoutError(ct.err_classification_field_invalid("relevant_topics"))
    rationale = payload.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ct.CloseoutError(ct.err_classification_field_invalid("rationale"))
    return payload


# --------------------------------------------------------------------------- #
# Envelope parsing
# --------------------------------------------------------------------------- #
def _parse_files_list(files_raw: object) -> list[dict[str, str]]:
    """Validate a ``files`` array into ``[{path, content}]``; raise on any structural problem."""
    if not isinstance(files_raw, list) or not files_raw:
        raise ct.CloseoutError(f"{ct.ERROR_PREFIX}files_missing")
    files: list[dict[str, str]] = []
    for item in files_raw:
        if not isinstance(item, dict):
            raise ct.CloseoutError(f"{ct.ERROR_PREFIX}file_entry_invalid")
        entry = cast("dict[str, object]", item)
        path, content = entry.get("path"), entry.get("content")
        if not isinstance(path, str) or not path:
            raise ct.CloseoutError(f"{ct.ERROR_PREFIX}file_entry_invalid")
        if not isinstance(content, str):
            raise ct.CloseoutError(f"{ct.ERROR_PREFIX}file_entry_invalid")
        files.append({"path": path, "content": content})
    return files


def parse_files_envelope(raw_response: str) -> tuple[list[dict[str, str]], str]:
    """Parse the shared ``{files, notes}`` envelope; raise ``CloseoutError`` on structural problems."""
    payload = llm._extract_json_object(raw_response)
    files = _parse_files_list(payload.get("files"))
    notes = payload.get("notes")
    return files, notes if isinstance(notes, str) else ""


def _collect_slots(files: list[dict[str, str]], *, allowed: tuple[str, ...]) -> dict[str, str]:
    """Map path -> content, enforcing the allowlist, no duplicates, no empty content."""
    slots: dict[str, str] = {}
    allowed_set = set(allowed)
    for item in files:
        path, content = item["path"], item["content"]
        if path not in allowed_set:
            raise ct.CloseoutError(ct.err_output_path_not_allowed(path))
        if path in slots:
            raise ct.CloseoutError(ct.err_output_path_duplicate(path))
        if not content.strip():
            raise ct.CloseoutError(ct.err_output_content_empty(path))
        slots[path] = content
    for path in allowed:
        if path not in slots:
            raise ct.CloseoutError(ct.err_output_slot_missing(path))
    return slots


def _has_heading(text: str, heading: str) -> bool:
    needle = heading.lower()
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#") and stripped.lstrip("#").strip().lower().startswith(needle):
            return True
    return False


def validate_finalize(files: list[dict[str, str]], *, experiment_id: str, believed_best_config: str) -> dict[str, str]:
    """Validate the FINALIZE memo; return {relative_path: content} ready for staging."""
    slots = _collect_slots(files, allowed=FINALIZE_ALLOWED_SLOTS)
    memo = slots[ct.CLOSEOUT_MEMO_FILENAME]
    for heading in FINALIZE_REQUIRED_HEADINGS:
        if not _has_heading(memo, heading):
            raise ct.CloseoutError(ct.err_memo_section_missing(heading))
    if experiment_id not in memo:
        raise ct.CloseoutError(ct.err_memo_reference_missing("experiment_id"))
    if believed_best_config not in memo:
        raise ct.CloseoutError(ct.err_memo_reference_missing("believed_best"))
    if len(memo) < FINALIZE_MIN_CHARS:
        raise ct.CloseoutError(ct.err_memo_too_short(len(memo), FINALIZE_MIN_CHARS))
    return slots


# --------------------------------------------------------------------------- #
# EXTRACT contract (§3.2)
# --------------------------------------------------------------------------- #
EXTRACT_TOPIC_SLOTS = tuple(f"{name}.md" for name in ct.MEMORY_TOPIC_FILES)
EXTRACT_ALLOWED_SLOTS = (ct.MEMORY_BRANCH_README, *EXTRACT_TOPIC_SLOTS)


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)


def validate_extract(files: list[dict[str, str]], *, experiment_id: str) -> dict[str, str]:
    """Validate the EXTRACT branch (README + six topic files); return {slot: content} for staging."""
    slots = _collect_slots(files, allowed=EXTRACT_ALLOWED_SLOTS)
    readme = slots[ct.MEMORY_BRANCH_README]
    if experiment_id not in readme:
        raise ct.CloseoutError(ct.err_readme_link_missing("experiment_id"))
    for topic_slot in EXTRACT_TOPIC_SLOTS:
        if topic_slot not in readme:
            raise ct.CloseoutError(ct.err_readme_link_missing(topic_slot))
    for name in ct.MEMORY_TOPIC_FILES:
        content = slots[f"{name}.md"]
        for heading in ct.EXTRACT_TOPIC_HEADINGS:
            if not _has_heading(content, heading):
                raise ct.CloseoutError(ct.err_topic_section_missing(name, heading))
        if not _contains_any(content, ct.EXTRACT_EVIDENCE_LEVELS):
            raise ct.CloseoutError(ct.err_evidence_level_invalid(name))
        if not _contains_any(content, ct.EXTRACT_DESIGN_SPACE_ROLES):
            raise ct.CloseoutError(ct.err_design_role_invalid(name))
        # Extraction is not the frontier writer: a topic file must not touch CURRENT.md.
        if ct.CURRENT_MD_FILENAME in content:
            raise ct.CloseoutError(ct.err_topic_section_missing(name, "no-current-md"))
    return slots


# --------------------------------------------------------------------------- #
# SYNTHESIZE contract (§3.3)
# --------------------------------------------------------------------------- #
def parse_synthesize_envelope(
    raw_response: str,
    *,
    relevant_topics: tuple[str, ...] = ct.MEMORY_TOPIC_FILES,
) -> tuple[dict[str, dict[str, str | None]], str, str]:
    """Parse ``{deltas, current_md, notes}``; return (deltas-by-topic, current_md, notes)."""
    payload = llm._extract_json_object(raw_response)
    deltas_raw = payload.get("deltas")
    if not isinstance(deltas_raw, list):
        raise ct.CloseoutError(f"{ct.ERROR_PREFIX}deltas_missing")
    deltas: dict[str, dict[str, str | None]] = {}
    for item in deltas_raw:
        if not isinstance(item, dict):
            raise ct.CloseoutError(f"{ct.ERROR_PREFIX}delta_entry_invalid")
        delta = cast("dict[str, object]", item)
        topic = delta.get("topic")
        entry = delta.get("new_entry_markdown")
        overview = delta.get("overview_replacement_markdown")
        best_understanding = delta.get("best_understanding_replacement_markdown")
        if not isinstance(topic, str) or topic not in ct.MEMORY_TOPIC_FILES:
            raise ct.CloseoutError(f"{ct.ERROR_PREFIX}delta_entry_invalid")
        if not isinstance(entry, str) or not entry.strip():
            raise ct.CloseoutError(ct.err_output_content_empty(topic))
        if overview is not None and not isinstance(overview, str):
            raise ct.CloseoutError(f"{ct.ERROR_PREFIX}delta_entry_invalid")
        if best_understanding is not None and not isinstance(best_understanding, str):
            raise ct.CloseoutError(f"{ct.ERROR_PREFIX}delta_entry_invalid")
        if topic in deltas:
            raise ct.CloseoutError(ct.err_duplicate_ledger_entry(topic))
        deltas[topic] = {
            "new_entry": entry,
            "overview": overview if isinstance(overview, str) else None,
            "best_understanding": best_understanding if isinstance(best_understanding, str) else None,
        }
    if set(deltas) - set(relevant_topics):
        raise ct.CloseoutError(ct.err_classification_field_invalid("synthesize_topics"))
    for name in relevant_topics:
        if name not in deltas:
            raise ct.CloseoutError(ct.err_output_slot_missing(name))
    current_md = payload.get("current_md")
    if not isinstance(current_md, str) or not current_md.strip():
        raise ct.CloseoutError(ct.err_output_content_empty(ct.CURRENT_MD_FILENAME))
    notes = payload.get("notes")
    return deltas, current_md, notes if isinstance(notes, str) else ""


def validate_entry_block(topic: str, new_entry: str, *, experiment_id: str) -> None:
    """The entry's sole markdown heading is ``### <experiment_id>`` and it links the branch file."""
    heading = f"{ct.LEDGER_ENTRY_PREFIX}{experiment_id}"
    headings = [line for line in new_entry.splitlines() if line.lstrip().startswith("#")]
    if headings != [heading] and [h.rstrip() for h in headings] != [heading]:
        raise ct.CloseoutError(ct.err_entry_heading_invalid(topic))
    if f"../experiments/{experiment_id}/{topic}.md" not in new_entry:
        raise ct.CloseoutError(ct.err_entry_link_missing(topic))


def validate_current_md(current_md: str, *, experiment_id: str) -> None:
    """CURRENT.md validator: required sections, id present, memory-branch Full-record pointer, length."""
    for heading in ct.CURRENT_MD_REQUIRED_SECTIONS:
        if not _has_heading(current_md, heading):
            raise ct.CloseoutError(ct.err_current_md_section_missing(heading))
    if experiment_id not in current_md:
        raise ct.CloseoutError(ct.err_current_md_reference_missing("experiment_id"))
    pointer = f"experiments/{experiment_id}/"
    has_full_record = any("full record" in line.lower() and pointer in line for line in current_md.splitlines())
    if not has_full_record:
        raise ct.CloseoutError(ct.err_current_md_reference_missing("full_record"))
    if len(current_md) < ct.CURRENT_MD_MIN_CHARS:
        raise ct.CloseoutError(ct.err_current_md_too_short(len(current_md), ct.CURRENT_MD_MIN_CHARS))
