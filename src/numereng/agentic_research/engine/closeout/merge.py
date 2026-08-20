"""Deterministic master-ledger merge for the SYNTHESIZE phase.

SYNTHESIZE never returns a whole ledger (audit F6/F7): the LLM returns a per-topic delta
(a new ``### <experiment_id>`` entry block plus optional per-section replacements), and this
module splices it into the on-disk ledger deterministically so existing entry bytes are preserved
by construction. A ledger has FOUR regions in order:

    <preamble>                            # title + metadata, before "## Current Overview"
    ## Current Overview ...               # mutable top section
    ## Current Best Understanding ...      # mutable top section
    ## Append-Only Experiment Learnings ... (one "### <experiment_id>" entry per experiment)

Each mutable top section's body is replaced only when its own replacement is supplied, spliced
heading-bounded so the other section stays byte-identical (audit F6: an overview replacement must
never destroy the best-understanding section). A new entry is appended at the end of the learnings
region (or, on ``--restart-from synthesize``, spliced in place of this experiment's existing
block). Everything else is copied verbatim.

USAGE:
    from numereng.agentic_research.engine.closeout import merge
    merged = merge.merge_ledger(
        existing_text, new_entry=entry_md, overview_replacement=None,
        best_understanding_replacement=None, experiment_id="2026-07-05_agentic-cyrus-scale-v1",
        replace_existing=False, topic="hyperparameters")
"""

from __future__ import annotations

from numereng.agentic_research.engine.closeout import types as ct


# --------------------------------------------------------------------------- #
# Ledger parsing
# --------------------------------------------------------------------------- #
def _anchor_index(text: str, anchor: str) -> int:
    """Return the offset of ``anchor`` when it starts a line, or -1 if absent."""
    if text.startswith(anchor + "\n") or text == anchor:
        return 0
    needle = "\n" + anchor
    pos = text.find(needle)
    return pos + 1 if pos != -1 else -1


def parse_ledger(text: str, *, topic: str) -> tuple[str, str, str, str]:
    """Split a ledger into (preamble, overview, best_understanding, learnings) sections.

    Every ledger has three sections in strict order: ``## Current Overview``,
    ``## Current Best Understanding``, ``## Append-Only Experiment Learnings``. Raise on any
    missing anchor or out-of-order anchors so a malformed ledger never gets silently spliced.
    """
    ov = _anchor_index(text, ct.LEDGER_OVERVIEW_ANCHOR)
    bu = _anchor_index(text, ct.LEDGER_BEST_UNDERSTANDING_ANCHOR)
    learn = _anchor_index(text, ct.LEDGER_LEARNINGS_ANCHOR)
    if ov == -1 or bu == -1 or learn == -1 or not (ov < bu < learn):
        raise ct.CloseoutError(ct.err_ledger_structure(topic))
    return text[:ov], text[ov:bu], text[bu:learn], text[learn:]


# --------------------------------------------------------------------------- #
# Entry-block helpers
# --------------------------------------------------------------------------- #
def entry_heading(experiment_id: str) -> str:
    return f"{ct.LEDGER_ENTRY_PREFIX}{experiment_id}"


def count_entries(learnings_region: str, experiment_id: str) -> int:
    """Count ``### <experiment_id>`` headings in a learnings region."""
    heading = entry_heading(experiment_id)
    count = 0
    for line in learnings_region.splitlines():
        if line.rstrip() == heading:
            count += 1
    return count


def _entry_span(learnings_region: str, experiment_id: str) -> tuple[int, int] | None:
    """Return (start, end) char offsets of this experiment's ``### <id>`` block, or None."""
    heading = entry_heading(experiment_id)
    lines = learnings_region.splitlines(keepends=True)
    start_off: int | None = None
    offset = 0
    end_off = len(learnings_region)
    for line in lines:
        stripped = line.rstrip()
        if start_off is None and stripped == heading:
            start_off = offset
        elif start_off is not None and stripped.startswith(ct.LEDGER_ENTRY_PREFIX):
            end_off = offset
            break
        offset += len(line)
    if start_off is None:
        return None
    return start_off, end_off


# --------------------------------------------------------------------------- #
# Merge
# --------------------------------------------------------------------------- #
def _section_has_heading(body: str) -> bool:
    """True if a section-body replacement carries a ``## `` heading (it must not — it is a body)."""
    return any(line.lstrip().startswith("## ") for line in body.splitlines())


def _splice_section(anchor: str, replacement: str) -> str:
    """Rebuild a mutable top section from its anchor and a replacement body."""
    body = replacement.strip("\n")
    return f"{anchor}\n\n{body}\n\n"


def merge_ledger(
    existing_text: str,
    *,
    new_entry: str | None,
    overview_replacement: str | None,
    best_understanding_replacement: str | None,
    experiment_id: str,
    replace_existing: bool,
    topic: str,
) -> str:
    """Splice a delta into a ledger deterministically; existing entry bytes are preserved.

    Each mutable top section (``## Current Overview``, ``## Current Best Understanding``) is
    replaced only when its own replacement is supplied, heading-bounded so the other section stays
    byte-identical (audit F6). A replacement body must be a section body, not a region: it may not
    contain a ``## `` heading. ``replace_existing`` (``--restart-from synthesize``) replaces this
    experiment's own learnings block in place; otherwise a pre-existing block is a hard conflict.
    """
    preamble, overview_region, best_understanding_region, learnings_region = parse_ledger(existing_text, topic=topic)

    if overview_replacement is not None:
        if _section_has_heading(overview_replacement):
            raise ct.CloseoutError(ct.err_section_replacement_invalid(topic, "overview"))
        overview_region = _splice_section(ct.LEDGER_OVERVIEW_ANCHOR, overview_replacement)

    if best_understanding_replacement is not None:
        if _section_has_heading(best_understanding_replacement):
            raise ct.CloseoutError(ct.err_section_replacement_invalid(topic, "best_understanding"))
        best_understanding_region = _splice_section(ct.LEDGER_BEST_UNDERSTANDING_ANCHOR, best_understanding_replacement)

    if new_entry is not None:
        entry_text = new_entry.strip("\n") + "\n"
        span = _entry_span(learnings_region, experiment_id)
        if span is not None and not replace_existing:
            raise ct.CloseoutError(ct.err_duplicate_ledger_entry(topic))
        if span is not None:
            start, end = span
            head = learnings_region[:start].rstrip("\n")
            tail = learnings_region[end:]
            joined = f"{head}\n\n{entry_text}"
            if tail.strip("\n"):
                joined = f"{joined}\n{tail.lstrip(chr(10))}"
            learnings_region = joined
        else:
            base = learnings_region.rstrip("\n")
            learnings_region = f"{base}\n\n{entry_text}"

    return preamble + overview_region + best_understanding_region + learnings_region


def extract_entry_block(ledger_text: str, experiment_id: str, *, topic: str) -> str:
    """Return this experiment's ``### <id>`` block from a ledger (for upstream fingerprinting).

    Trailing newlines are stripped so the block is position-independent: the same entry hashes
    identically whether it is currently the last block (bounded by EOF, one trailing newline) or
    a later entry has been appended after it (bounded by the next ``### `` heading, so a blank
    separator line would otherwise be included). Without this, a cross-experiment append between
    synthesize would trip ``_stale_upstream`` (audit F5)."""
    _preamble, _overview, _best_understanding, learnings_region = parse_ledger(ledger_text, topic=topic)
    span = _entry_span(learnings_region, experiment_id)
    if span is None:
        return ""
    start, end = span
    return learnings_region[start:end].strip("\n")


def newest_entries(learnings_region: str, *, count: int) -> str:
    """Return the concatenated newest ``count`` entry blocks (for bounded synthesize context)."""
    lines = learnings_region.splitlines(keepends=True)
    starts: list[int] = []
    offset = 0
    for line in lines:
        if line.rstrip().startswith(ct.LEDGER_ENTRY_PREFIX):
            starts.append(offset)
        offset += len(line)
    if not starts:
        return ""
    chosen = starts[-count:]
    return learnings_region[chosen[0] :]
