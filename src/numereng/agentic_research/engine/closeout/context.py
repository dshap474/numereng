"""Bounded per-phase context assembly for the closeout chain.

The 890 KB prompt that killed a 500-round run (see the feature AGENTS.md) is the standing
lesson: nothing in a closeout context may grow unbounded. Every context is capped at
``MAX_CLOSEOUT_CONTEXT_CHARS``; the deterministic evidence summary is placed first and never
truncated, and only bulk text (round memos, oldest-first) is dropped under pressure, always
leaving an explicit ``...[truncated: N items dropped]`` marker.

USAGE:
    from numereng.agentic_research.engine.closeout import context
    ctx = context.build_finalize_context(
        experiment=rec, state=state_dict, evidence=summary, program_text=program_md)
"""

from __future__ import annotations

import json
from pathlib import Path

from numereng.agentic_research.engine import memory
from numereng.agentic_research.engine import types as ar_types
from numereng.agentic_research.engine.closeout import merge
from numereng.agentic_research.engine.closeout import types as ct
from numereng.features.experiments import ExperimentRecord

_PROGRAM_CAP = 40_000
_WORKING_SET_CAP = 40_000
_MEMO_CAP = ar_types.MAX_CONTEXT_CHARS  # 12_000, mirrors the in-run per-memo cap
_DECISION_MEMO_CAP = 60_000
_BRANCH_FILE_CAP = 20_000


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


def _state_summary(state: dict[str, object]) -> dict[str, object]:
    keys = (
        "status",
        "stop_reason",
        "next_round_number",
        "total_rounds_completed",
        "failed_rounds_counter",
        "champion",
        "believed_best",
        "believed_best_changed_round",
        "best_overall",
        "last_error",
    )
    return {key: state.get(key) for key in keys if key in state}


def _round_memos_newest_first(experiment: ExperimentRecord) -> list[dict[str, str]]:
    directory = memory.rounds_dir(experiment)
    if not directory.is_dir():
        return []
    candidates = [path for path in directory.glob("r*.md") if path.stem[1:].isdigit()]
    candidates.sort(key=lambda path: int(path.stem[1:]), reverse=True)
    memos: list[dict[str, str]] = []
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        memos.append({"round_label": path.stem, "memo": _truncate(text, _MEMO_CAP)})
    return memos


def build_finalize_context(
    *,
    experiment: ExperimentRecord,
    state: dict[str, object],
    evidence: dict[str, object],
    program_text: str,
) -> dict[str, object]:
    """FINALIZE context: evidence (never truncated) > state > program > working set > round memos."""
    working_set = ar_types.read_text(memory.experiment_markdown_path(experiment), limit=_WORKING_SET_CAP)
    base: dict[str, object] = {
        "phase": ct.PHASE_FINALIZE,
        "experiment_id": experiment.experiment_id,
        "evidence_summary": evidence,
        "state": _state_summary(state),
        "program": _truncate(program_text, _PROGRAM_CAP),
        "experiment_working_set": working_set,
        "round_memos": [],
    }
    running = len(json.dumps(base, default=str))
    kept: list[dict[str, str]] = []
    dropped = 0
    memos = _round_memos_newest_first(experiment)
    for index, memo in enumerate(memos):
        cost = len(json.dumps(memo, default=str)) + 2
        if running + cost <= ct.MAX_CLOSEOUT_CONTEXT_CHARS:
            kept.append(memo)
            running += cost
        else:
            dropped = len(memos) - index
            break
    base["round_memos"] = kept
    if dropped:
        base["round_memos_truncation"] = f"...[truncated: {dropped} items dropped]"
    return base


def build_extract_context(
    *,
    experiment: ExperimentRecord,
    memo_text: str,
    evidence: dict[str, object],
) -> dict[str, object]:
    """EXTRACT context: decision memo > evidence summary > compact per-round table (never raw rounds)."""
    rounds_table = evidence.get("rounds_table")
    rounds_table = rounds_table if isinstance(rounds_table, list) else []
    evidence_head = {key: value for key, value in evidence.items() if key != "rounds_table"}
    base: dict[str, object] = {
        "phase": ct.PHASE_EXTRACT,
        "experiment_id": experiment.experiment_id,
        "decision_memo": _truncate(memo_text, _DECISION_MEMO_CAP),
        "evidence_summary": evidence_head,
        "rounds_table": [],
    }
    running = len(json.dumps(base, default=str))
    kept: list[object] = []
    dropped = 0
    for index, row in enumerate(rounds_table):
        cost = len(json.dumps(row, default=str)) + 2
        if running + cost <= ct.MAX_CLOSEOUT_CONTEXT_CHARS:
            kept.append(row)
            running += cost
        else:
            dropped = len(rounds_table) - index
            break
    base["rounds_table"] = kept
    if dropped:
        base["rounds_table_truncation"] = f"...[truncated: {dropped} rows dropped]"
    return base


def build_classify_context(*, memo_text: str, evidence: dict[str, object]) -> dict[str, object]:
    """CLASSIFY context: bounded memo and deterministic evidence only."""
    return {
        "phase": ct.PHASE_CLASSIFY,
        "decision_memo": _truncate(memo_text, _DECISION_MEMO_CAP),
        "evidence_summary": evidence,
        "allowed_topics": list(ct.MEMORY_TOPIC_FILES),
    }


def _ledger_view(text: str | None, *, topic: str) -> str:
    """A ledger's two mutable top sections plus its newest entries, capped with a truncation marker."""
    if not text:
        return ""
    try:
        _preamble, overview, best_understanding, learnings = merge.parse_ledger(text, topic=topic)
    except ct.CloseoutError:
        return _truncate(text, ct.LEDGER_CONTEXT_CAP)
    newest = merge.newest_entries(learnings, count=ct.LEDGER_NEWEST_ENTRIES)
    view = (
        f"{overview.rstrip(chr(10))}\n\n"
        f"{best_understanding.rstrip(chr(10))}\n\n"
        f"{ct.LEDGER_LEARNINGS_ANCHOR}\n\n{newest}"
    )
    return _truncate(view, ct.LEDGER_CONTEXT_CAP)


def build_synthesize_context(
    *,
    experiment_id: str,
    memory_root: Path,
    relevant_topics: tuple[str, ...] = ct.MEMORY_TOPIC_FILES,
) -> dict[str, object]:
    """SYNTHESIZE context for selected branch topics, their ledgers, and CURRENT.md."""
    branch_dir = memory_root / "experiments" / experiment_id
    branch_files: dict[str, str] = {}
    for name in (ct.MEMORY_BRANCH_README, *(f"{topic}.md" for topic in relevant_topics)):
        branch_files[name] = ar_types.read_text(branch_dir / name, limit=_BRANCH_FILE_CAP) or ""
    topics_dir = memory_root / "topics"
    ledger_views = {
        topic: _ledger_view(
            ar_types.read_text(topics_dir / f"{topic}.md", limit=ct.MAX_CLOSEOUT_CONTEXT_CHARS), topic=topic
        )
        for topic in relevant_topics
    }
    current_md = ar_types.read_text(memory_root / ct.CURRENT_MD_FILENAME, limit=ct.MAX_CLOSEOUT_CONTEXT_CHARS) or ""

    base: dict[str, object] = {
        "phase": ct.PHASE_SYNTHESIZE,
        "experiment_id": experiment_id,
        "relevant_topics": list(relevant_topics),
        "branch_files": branch_files,
        "ledgers": {},
        "current_md": "",
    }
    running = len(json.dumps(base, default=str))
    ledgers: dict[str, str] = {}
    dropped = 0
    for index, topic in enumerate(relevant_topics):
        view = ledger_views[topic]
        cost = len(json.dumps({topic: view}, default=str))
        if running + cost <= ct.MAX_CLOSEOUT_CONTEXT_CHARS:
            ledgers[topic] = view
            running += cost
        else:
            dropped = len(relevant_topics) - index
            break
    base["ledgers"] = ledgers
    if dropped:
        base["ledgers_truncation"] = f"...[truncated: {dropped} ledgers dropped]"
    remaining = ct.MAX_CLOSEOUT_CONTEXT_CHARS - running
    base["current_md"] = current_md if len(current_md) <= remaining else _truncate(current_md, max(remaining, 0))
    return base
