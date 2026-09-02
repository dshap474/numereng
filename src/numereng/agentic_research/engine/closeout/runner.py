"""Closeout runner: gate, evidence, one finalize memo.

Closeout turns a completed agentic experiment into the two artifacts a human reads: the
deterministic evidence bundle (including the one-time sealed-holdout opening) and one LLM-written
decision memo. Nothing else happens here — research-memory writes belong to the
``research-memory-update`` skill. Re-running overwrites both artifacts. Every failure raises
``CloseoutError`` (-> ``PackageError`` -> CLI exit 1), with the raw LLM response left on disk beside
the evidence for inspection.

USAGE:
    from numereng.agentic_research.engine.closeout import runner
    result = runner.run_closeout(store_root=".numereng", experiment_id="x")
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable
from pathlib import Path

from numereng.agentic_research.engine import llm, memory
from numereng.agentic_research.engine import types as ar_types
from numereng.agentic_research.engine.closeout import evidence as evidence_mod
from numereng.agentic_research.engine.closeout import types as ct
from numereng.features.experiments import ExperimentRecord, get_experiment
from numereng.features.store import resolve_store_root

FINALIZE_PROMPT_PATH = Path(__file__).parents[2] / "prompts" / "closeout-finalize.md"
FINALIZE_LABEL = "finalize"
_WORKING_SET_CAP = 40_000
_MEMO_CAP = ar_types.MAX_CONTEXT_CHARS  # 12_000, mirrors the in-run per-memo cap


# --------------------------------------------------------------------------- #
# Gate
# --------------------------------------------------------------------------- #
def _journal_nonempty(experiment: ExperimentRecord) -> bool:
    path = memory.journal_path(experiment)
    if not path.is_file():
        return False
    with contextlib.suppress(OSError):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                return True
    return False


def _gate(experiment: ExperimentRecord, state: dict[str, object] | None, *, allow_incomplete: bool) -> None:
    """Refuse what closeout cannot distill. ``allow_incomplete`` waives the live-run and budget gates."""
    if experiment.status == "archived":
        raise ct.CloseoutError(ct.ERR_EXPERIMENT_ARCHIVED)
    if state is None or not memory.state_path(experiment).is_file():
        raise ct.CloseoutError(ct.ERR_NOT_AGENTIC)
    if not _journal_nonempty(experiment):
        raise ct.CloseoutError(ct.ERR_NO_ROUNDS)
    if allow_incomplete:
        return
    if state.get("status") == "running":
        raise ct.CloseoutError(ct.ERR_RUN_ACTIVE)
    budget = experiment.metadata.get(ar_types.BUDGET_ROUNDS_METADATA_KEY)
    if isinstance(budget, int) and not isinstance(budget, bool):
        done = ar_types.as_int(state.get("total_rounds_completed"), default=0)
        if done < budget:
            raise ct.CloseoutError(f"{ct.ERROR_PREFIX}budget_not_reached:{done}/{budget}")


# --------------------------------------------------------------------------- #
# Bounded finalize context
# --------------------------------------------------------------------------- #
def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


def _fill_bounded[T](running: int, items: list[T], cost_of: Callable[[T], int]) -> tuple[list[T], int]:
    """Greedily keep items in order while the running size stays within the context cap.

    Returns ``(kept, dropped)``. The 890 KB prompt that killed a 500-round run is the standing
    lesson: no term of a closeout context may grow with round count.
    """
    kept: list[T] = []
    dropped = 0
    for index, item in enumerate(items):
        cost = cost_of(item)
        if running + cost <= ct.MAX_CLOSEOUT_CONTEXT_CHARS:
            kept.append(item)
            running += cost
        else:
            dropped = len(items) - index
            break
    return kept, dropped


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
        "last_error",
    )
    return {key: state.get(key) for key in keys if key in state}


def _round_memos_newest_first(experiment: ExperimentRecord) -> list[dict[str, str]]:
    directory = memory.rounds_dir(experiment)
    if not directory.is_dir():
        return []
    candidates = [path for path in directory.glob("r*.md") if memory.parse_round_label(path.stem) is not None]
    candidates.sort(key=lambda path: memory.parse_round_label(path.stem) or 0, reverse=True)
    memos: list[dict[str, str]] = []
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        memos.append({"round_label": path.stem, "memo": _truncate(text, _MEMO_CAP)})
    return memos


def build_finalize_context(
    *, experiment: ExperimentRecord, state: dict[str, object], evidence: dict[str, object]
) -> dict[str, object]:
    """Finalize context: evidence (never truncated) > state > working set > round memos."""
    working_set = ar_types.read_text(memory.experiment_markdown_path(experiment), limit=_WORKING_SET_CAP)
    base: dict[str, object] = {
        "phase": FINALIZE_LABEL,
        "experiment_id": experiment.experiment_id,
        "evidence_summary": evidence,
        "state": _state_summary(state),
        "experiment_working_set": working_set,
        "round_memos": [],
    }
    kept, dropped = _fill_bounded(
        len(json.dumps(base, default=str)),
        _round_memos_newest_first(experiment),
        lambda memo: len(json.dumps(memo, default=str)) + 2,
    )
    base["round_memos"] = kept
    if dropped:
        base["round_memos_truncation"] = f"...[truncated: {dropped} items dropped]"
    return base


# --------------------------------------------------------------------------- #
# Finalize memo
# --------------------------------------------------------------------------- #
def _write_memo(context: dict[str, object], directory: Path) -> Path:
    """Ask the LLM for the memo, persist the raw response, validate it, then write the memo."""
    raw, _source = llm.call_research_llm(
        prompt=llm.render_context_prompt(context, prompt_path=FINALIZE_PROMPT_PATH),
        artifact_dir=ct.debug_dir(directory),
        round_label=FINALIZE_LABEL,
        schema=None,
        timeout_seconds=ct.CLOSEOUT_TIMEOUT_SECONDS,
    )
    ar_types.write_text(directory / ct.CLOSEOUT_RESPONSE_FILENAME, raw)
    memo = raw.strip()
    if ct.MEMO_REQUIRED_HEADING not in memo:
        raise ct.CloseoutError(f"{ct.ERROR_PREFIX}memo_section_missing:{ct.MEMO_REQUIRED_HEADING}")
    if len(memo) < ct.MEMO_MIN_CHARS:
        raise ct.CloseoutError(f"{ct.ERROR_PREFIX}memo_too_short:{len(memo)}/{ct.MEMO_MIN_CHARS}")
    path = directory / ct.CLOSEOUT_MEMO_FILENAME
    ar_types.write_text(path, memo if memo.endswith("\n") else memo + "\n")
    return path


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def run_closeout(
    *,
    store_root: str | Path = ".numereng",
    experiment_id: str,
    allow_incomplete: bool = False,
) -> ct.CloseoutResult:
    """Gate, build the evidence bundle, write one decision memo. Re-running overwrites both."""
    root = resolve_store_root(store_root)
    experiment = get_experiment(store_root=root, experiment_id=experiment_id)

    run_state = memory.load_state(memory.state_path(experiment))
    _gate(experiment, run_state, allow_incomplete=allow_incomplete)
    run_state = run_state or {}

    directory = ct.closeout_dir(memory.agentic_dir(experiment))
    directory.mkdir(parents=True, exist_ok=True)
    evidence = evidence_mod.build_evidence(experiment=experiment, state=run_state, store_root=root)
    evidence_path = directory / ct.CLOSEOUT_EVIDENCE_FILENAME
    ar_types.write_json(evidence_path, evidence)

    context = build_finalize_context(experiment=experiment, state=run_state, evidence=evidence)
    memo_path = _write_memo(context, directory)

    holdout = evidence.get("holdout")
    return ct.CloseoutResult(
        experiment_id=experiment.experiment_id,
        evidence_path=evidence_path,
        memo_path=memo_path,
        holdout_summary=holdout if isinstance(holdout, dict) else None,
    )
