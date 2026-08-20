"""Closeout runner: gate, locks, memory-root resolution, commit protocol, and the phase loop.

The runner turns a completed agentic experiment into finalized artifacts. It runs a strict gate
(status, budget, memory root, locks), builds the deterministic evidence bundle (phase 0), then runs
the phase chain up to a requested boundary. CLASSIFY routes finalized evidence to master memory, an
experiment branch only, or exclusion; the chain ends once research memory is synthesized —
designing the next experiment belongs to the pre-run INIT-PROGRAM playbook, not closeout.

Failure model: gate and request-validation problems RAISE (``CloseoutError`` -> ``PackageError`` ->
CLI exit 1). FINALIZE/CLASSIFY/EXTRACT/SYNTHESIZE failures are captured in ``CloseoutResult`` as errors.
Every write goes through the commit journal so a process death rolls forward consistently.

USAGE:
    from numereng.agentic_research.engine.closeout import runner
    result = runner.run_closeout(store_root=".numereng", experiment_id="x", until="finalize")
    status = runner.get_closeout_status(store_root=".numereng", experiment_id="x")
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import socket
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from numereng.agentic_research.engine import llm, memory
from numereng.agentic_research.engine import types as ar_types
from numereng.agentic_research.engine.closeout import context as ctx_mod
from numereng.agentic_research.engine.closeout import evidence as evidence_mod
from numereng.agentic_research.engine.closeout import merge, phases
from numereng.agentic_research.engine.closeout import types as ct
from numereng.features.experiments import ExperimentRecord, get_experiment
from numereng.features.store import resolve_store_root

_PROMPTS_DIR = Path(__file__).parents[2] / "prompts"
_FINALIZE_PROMPT_PATH = _PROMPTS_DIR / "stage-1_finalize.md"
_CLASSIFY_PROMPT_PATH = _PROMPTS_DIR / "stage-2_classify.md"
_EXTRACT_PROMPT_PATH = _PROMPTS_DIR / "stage-3_extract.md"
_SYNTHESIZE_PROMPT_PATH = _PROMPTS_DIR / "stage-4_synthesize.md"


# --------------------------------------------------------------------------- #
# Locks (O_CREAT|O_EXCL; stale = holder dead or older than LOCK_STALE_SECONDS)
# --------------------------------------------------------------------------- #
def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _lock_is_stale(payload: dict[str, object]) -> bool:
    acquired = payload.get("acquired_at")
    if isinstance(acquired, str):
        try:
            acquired_at = datetime.fromisoformat(acquired)
            if acquired_at.tzinfo is None:
                acquired_at = acquired_at.replace(tzinfo=UTC)
            age = (datetime.now(UTC) - acquired_at).total_seconds()
        except ValueError:
            age = None
        if age is not None and age > ct.LOCK_STALE_SECONDS:
            return True
    host = payload.get("hostname")
    pid = payload.get("pid")
    if host == socket.gethostname() and isinstance(pid, int) and not _pid_alive(pid):
        return True
    return False


def _acquire_lock(path: Path) -> None:
    """Create an exclusive lock file; raise CloseoutError if held (or stale, for manual clearing)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        payload: dict[str, object] = {}
        with contextlib.suppress(OSError, json.JSONDecodeError):
            payload = json.loads(path.read_text(encoding="utf-8"))
        if _lock_is_stale(payload):
            raise ct.CloseoutError(ct.err_lock_stale(str(path)))
        raise ct.CloseoutError(ct.err_lock_held(str(path)))
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(
            {"pid": os.getpid(), "hostname": socket.gethostname(), "acquired_at": ar_types.utc_now_iso()},
            handle,
        )


def _release_lock(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink()


# --------------------------------------------------------------------------- #
# Memory root resolution + validation (§2.3)
# --------------------------------------------------------------------------- #
def _resolve_memory_root(root: Path, memory_root: str | None, workspace_root: Path) -> Path:
    if memory_root is None:
        return (root / "notes" / "__RESEARCH_MEMORY__").resolve()
    candidate = Path(memory_root).expanduser()
    if not candidate.is_absolute():
        candidate = workspace_root / candidate
    return candidate.resolve()


def _validate_memory_root(path: Path) -> None:
    if not path.is_dir() or not (path / "CURRENT.md").is_file():
        raise ct.CloseoutError(ct.ERR_MEMORY_ROOT_INVALID)
    topics = path / "topics"
    if not topics.is_dir():
        raise ct.CloseoutError(ct.ERR_MEMORY_ROOT_INVALID)
    for name in ct.MEMORY_TOPIC_FILES:
        if not (topics / f"{name}.md").is_file():
            raise ct.CloseoutError(ct.ERR_MEMORY_ROOT_INVALID)


# --------------------------------------------------------------------------- #
# Gate (§2.1)
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


def _gate(
    experiment: ExperimentRecord,
    state: dict[str, object] | None,
    *,
    accept_stale_running: bool,
    allow_incomplete: bool,
) -> None:
    if experiment.status == "archived":
        raise ct.CloseoutError(ct.ERR_EXPERIMENT_ARCHIVED)
    if state is None or not memory.state_path(experiment).is_file():
        raise ct.CloseoutError(ct.ERR_NOT_AGENTIC)
    if not _journal_nonempty(experiment):
        raise ct.CloseoutError(ct.ERR_NO_ROUNDS)
    if state.get("status") == "running" and not accept_stale_running:
        raise ct.CloseoutError(ct.ERR_RUN_ACTIVE)
    budget = experiment.metadata.get(ar_types.BUDGET_ROUNDS_METADATA_KEY)
    if isinstance(budget, int) and not isinstance(budget, bool):
        done = ar_types.as_int(state.get("total_rounds_completed"), default=0)
        if done < budget and not allow_incomplete:
            raise ct.CloseoutError(ct.err_budget_not_reached(done, budget))


# --------------------------------------------------------------------------- #
# Closeout state persistence
# --------------------------------------------------------------------------- #
def _closeout_dir(experiment: ExperimentRecord) -> Path:
    return memory.agentic_dir(experiment) / ct.CLOSEOUT_DIRNAME


def _write_state(state: ct.CloseoutState, path: Path) -> None:
    ar_types.write_json(path, state.to_dict())


def _load_or_init_state(state_path: Path, *, experiment_id: str, memory_root_identity: str) -> ct.CloseoutState:
    if state_path.is_file():
        raw = json.loads(state_path.read_text(encoding="utf-8"))
        state = ct.CloseoutState.from_dict(raw)
        if state.memory_root_identity and state.memory_root_identity != memory_root_identity:
            raise ct.CloseoutError(ct.ERR_MEMORY_ROOT_CHANGED)
        phases_raw = raw.get("phases") if isinstance(raw, dict) else None
        if isinstance(phases_raw, dict) and ct.PHASE_CLASSIFY not in phases_raw:
            if state.phases[ct.PHASE_CLASSIFY].notes == ct.LEGACY_MASTER_NOTE:
                _write_state(state, state_path)
        return state
    state = ct.CloseoutState.new(experiment_id=experiment_id, memory_root_identity=memory_root_identity)
    _write_state(state, state_path)
    return state


# --------------------------------------------------------------------------- #
# Commit protocol + crash roll-forward (§2.4)
# --------------------------------------------------------------------------- #
def _commit_path(closeout_dir: Path) -> Path:
    return closeout_dir / ct.CLOSEOUT_COMMIT_FILENAME


def _rmtree(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def _apply_commit(closeout_dir: Path, commit: dict[str, object], state: ct.CloseoutState, state_path: Path) -> None:
    """Idempotently apply a commit journal: write any slot still at its old hash, then mark done."""
    phase = str(commit["phase"])
    slots = commit.get("slots", [])
    derived_outputs: dict[str, str] = {}
    for slot in slots if isinstance(slots, list) else []:
        dest = Path(str(slot["path"]))
        stage = closeout_dir / str(slot["stage"])
        new_sha = str(slot["new_sha256"])
        old_sha = slot.get("old_sha256")
        derived_outputs[str(slot["path"])] = new_sha
        on_disk = ct.sha256_file(dest)
        if on_disk == new_sha:
            continue
        if on_disk == old_sha:
            content = stage.read_text(encoding="utf-8")
            if ct.sha256_text(content) != new_sha:
                raise ct.CloseoutError(ct.err_commit_conflict(str(dest)))
            ct.write_text_atomic(dest, content)
            continue
        raise ct.CloseoutError(ct.err_commit_conflict(str(dest)))
    # Recorded outputs default to the whole-file slot hashes, but a phase may record a different
    # upstream fingerprint (synthesize records per-experiment entry-block hashes, not whole ledgers).
    outputs_override = commit.get("outputs")
    outputs = (
        {str(k): str(v) for k, v in outputs_override.items()} if isinstance(outputs_override, dict) else derived_outputs
    )
    state.phases[phase] = ct.CloseoutPhaseState(
        status="done",
        completed_at=str(commit.get("completed_at")) if commit.get("completed_at") else ar_types.utc_now_iso(),
        duration_seconds=commit.get("duration_seconds")
        if isinstance(commit.get("duration_seconds"), (int, float))
        else None,
        notes=str(commit.get("notes")) if commit.get("notes") is not None else None,
        outputs=outputs,
    )
    _write_state(state, state_path)
    _commit_path(closeout_dir).unlink(missing_ok=True)
    _rmtree(closeout_dir / "stage" / phase)


def _roll_forward(closeout_dir: Path, state: ct.CloseoutState, state_path: Path) -> None:
    commit_path = _commit_path(closeout_dir)
    if not commit_path.is_file():
        return
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    _apply_commit(closeout_dir, commit, state, state_path)


def _commit_phase(
    closeout_dir: Path,
    *,
    phase: str,
    slots_content: dict[str, str],
    notes: str | None,
    duration_seconds: float | None,
    state: ct.CloseoutState,
    state_path: Path,
    outputs: dict[str, str] | None = None,
) -> None:
    """Stage -> write commit.json -> apply slots -> mark phase done -> delete journal + stage.

    ``outputs`` overrides the recorded upstream fingerprints; when None they default to the
    whole-file slot hashes (used by finalize + extract). Synthesize passes per-experiment
    entry-block hashes so a cross-experiment ledger change never trips ``_verify_upstream``.
    """
    stage_dir = closeout_dir / "stage" / phase
    _rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    items = sorted(slots_content.items())
    slot_records: list[dict[str, object]] = []
    for index, (dest_str, content) in enumerate(items):
        stage_file = stage_dir / f"slot_{index}"
        ct.write_text_atomic(stage_file, content)
        slot_records.append(
            {
                "path": dest_str,
                "stage": str(stage_file.relative_to(closeout_dir)),
                "old_sha256": ct.sha256_file(Path(dest_str)),
                "new_sha256": ct.sha256_text(content),
            }
        )
    commit: dict[str, object] = {
        "phase": phase,
        "completed_at": ar_types.utc_now_iso(),
        "duration_seconds": duration_seconds,
        "notes": notes,
        "slots": slot_records,
    }
    if outputs is not None:
        commit["outputs"] = dict(outputs)
    ar_types.write_json(_commit_path(closeout_dir), commit)
    _apply_commit(closeout_dir, commit, state, state_path)


# --------------------------------------------------------------------------- #
# Phase planning + upstream integrity
# --------------------------------------------------------------------------- #
def _plan(state: ct.CloseoutState, until: str | None) -> list[str]:
    plan: list[str] = []
    for phase in ct.PHASE_ORDER:
        if state.phases[phase].status not in ct.PHASE_TERMINAL_STATUSES:
            plan.append(phase)
        if until is not None and phase == until:
            break
    return plan


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _verify_upstream(state: ct.CloseoutState, closeout_dir: Path, *, memory_root: Path, experiment_id: str) -> None:
    """Re-verify recorded upstream fingerprints of already-done phases.

    Closeout-owned files (memo, evidence) and the extract branch files are whole-file hashes. Topic
    ledgers are verified only by this experiment's ``### <id>`` entry-block hash, so a cross-experiment
    append to a shared ledger between phases does not trip staleness. CURRENT.md is a full rewrite,
    never an upstream artifact, and is never recorded/verified.
    """
    branch_dir = memory_root / "experiments" / experiment_id
    topics_dir = memory_root / "topics"
    for phase in ct.PHASE_ORDER:
        record = state.phases[phase]
        if record.status != "done":
            continue
        for dest_str, expected in record.outputs.items():
            dest = Path(dest_str)
            if _is_relative_to(dest, closeout_dir) or _is_relative_to(dest, branch_dir):
                actual = ct.sha256_file(dest)
            elif dest.parent == topics_dir and dest.name != ct.CURRENT_MD_FILENAME:
                text = ar_types.read_text(dest, limit=ct.MAX_CLOSEOUT_CONTEXT_CHARS)
                topic = dest.stem
                block = merge.extract_entry_block(text, experiment_id, topic=topic) if text else ""
                actual = ct.sha256_text(block)
            else:
                continue
            if actual != expected:
                raise ct.CloseoutError(ct.err_stale_upstream(phase, dest_str))


# --------------------------------------------------------------------------- #
# FINALIZE phase (§3.1)
# --------------------------------------------------------------------------- #
def _run_finalize(
    *,
    experiment: ExperimentRecord,
    state: dict[str, object],
    evidence: dict[str, object],
    closeout_dir: Path,
) -> tuple[dict[str, str], str]:
    """Build context, call the LLM (codex transport), validate the memo; return {abs_path: content}, notes."""
    program_text = _FINALIZE_PROMPT_PATH.read_text(encoding="utf-8")
    finalize_context = ctx_mod.build_finalize_context(
        experiment=experiment, state=state, evidence=evidence, program_text=program_text
    )
    prompt = llm.render_prompt(finalize_context, program_path=_FINALIZE_PROMPT_PATH)
    debug_dir = closeout_dir / "debug"
    raw, _source = llm._call_research_llm(
        prompt=prompt,
        artifact_dir=debug_dir,
        round_label=ct.PHASE_FINALIZE,
        schema=phases.FILES_ENVELOPE_SCHEMA,
        timeout_seconds=ct.CLOSEOUT_TIMEOUT_SECONDS,
        transport="codex",
    )
    try:
        files, notes = phases.parse_files_envelope(raw)
        believed_best = evidence.get("believed_best", {})
        believed_best_config = believed_best.get("config") if isinstance(believed_best, dict) else None
        slots = phases.validate_finalize(
            files,
            experiment_id=experiment.experiment_id,
            believed_best_config=str(believed_best_config or ""),
        )
    except ct.CloseoutError as exc:
        memory.write_failure_debug(
            artifact_dir=debug_dir, round_label=ct.PHASE_FINALIZE, prompt=prompt, error=str(exc), raw_response=raw
        )
        raise
    dest_content = {str(closeout_dir / rel_path): content for rel_path, content in slots.items()}
    return dest_content, notes


# --------------------------------------------------------------------------- #
# CLASSIFY phase
# --------------------------------------------------------------------------- #
def _run_classify(*, closeout_dir: Path) -> tuple[dict[str, str], str, dict[str, object]]:
    """Classify the finalized experiment before any research-memory write."""
    memo_path = closeout_dir / ct.CLOSEOUT_MEMO_FILENAME
    if not memo_path.is_file():
        raise ct.CloseoutError(ct.err_memo_missing(str(memo_path)))
    classify_context = ctx_mod.build_classify_context(
        memo_text=memo_path.read_text(encoding="utf-8"),
        evidence=_load_evidence(closeout_dir),
    )
    prompt = llm.render_prompt(classify_context, program_path=_CLASSIFY_PROMPT_PATH)
    debug_dir = closeout_dir / "debug"
    raw, _source = llm._call_research_llm(
        prompt=prompt,
        artifact_dir=debug_dir,
        round_label=ct.PHASE_CLASSIFY,
        schema=phases.CLASSIFY_SCHEMA,
        timeout_seconds=ct.CLOSEOUT_TIMEOUT_SECONDS,
        transport="codex",
    )
    try:
        classification = phases.parse_classification(raw)
    except ct.CloseoutError as exc:
        memory.write_failure_debug(
            artifact_dir=debug_dir,
            round_label=ct.PHASE_CLASSIFY,
            prompt=prompt,
            error=str(exc),
            raw_response=raw,
        )
        raise
    content = json.dumps(classification, indent=2, sort_keys=True) + "\n"
    return (
        {str(closeout_dir / ct.CLOSEOUT_CLASSIFICATION_FILENAME): content},
        str(classification["rationale"]),
        classification,
    )


def _load_classification(closeout_dir: Path, state: ct.CloseoutState) -> dict[str, object]:
    """Load new classification state or map a progressed schema-v1 chain to legacy master."""
    if state.phases[ct.PHASE_CLASSIFY].notes == ct.LEGACY_MASTER_NOTE:
        return {"disposition": "master", "relevant_topics": list(ct.MEMORY_TOPIC_FILES)}
    path = closeout_dir / ct.CLOSEOUT_CLASSIFICATION_FILENAME
    if not path.is_file():
        raise ct.CloseoutError(ct.err_evidence_missing(str(path)))
    return phases.parse_classification(path.read_text(encoding="utf-8"))


def _apply_classification_route(
    classification: dict[str, object],
    *,
    state: ct.CloseoutState,
    state_path: Path,
) -> None:
    """Persist downstream skips implied by one classification."""
    disposition = str(classification["disposition"])
    skipped: tuple[str, ...] = ()
    if disposition == "branch_only":
        skipped = (ct.PHASE_SYNTHESIZE,)
    elif disposition == "exclude":
        skipped = (ct.PHASE_EXTRACT, ct.PHASE_SYNTHESIZE)
    for phase in skipped:
        state.phases[phase] = ct.CloseoutPhaseState(status="skipped", notes=f"classification:{disposition}")
    if skipped:
        _write_state(state, state_path)


# --------------------------------------------------------------------------- #
# EXTRACT phase (§3.2)
# --------------------------------------------------------------------------- #
def _branch_files(branch_dir: Path) -> list[Path]:
    return [p for p in branch_dir.iterdir() if p.is_file()] if branch_dir.is_dir() else []


def _backup_branch(branch_dir: Path, closeout_dir: Path) -> None:
    """Copy an existing branch to closeout/backups/<timestamp>/ before a restart overwrites it."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = closeout_dir / "backups" / stamp
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in _branch_files(branch_dir):
        shutil.copy2(path, backup_dir / path.name)


def _load_evidence(closeout_dir: Path) -> dict[str, object]:
    evidence_path = closeout_dir / ct.CLOSEOUT_EVIDENCE_FILENAME
    if not evidence_path.is_file():
        raise ct.CloseoutError(ct.err_evidence_missing(str(evidence_path)))
    return json.loads(evidence_path.read_text(encoding="utf-8"))


def _run_extract(
    *,
    experiment: ExperimentRecord,
    memory_root: Path,
    closeout_dir: Path,
    restart: bool,
) -> tuple[dict[str, str], str]:
    """Draft the seven-file research-memory branch (README + six topics); return {abs_path: content}, notes."""
    branch_dir = memory_root / "experiments" / experiment.experiment_id
    if _branch_files(branch_dir):
        if not restart:
            raise ct.CloseoutError(ct.err_branch_exists(str(branch_dir)))
        _backup_branch(branch_dir, closeout_dir)
        _rmtree(branch_dir)

    memo_path = closeout_dir / ct.CLOSEOUT_MEMO_FILENAME
    if not memo_path.is_file():
        raise ct.CloseoutError(ct.err_memo_missing(str(memo_path)))
    memo_text = memo_path.read_text(encoding="utf-8")
    evidence = _load_evidence(closeout_dir)

    extract_context = ctx_mod.build_extract_context(experiment=experiment, memo_text=memo_text, evidence=evidence)
    prompt = llm.render_prompt(extract_context, program_path=_EXTRACT_PROMPT_PATH)
    debug_dir = closeout_dir / "debug"
    raw, _source = llm._call_research_llm(
        prompt=prompt,
        artifact_dir=debug_dir,
        round_label=ct.PHASE_EXTRACT,
        schema=phases.FILES_ENVELOPE_SCHEMA,
        timeout_seconds=ct.CLOSEOUT_TIMEOUT_SECONDS,
        transport="codex",
    )
    try:
        files, notes = phases.parse_files_envelope(raw)
        slots = phases.validate_extract(files, experiment_id=experiment.experiment_id)
    except ct.CloseoutError as exc:
        memory.write_failure_debug(
            artifact_dir=debug_dir, round_label=ct.PHASE_EXTRACT, prompt=prompt, error=str(exc), raw_response=raw
        )
        raise
    dest_content = {str(branch_dir / rel_path): content for rel_path, content in slots.items()}
    return dest_content, notes


def _assert_branch_complete(branch_dir: Path) -> None:
    """After committing extract, the branch dir must hold exactly the seven canonical files."""
    files = {p.name for p in _branch_files(branch_dir)}
    expected = {ct.MEMORY_BRANCH_README, *(f"{topic}.md" for topic in ct.MEMORY_TOPIC_FILES)}
    if files != expected:
        raise ct.CloseoutError(ct.err_branch_file_count(len(files)))


# --------------------------------------------------------------------------- #
# SYNTHESIZE phase (§3.3)
# --------------------------------------------------------------------------- #
def _run_synthesize(
    *,
    experiment: ExperimentRecord,
    memory_root: Path,
    closeout_dir: Path,
    restart: bool,
    relevant_topics: tuple[str, ...],
) -> tuple[dict[str, str], str, dict[str, str]]:
    """Merge selected topic deltas and rewrite CURRENT.md."""
    experiment_id = experiment.experiment_id
    branch_dir = memory_root / "experiments" / experiment_id
    topics_dir = memory_root / "topics"

    synth_context = ctx_mod.build_synthesize_context(
        experiment_id=experiment_id,
        memory_root=memory_root,
        relevant_topics=relevant_topics,
    )
    prompt = llm.render_prompt(synth_context, program_path=_SYNTHESIZE_PROMPT_PATH)
    debug_dir = closeout_dir / "debug"
    raw, _source = llm._call_research_llm(
        prompt=prompt,
        artifact_dir=debug_dir,
        round_label=ct.PHASE_SYNTHESIZE,
        schema=phases.SYNTHESIZE_SCHEMA,
        timeout_seconds=ct.CLOSEOUT_TIMEOUT_SECONDS,
        transport="codex",
    )
    try:
        deltas, current_md, notes = phases.parse_synthesize_envelope(raw, relevant_topics=relevant_topics)
        phases.validate_current_md(current_md, experiment_id=experiment_id)
        slots_content: dict[str, str] = {}
        outputs: dict[str, str] = {}
        for topic in relevant_topics:
            new_entry = str(deltas[topic]["new_entry"])
            overview = deltas[topic]["overview"]
            best_understanding = deltas[topic]["best_understanding"]
            phases.validate_entry_block(topic, new_entry, experiment_id=experiment_id)
            if not (branch_dir / f"{topic}.md").is_file():
                raise ct.CloseoutError(ct.err_branch_file_missing(f"{topic}.md"))
            ledger_path = topics_dir / f"{topic}.md"
            existing = ledger_path.read_text(encoding="utf-8")
            merged = merge.merge_ledger(
                existing,
                new_entry=new_entry,
                overview_replacement=overview if isinstance(overview, str) else None,
                best_understanding_replacement=best_understanding if isinstance(best_understanding, str) else None,
                experiment_id=experiment_id,
                replace_existing=restart,
                topic=topic,
            )
            _preamble, _overview, _best_understanding, learnings = merge.parse_ledger(merged, topic=topic)
            count = merge.count_entries(learnings, experiment_id)
            if count != 1:
                raise ct.CloseoutError(ct.err_entry_count_invalid(topic, count))
            slots_content[str(ledger_path)] = merged
            outputs[str(ledger_path)] = ct.sha256_text(merge.extract_entry_block(merged, experiment_id, topic=topic))
        slots_content[str(memory_root / ct.CURRENT_MD_FILENAME)] = current_md
    except ct.CloseoutError as exc:
        memory.write_failure_debug(
            artifact_dir=debug_dir, round_label=ct.PHASE_SYNTHESIZE, prompt=prompt, error=str(exc), raw_response=raw
        )
        raise
    return slots_content, notes, outputs


def _backup_synthesize_memory(
    *,
    memory_root: Path,
    closeout_dir: Path,
    relevant_topics: tuple[str, ...],
) -> None:
    """Snapshot CURRENT.md and selected topic ledgers before SYNTHESIZE commits."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    backup_root = closeout_dir / "backups"
    backup_dir = backup_root / stamp
    stage_dir = backup_root / f".{stamp}.tmp"
    paths = [Path(ct.CURRENT_MD_FILENAME), *(Path("topics") / f"{topic}.md" for topic in relevant_topics)]
    try:
        for relative_path in paths:
            destination = stage_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(memory_root / relative_path, destination)
        os.replace(stage_dir, backup_dir)
    except OSError as exc:
        _rmtree(stage_dir)
        raise ct.CloseoutError(ct.ERR_SYNTHESIZE_BACKUP_FAILED) from exc


# --------------------------------------------------------------------------- #
# Result assembly
# --------------------------------------------------------------------------- #
def _reports(state: ct.CloseoutState) -> tuple[ct.CloseoutPhaseReport, ...]:
    return tuple(
        ct.CloseoutPhaseReport(
            name=phase,
            status=state.phases[phase].status,
            notes=state.phases[phase].notes,
            duration_seconds=state.phases[phase].duration_seconds,
            outputs=dict(state.phases[phase].outputs),
        )
        for phase in ct.PHASE_ORDER
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def get_closeout_status(*, store_root: str | Path = ".numereng", experiment_id: str) -> ct.CloseoutResult:
    """Read-only report of the closeout chain state; loads all-pending if never run."""
    root = resolve_store_root(store_root)
    experiment = get_experiment(store_root=root, experiment_id=experiment_id)
    state_path = _closeout_dir(experiment) / ct.CLOSEOUT_STATE_FILENAME
    if state_path.is_file():
        state = ct.CloseoutState.from_dict(json.loads(state_path.read_text(encoding="utf-8")))
    else:
        state = ct.CloseoutState.new(experiment_id=experiment.experiment_id, memory_root_identity="")
    return ct.CloseoutResult(
        experiment_id=experiment.experiment_id, phases=_reports(state), stopped_at_phase=None, error=None
    )


def run_closeout(
    *,
    store_root: str | Path = ".numereng",
    experiment_id: str,
    until: str | None = None,
    restart_from: str | None = None,
    memory_root: str | None = None,
    accept_stale_running: bool = False,
    allow_incomplete: bool = False,
) -> ct.CloseoutResult:
    """Run the closeout chain up to ``until``. Gate failures raise; phase failures are captured."""
    if until is not None and until not in ct.PHASE_ORDER:
        raise ct.CloseoutError(ct.err_until_invalid(until))
    if restart_from is not None:
        if restart_from not in ct.PHASE_ORDER:
            raise ct.CloseoutError(ct.err_restart_from_invalid(restart_from))
        if restart_from not in ct.IMPLEMENTED_PHASES:
            raise ct.CloseoutError(ct.err_phase_not_implemented(restart_from))

    root = resolve_store_root(store_root)
    workspace_root = root.parent
    experiment = get_experiment(store_root=root, experiment_id=experiment_id)

    run_state = memory.load_state(memory.state_path(experiment))
    _gate(experiment, run_state, accept_stale_running=accept_stale_running, allow_incomplete=allow_incomplete)
    run_state = run_state or {}

    memory_root_path = _resolve_memory_root(root, memory_root, workspace_root)
    _validate_memory_root(memory_root_path)

    closeout_dir = _closeout_dir(experiment)
    closeout_dir.mkdir(parents=True, exist_ok=True)
    lock_path = closeout_dir / ct.CLOSEOUT_LOCK_FILENAME
    _acquire_lock(lock_path)
    try:
        state_path = closeout_dir / ct.CLOSEOUT_STATE_FILENAME
        state = _load_or_init_state(
            state_path, experiment_id=experiment.experiment_id, memory_root_identity=str(memory_root_path)
        )
        _roll_forward(closeout_dir, state, state_path)

        # Restarting upstream of a completed merge can orphan its ledger entries. Restart at
        # SYNTHESIZE or later when master memory has already changed.
        upstream_restarts = (ct.PHASE_FINALIZE, ct.PHASE_CLASSIFY, ct.PHASE_EXTRACT)
        if restart_from in upstream_restarts and state.phases[ct.PHASE_SYNTHESIZE].status == "done":
            raise ct.CloseoutError(ct.ERR_RESTART_BLOCKED_AFTER_SYNTHESIZE)
        if restart_from in (ct.PHASE_FINALIZE, ct.PHASE_CLASSIFY) and state.phases[ct.PHASE_EXTRACT].status == "done":
            raise ct.CloseoutError(ct.ERR_RESTART_BLOCKED_AFTER_EXTRACT)

        if restart_from is not None:
            resetting = False
            for phase in ct.PHASE_ORDER:
                if phase == restart_from:
                    resetting = True
                if resetting:
                    state.phases[phase] = ct.CloseoutPhaseState()
            _write_state(state, state_path)

        if state.phases[ct.PHASE_CLASSIFY].status == "done":
            _apply_classification_route(
                _load_classification(closeout_dir, state),
                state=state,
                state_path=state_path,
            )

        plan = _plan(state, until)
        evidence: dict[str, object] | None = None
        if ct.PHASE_FINALIZE in plan:
            evidence = evidence_mod.build_evidence(experiment=experiment, state=run_state, runs_dir=root / "runs")
            ar_types.write_json(closeout_dir / ct.CLOSEOUT_EVIDENCE_FILENAME, evidence)

        needs_memory_lock = any(phase in (ct.PHASE_EXTRACT, ct.PHASE_SYNTHESIZE) for phase in plan)
        memory_lock_path = memory_root_path / ct.MEMORY_ROOT_LOCK_FILENAME
        if needs_memory_lock:
            _acquire_lock(memory_lock_path)
        try:
            stopped_at_phase, error = _run_plan(
                plan,
                experiment=experiment,
                run_state=run_state,
                evidence=evidence,
                memory_root=memory_root_path,
                closeout_dir=closeout_dir,
                state=state,
                state_path=state_path,
                restart_from=restart_from,
            )
        finally:
            if needs_memory_lock:
                _release_lock(memory_lock_path)

        return ct.CloseoutResult(
            experiment_id=experiment.experiment_id,
            phases=_reports(state),
            stopped_at_phase=stopped_at_phase,
            error=error,
        )
    finally:
        _release_lock(lock_path)


def _run_plan(
    plan: list[str],
    *,
    experiment: ExperimentRecord,
    run_state: dict[str, object],
    evidence: dict[str, object] | None,
    memory_root: Path,
    closeout_dir: Path,
    state: ct.CloseoutState,
    state_path: Path,
    restart_from: str | None,
) -> tuple[str | None, str | None]:
    """Execute the planned phases in order; return (stopped_at_phase, error) or (None, None)."""
    for phase in plan:
        if state.phases[phase].status in ct.PHASE_TERMINAL_STATUSES:
            continue
        if phase not in ct.IMPLEMENTED_PHASES:
            return phase, ct.err_phase_not_implemented(phase)
        _verify_upstream(state, closeout_dir, memory_root=memory_root, experiment_id=experiment.experiment_id)
        try:
            started = time.monotonic()
            outputs: dict[str, str] | None = None
            if phase == ct.PHASE_FINALIZE:
                dest_content, notes = _run_finalize(
                    experiment=experiment, state=run_state, evidence=evidence or {}, closeout_dir=closeout_dir
                )
            elif phase == ct.PHASE_CLASSIFY:
                dest_content, notes, classification = _run_classify(closeout_dir=closeout_dir)
            elif phase == ct.PHASE_EXTRACT:
                dest_content, notes = _run_extract(
                    experiment=experiment,
                    memory_root=memory_root,
                    closeout_dir=closeout_dir,
                    restart=restart_from == ct.PHASE_EXTRACT,
                )
            elif phase == ct.PHASE_SYNTHESIZE:
                classification = _load_classification(closeout_dir, state)
                relevant_topics = tuple(cast("list[str]", classification["relevant_topics"]))
                dest_content, notes, outputs = _run_synthesize(
                    experiment=experiment,
                    memory_root=memory_root,
                    closeout_dir=closeout_dir,
                    restart=restart_from == ct.PHASE_SYNTHESIZE,
                    relevant_topics=relevant_topics,
                )
                _backup_synthesize_memory(
                    memory_root=memory_root,
                    closeout_dir=closeout_dir,
                    relevant_topics=relevant_topics,
                )
            else:  # pragma: no cover - guarded by the IMPLEMENTED_PHASES check above
                return phase, ct.err_phase_not_implemented(phase)
            _commit_phase(
                closeout_dir,
                phase=phase,
                slots_content=dest_content,
                notes=notes,
                duration_seconds=time.monotonic() - started,
                state=state,
                state_path=state_path,
                outputs=outputs,
            )
            if phase == ct.PHASE_EXTRACT:
                _assert_branch_complete(memory_root / "experiments" / experiment.experiment_id)
            elif phase == ct.PHASE_CLASSIFY:
                _apply_classification_route(classification, state=state, state_path=state_path)
        except ar_types.AgenticResearchError as exc:
            return phase, str(exc)
    return None, None
