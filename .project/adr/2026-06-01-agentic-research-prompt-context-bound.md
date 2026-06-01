# ADR: Agentic Research Prompt Context — Bound Unbounded Collections

**Status:** Accepted
**Date:** 2026-06-01
**Commit:** a676b8a

---

## Context

The `_build_context` helper in `run.py` assembled the LLM prompt by passing the full raw
`state` dict under `context["state"]`. Two collections inside `state` grow without bound
across a run:

- `confirmations` — one entry per config ever confirmed, append-only. After the 521-round
  `2026-05-27_wide-diversification-test` run it held 481 entries (~126 KB of JSON).
- `tried_signatures` — the 100-entry rolling dedup window (~32 KB).

Both collections were already surfaced as dedicated top-level context keys:
`confirmations` capped at 30 via `_recent_confirmations`, and `tried_signatures` verbatim
(~line 1563). The raw `state` copy was pure duplication for these two keys, and
`confirmations` was the single unbounded growth term in the prompt.

On the 521-round run the rendered prompt reached ~890 KB. The codex-exec LLM stream
disconnected repeatedly on oversized prompts, accumulating `agentic_research_codex_failed`
events and ultimately triggering the 5-consecutive-failure bail at round 521. The run did
not converge — it crashed. The raw state dump was confirmed as the cause by inspecting the
`rNNN.debug.prompt.md` artifacts: the prompt size scaled linearly with `len(confirmations)`.

The programs active on long runs (`PROGRAM.md`, `custom_programs/`) read only scalar state
fields — `phase`, `next_round_number`, `total_rounds_completed` — and `phase_history`. All
of these are retained in the stripped copy; none are in the omitted keys. This was verified
by grepping every active program markdown.

---

## Decision

Introduce helper `_state_context(state)` in `run.py`, guarded by the module-level constant
`_STATE_CONTEXT_OMIT_KEYS = frozenset({"confirmations", "tried_signatures"})`. The helper
returns a shallow copy of `state` with those two keys removed. `_build_context` passes this
trimmed copy under `context["state"]` instead of the raw dict.

**Scope:** prompt-rendering only. `state.json` on disk is unchanged — it remains the source
of truth for the full collections, used by rotation, promotion, and resume logic. Only the
LLM prompt copy is trimmed.

Measured on the real 521-round state: state-derived prompt bytes fell from ~237 KB to ~47 KB
(80% reduction). The prompt no longer grows with round count; it is bounded by the fixed
scalar fields and `phase_history`.

---

## Alternatives Considered

- **Truncate `confirmations` in `_recent_confirmations` more aggressively (e.g., 10 entries
  instead of 30):** this would shrink the curated top-level key but would not fix the root
  cause — the raw `state` copy would still embed all 481 entries untruncated.

- **Compress or summarize the raw state dump before embedding it:** adds complexity without
  benefit when the LLM programs do not read the full collections from `context["state"]` at
  all. Stripping is simpler and lossless for the actual consumer.

- **Remove `context["state"]` entirely:** the scalar fields (`phase`, `next_round_number`,
  `total_rounds_completed`) and `phase_history` are genuinely useful to programs for
  budget-awareness and phase-state reasoning. Dropping the whole key would regress that.

---

## Consequences

- Prompt size is now bounded regardless of run length. A 500-round run produces the same
  `context["state"]` size as a 10-round run.
- Programs that previously read `confirmations` or `tried_signatures` from `context["state"]`
  would now see them absent. No active program does this (verified); both collections remain
  available via their dedicated top-level keys.
- `_STATE_CONTEXT_OMIT_KEYS` is the extension point if future state fields exhibit the same
  unbounded-growth pattern.
- Tests added: `test_state_context_drops_unbounded_collections_keeps_program_fields` (field
  retention) and `test_build_context_prompt_size_bounded_as_confirmations_grow` (prompt for
  1000 confirmations ≈ prompt for 10). 131 tests pass.
