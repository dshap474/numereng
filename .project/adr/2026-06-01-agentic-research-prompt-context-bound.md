# ADR: Agentic Research Prompt Context — Bound Unbounded Collections

**Status:** Accepted
**Date:** 2026-06-01
**Commits:** a676b8a (state-dump trim), d1cedaf (config-context trim)

---

## Context

On the 521-round `2026-05-27_wide-diversification-test` run the rendered LLM prompt reached
~860 KB (measured from the r517 `rNNN.debug.prompt.md` artifact). The codex-exec stream
disconnected repeatedly on oversized prompts, accumulating `agentic_research_codex_failed`
events and triggering the 5-consecutive-failure bail at round 521. The run did not converge
— it crashed.

Byte-level decomposition of the r517 prompt:

| Term | Bytes | Share | Growth |
| --- | --- | --- | --- |
| `configs` (all generated configs embedded) | 498,943 | 60% | +1 entry per round |
| `state` raw dump (incl. `confirmations`) | 176,426 | 21% | unbounded (`confirmations`) |
| `tried_signatures` top-level key | ~32,000 | 4% | capped at 100 entries |
| `latest_round_markdown` | ~2,800 | <1% | already capped at `MAX_CONTEXT_CHARS` |
| all other terms | remainder | — | bounded |

**`configs` was the dominant term (60%).** `_build_context` embedded every config file on
disk: after 521 rounds that was ~500 files (~499 KB). This grows by one entry per round with
no cap.

**`state` was the secondary term (21%).** `_build_context` passed the full raw `state` dict
under `context["state"]`. Two collections inside `state` grow without bound:

- `confirmations` — one entry per config ever confirmed, append-only. At r517 it held 481
  entries (~126 KB of the 176 KB state dump).
- `tried_signatures` — the 100-entry rolling dedup window (~32 KB). Already capped, but
  duplicated verbatim both inside the raw `state` dump and as a dedicated top-level key.

**`latest_round_markdown` was not the problem.** It is already bounded by `MAX_CONTEXT_CHARS`
and contributed only ~2.8 KB. The "rolling memo is too large" theory is incorrect; do not
re-investigate it.

---

## Decision

Two fixes in `run.py`, one per unbounded term.

### Fix 1 — Config context (commit d1cedaf, primary term)

New helper `_relevant_config_paths(state, config_dir)` + constant `CONFIG_CONTEXT_RECENT = 40`.
`_build_context` now surfaces only:

- all seed configs (files not matching `config_NNN.json` — the non-generated baseline set),
- the confirmed champion's config (`state.confirmed_champion.parent_config`), retained
  regardless of age so the champion config is always visible to the LLM even if it was
  generated early (e.g. `config_040` from round 42 stays in context),
- the most recent 40 generated configs by filename sort.

Older generated configs remain on disk and are still fully nameable as `parent_config` in
LLM decisions — `_materialize_decision_config` resolves against disk via
`parent_path.is_file()`, not the context list. They are also visible by metric in the
experiment report. The context trim is rendering-only.

Measured on the real 521-round state: configs in prompt 499 KB → 43 KB (42 configs after
champion dedup, 91% reduction).

### Fix 2 — State-dump trim (commit a676b8a, secondary term)

New helper `_state_context(state)` guarded by module-level constant
`_STATE_CONTEXT_OMIT_KEYS = frozenset({"confirmations", "tried_signatures"})`. Returns a
shallow copy of `state` with those two keys removed. `_build_context` passes this trimmed
copy under `context["state"]` instead of the raw dict.

Both collections were already surfaced as dedicated top-level context keys:
`confirmations` capped at 30 via `_recent_confirmations`, and `tried_signatures` verbatim.
The raw `state` copy was pure duplication.

The programs active on long runs (`PROGRAM.md`, `custom_programs/`) read only scalar state
fields — `phase`, `next_round_number`, `total_rounds_completed` — and `phase_history`. All
are retained; none are in the omitted keys. Verified by grepping every active program
markdown.

Measured on the real 521-round state: state-derived prompt bytes 176 KB → 47 KB (73%
reduction).

**Scope of both fixes:** prompt-rendering only. `state.json` on disk is unchanged — it
retains the full `confirmations` and `tried_signatures` collections as the source of truth
for rotation, promotion, and resume logic. Config files on disk are unchanged. Only the LLM
prompt copies are trimmed.

**Combined effect:** projected r517 prompt ~860 KB → ~130 KB. No prompt term grows
unbounded with round count.

---

## Alternatives Considered

- **Truncate `confirmations` in `_recent_confirmations` more aggressively:** shrinks the
  curated top-level key but does not fix the raw `state` copy, which would still embed all
  481 entries.

- **Cap configs at a smaller window (e.g., 10 or 20):** would reduce prompt size further but
  risks hiding a recently generated config the LLM may want as a parent. 40 is large enough
  to cover a full shallow phase and still includes the champion explicitly.

- **Remove `context["state"]` entirely:** the scalar fields and `phase_history` are
  genuinely useful to programs for budget-awareness and phase-state reasoning. Dropping the
  whole key would regress that.

- **Compress or summarize the raw state dump:** adds complexity without benefit when the LLM
  programs do not read `confirmations` or `tried_signatures` from `context["state"]` at all.

- **Investigate `latest_round_markdown` as the culprit:** ruled out by measurement. It is
  already capped at `MAX_CONTEXT_CHARS` and contributed only ~2.8 KB at r517.

---

## Consequences

- No prompt term grows unbounded with round count. A 500-round run produces the same bounded
  prompt size as a 10-round run.
- LLM decisions referencing older configs (outside the recent-40 window) are still valid:
  `_materialize_decision_config` resolves against disk, not the context list. Older configs
  remain discoverable via the experiment report.
- `_STATE_CONTEXT_OMIT_KEYS` is the extension point if future state fields exhibit the same
  unbounded-growth pattern.
- Tests added: `test_state_context_drops_unbounded_collections_keeps_program_fields`,
  `test_build_context_prompt_size_bounded_as_confirmations_grow`, and
  `test_config_context_bounds_recent_plus_seed_and_champion`. 132 tests pass.
