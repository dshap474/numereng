# ADR: Agentic Research Controller-Level Enforcement for Diversification and Inert-Knob Prevention

**Status:** Accepted
**Date:** 2026-05-28
**Commits:** 5461142 (controller + tests), 4743570 (base PROGRAM.md prompt)

---

## Context

A 45-round experiment (`2026-05-27_wide-diversification-test`) audited the agentic HPO/architecture search loop (`src/numereng/features/agentic_research/run.py`). The search diversified well for ~27 rounds then collapsed into tunnel-tuning a single cell (XGB/small/target_alpha_60) for its back half, including re-running bit-identical models (min_child_weight 100→50→200 all scored identically — an inert knob the LLM itself acknowledged yet re-probed).

Root cause: diversification, deduplication, and confirmation discipline were delegated to the LLM as soft prompt rules. The LLM did not reliably self-enforce. The old observe-only `_check_diversification_dry_run` never fired in practice (threshold 8, real streak 6; cell-exact counting so a family flip reset it without addressing target-level tunneling).

---

## Decision

Move critical invariants to controller-level enforcement. Six fixes (F1–F6) across controller and prompt.

### F1 — Diversification enforcement (hybrid soft + hard)

**Rejected:** purely soft (proved insufficient); purely hard (too brittle for edge cases).
**Chosen:** hybrid. `_diversification_streaks` tracks consecutive seed-42 discovery streaks at two granularities: exact cell `(family, feature_set, target)` AND target-only. Soft directive injected into LLM context at streak ≥ `DIVERSIFICATION_SOFT_THRESHOLD=4`; hard reject of a streak-extending seed-42 discovery at ≥ `DIVERSIFICATION_HARD_THRESHOLD=6` (confirmations exempt), gated by `DIVERSIFICATION_ENFORCED=True`. Removed `_check_diversification_dry_run` and `DIVERSIFICATION_CELL_THRESHOLD`.

### F2 — Inert-knob detect and prevent

**Scope chosen:** `(parent_config, path)` pair (not global axis blacklist — too broad; not per-run global — persists stale data across very different parents).
`_detect_inert_change` flags a single-axis discovery whose primary metric equals its parent's seed-42 metric within `INERT_METRIC_EPSILON=1e-9`, recorded in new state field `inert_axes`. `_reject_inert_change` blocks single-axis re-probes of that pair; multi-axis probes are allowed to unblock genuine compound searches. Also added `_normalize_xgb_effective_params` (clamp XGB max_leaves to 2**max_depth, mirroring the existing LGBM rule) closing a dedup gap the prompt promised but the controller did not enforce.

### F3 — Confirm winners not ties (prompt)

Confirmation-backlog trigger changed from "within 3e-4 of champion" to ">3e-4 ABOVE champion seed-42", with explicit "do not confirm a tie". Removes the confirmation treadmill that consumed budget confirming non-improvements during the exploit phase.

### F4 — Parent selection (prompt)

New-axis probe must branch from the champion, never from a regressed config. Prevents compounding regressions during exploration.

### F5 — Phase-transition deferral

`_has_inflight_confirmation` + guard in `_maybe_transition_phase`: defer phase transition while a confirmation trio is mid-flight (1–2 canonical seeds done, not yet promoted, last_attempt within 1 round). Ensures a champion discovered late in a phase is credited to that phase, not the next. Recency guard prevents abandoned partials blocking indefinitely. New trace event: `phase_transition_deferred`. Consequence: a fresh champion promoted at a graduation boundary resets the plateau and correctly delays graduation — consistent with the program's own semantics.

### F6 — Budget from experiment metadata

`_budget_rounds` reads metadata key `agentic_research_budget_rounds` into context as `experiment.budget_rounds` (null when unset). Replaces hard-coded 250-round assumptions so longer runs (500-round/week-long) are not anchored to a stale literal.

---

## New state / context surface

- State field: `inert_axes` (auto-migrated via `_apply_state_defaults`)
- Trace events: `inert_change_detected`, `phase_transition_deferred`
- Context keys: `inert_axes`, `diversification_status` (cell/target streaks + thresholds + directive), `experiment.budget_rounds`
- New constants: `DIVERSIFICATION_SOFT_THRESHOLD`, `DIVERSIFICATION_HARD_THRESHOLD`, `DIVERSIFICATION_ENFORCED`, `INERT_METRIC_EPSILON`, `BUDGET_ROUNDS_METADATA_KEY`

---

## Alternatives Considered

- **Purely soft prompt rules:** already proven insufficient in the audit experiment.
- **Full hard dedup on all axes globally:** too broad; blocks legitimate compound-axis exploration and does not account for different parent contexts.
- **Raising the old dry-run threshold:** would not address the target-level tunneling gap or the cell-flip loophole; treats the symptom, not the cause.

---

## Consequences

- Controller now enforces diversification and inert-knob invariants regardless of LLM behavior; prompt rules for these become informational rather than load-bearing.
- Custom programs (`custom_programs/`) are gitignored by design; the base `PROGRAM.md` prompt changes (F3, F4, enforcement documentation) are the tracked reference.
- Tests: 108 pass; 5 obsolete dry-run tests removed, 10 new tests added.
- Validation pending: a fresh 30-round run to confirm inert count ≤ 1, no cross-family target tunnel past the hard cap, and champion credited to its discovering phase.
