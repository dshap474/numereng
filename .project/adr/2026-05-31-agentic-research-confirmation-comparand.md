# ADR: Agentic Research Confirmation Comparand and Promotion Margin

**Status:** Accepted
**Date:** 2026-05-31

---

## Context

In the `2026-05-27_wide-diversification-test` run (521 rounds), only 4 champion promotions occurred — all by round 42 — and **zero in the final 479 rounds**. A programmatic pass over `state.json` showed this was not an LLM failure but a structurally unreachable gate.

The seed-averaged confirmation protocol ranks a challenger by its **3-seed trio mean** but had two flaws in how a challenger was selected for confirmation and promoted:

1. **Wrong entry comparand.** A single-seed run was queued as a confirmation candidate only when it beat the **champion's seed-42 score** (the champion's *luckiest* single seed) by more than `3e-4`. With champion `config_040` at seed-42 = 0.0039335, that bar was `0.0042335`. The highest score *any* config reached in the entire run was 0.0039713 (r232) — **0.00026 below the bar**. At least 13 distinct configs reached seed-42 ≈ 0.003971 (above the champion's seed-42 *and* far above the champion's trio mean of 0.0035387), yet not one was ever confirmed: the LLM correctly followed a rule whose bar the search space could not reach.

2. **Absolute margin on a near-saturated surface.** Promotion required the challenger trio mean to exceed the champion trio mean by an absolute `3e-4`. But `3e-4` is the *single-seed* noise floor; a 3-seed mean already suppresses that noise (standard error ≈ `3e-4 / √3 ≈ 1.7e-4`). Requiring a full single-seed margin on top of trio-averaging double-counts the noise guard and makes promotion unreachable once scores saturate.

The relevant gates: the entry comparand lived in `PROGRAM.md` (the LLM initiates confirmations; the controller does not hard-block entry) and was mirrored in `_has_inflight_confirmation` for phase-transition deferral. The promotion margin is hard-enforced in `_maybe_promote_confirmation`.

---

## Decision

**Entry comparand → champion trio mean.** A single-seed run becomes a confirmation candidate when it beats the champion's confirmed **trio mean** (`confirmed_champion.seed_trio_primary_mean`), not the champion's luckiest single seed. The challenger's own trio mean is what it will be promoted against, so the trio mean is the fair "worth confirming" bar. A single seed above the trio bar is a genuine signal worth the 2-round trio test; the trio then averages out the single-seed noise. Implemented in both program prompts (`PROGRAM.md`, `custom_programs/wide_search_v1.md`) and mirrored in `_has_inflight_confirmation` via a new `_champion_trio_mean_metric` helper, so the phase-deferral recognition and the prompt rule agree.

**Promotion margin → trio-mean standard error.** Renamed `CONFIRMATION_PROMOTION_THRESHOLD = 3e-4` to `CONFIRMATION_PROMOTION_MARGIN = 1.5e-4`. A challenger's trio mean must exceed the champion's trio mean by more than `1.5e-4` (≈ single-seed noise / √3) to promote. The comparand was already correct (trio vs trio); only the scale changed, from the single-seed floor to the honest trio-mean standard error. The gate stays hard-enforced in `_maybe_promote_confirmation` — promotion is noise-vs-signal enforcement and is not delegated to the LLM.

`PHASE_IMPROVEMENT_THRESHOLD = 3e-4` is unchanged (separate concern).

---

## Alternatives Considered

**Strict (margin → tiny epsilon):** promote whenever the challenger trio mean exceeds the champion's at all. Maximally reachable, and the trio averaging is itself the noise control — but it risks champion churn (flip-flopping between near-equal configs). Rejected in favor of a small margin that still guards against churn.

**Relative margin (fraction of the champion value):** adapts automatically as scores saturate. Rejected as a less intuitive operator-facing knob; the trio-mean standard error is a concrete, defensible absolute scale and `1.5e-4` is reachable at the observed score levels (~0.0035–0.0040).

**Trigger on "beats champion seed-42 at all":** an alternative entry rule from the analysis. Rejected because seed-42 is one noisy sample of the champion; the trio mean is the stable, fair comparand and matches what the challenger is ultimately promoted against.

**Leave the gate, fix only the prompt:** the entry rule is prompt-only, but the promotion margin is enforced in code, so a prompt-only change would leave the code gate at `3e-4`. Both layers were fixed so they agree.

---

## Consequences

- Legitimate challengers (a single seed above the champion's trio mean) now enter confirmation and can promote. The 13+ configs that were silently ignored in the prior run would now each be tested.
- More confirmation rounds are spent (2 extra rounds per candidate), bounded by the existing "one unconfirmed candidate in flight at a time" backlog cap. This is the intended cost — testing real candidates — versus the prior 479 dead rounds.
- A challenger whose lucky seed-42 cleared the trio bar but whose full trio regresses is correctly recorded as confirmed-but-not-champion: the trio is doing its noise-rejection job.
- The promotion gate remains controller-enforced (unlike ensemble *timing*, which moved to the LLM); only its threshold value and the entry comparand changed.
- Regression test `test_confirmation_promotes_above_trio_margin_below_old_threshold` locks the new behavior: a trio improvement between `1.5e-4` and `3e-4` now promotes (the old gate rejected it).
