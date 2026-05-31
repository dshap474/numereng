# ADR: Agentic Research Native Ensemble Round Action

**Status:** Accepted
**Date:** 2026-05-31
**Commit:** 60a9b06

---

## Context

A 479-round validation experiment reached zero metric gain after the single-model search converged. The LLM continued probing config mutations but the primary metric (`bmc_last_200_eras_mean`) was flat across every new child run. The existing loop had no mechanism to escape single-model plateau and try blending the best scored candidates. Manual `numereng ensemble build` + `numereng run submit` after the fact was operationally viable but disconnected from the research loop's budget tracking, round ledger, and champion bookkeeping.

---

## Decision

Add a native `"ensemble"` LLM action to the agentic research controller, gated behind a plateau counter threshold, scored on the same primary metric as single runs, and tracked on a separate `best_ensemble` / `tried_ensembles` state track that never touches the single-model seed-trio confirmation/promotion machinery.

### Ensemble action schema

The LLM can return `action: "ensemble"` with:
- `ensemble_run_ids`: 2–8 existing scored run IDs belonging to the experiment.
- `ensemble_weights` (optional): per-member weights; omit for equal-weight rank average.
- No `parent_config` / `changes` fields — a blend has no parent config to mutate.

### Plateau gate

The ensemble action is offered to the LLM only once `phase_plateau_counter >= threshold`. Threshold is read from experiment metadata key `agentic_research_ensemble_plateau_threshold` (module default 40). For the `codex-exec` backend the gate is enforced by restricting the output-schema action enum to `"ensemble" | "run"` only after the threshold is crossed. For the OpenRouter backend (which cannot enforce schema-level enums at request time) the gate is re-checked in `_run_one_round` and a premature ensemble proposal is rejected as a validation error (`agentic_research_ensemble_locked`). (Distinct from the dedup *soft-skip*, which applies only to already-tried member sets.)

### Scoring and track separation

Ensemble blends are deterministic (rank-average, no seed trio), so each unique member set is scored exactly once via `score_prediction_file_with_details` — the same scorer used for single runs — with `bmc_last_200_eras_mean` as the primary metric. `features.ensemble.build_ensemble` handles the blend itself. Results accumulate in two new state fields:
- `state.best_ensemble`: the ensemble that achieved the highest primary metric so far.
- `state.tried_ensembles`: a dedup ledger of member-set signatures (sorted member IDs + rounded weights); an identical proposal is soft-skipped (cached score reused) without failure.

Ensemble results **never** enter the `confirmed_champion` / seed-trio promotion track. A blend has no meaningful seed-trio confirmation semantics and injecting it would corrupt the single-model promotion logic.

### Round accounting

- Ensemble rounds appear in `rounds/decision.json` with `round_type: "ensemble"`.
- The synthetic `run_id` written to the decision log is `ensemble:<ensemble_id>` to distinguish it from training runs that resolve under `runs/<id>/`.
- Ensemble rounds neither tick nor reset `phase_plateau_counter`. This preserves the plateau signal for future single-run diversification decisions.
- `best_ensemble` members are excluded from artifact rotation so their predictions stay on disk.

### Eligibility constraints

Member runs must be:
- `FINISHED` runs belonging to the experiment,
- have predictions still present on disk (same requirement as `serve` and `ensemble select`),
- share one feature set and dataset scope.

---

## Alternatives Considered

**Manual `numereng ensemble` CLI after the fact:** Viable for one-off blends but disconnected from the research loop's budget, round ledger, dedup tracking, and autonomous scheduling. The operator must monitor the run and intervene manually once plateau is detected.

**Route ensembles through the seed-trio champion slot:** Rejected. Seed-trio confirmation exists to separate noise from a genuinely improved config. A blend has no single parent config, no meaningful seed-42 discovery metric, and no `parent_config` / `changes` genealogy. Injecting an ensemble into the champion slot would corrupt the confirmation state machine and mix two fundamentally different selection tracks.

**Require equal-weight only:** Rejected as unnecessarily restrictive. The LLM may have evidence that some runs are stronger candidates. Optional weights are supported, but the default (and most common) path is equal-weight rank average.

---

## Consequences

- Single-model search and ensemble search are now first-class autonomous strategies within one research run; the loop can shift to blending when mutations plateau, then return to single-model exploration if the ensemble track also plateaus.
- Ensemble rounds do not appear as training runs in viz (no `runs/<ensemble_id>/` directory). The `ensemble:<ensemble_id>` synthetic ID in the decision log is a research artifact only.
- OpenRouter backend cannot enforce the plateau gate via JSON schema; the controller re-checks the counter and rejects a premature ensemble attempt as a validation error. This is the same pattern used for other OpenRouter-specific guardrails.
- Experiment metadata key `agentic_research_ensemble_plateau_threshold` is the operator-facing knob to tune how many plateau rounds are required before ensemble rounds are offered.
- `best_ensemble` members are pinned from artifact rotation for the life of the research run, so long runs with many ensemble attempts may accumulate more prediction artifacts on disk than single-model-only runs of the same depth.
