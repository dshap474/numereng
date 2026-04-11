---
name: experiment-design
description: "Design and manage numereng experiments for any model idea, including round planning, scout-to-scale decisions, plateau logic, reporting, and champion handoff."
user-invocable: true
---

# Experiment Design
Use this workflow to plan, run, and report numereng experiments for any model idea.

Note: run commands from `<workspace>` with `numereng ...`.

Use `numereng-experiment-ops` for numereng-specific experiment layout, config templates, schema
questions, `EXPERIMENT.md` formatting, and run artifact expectations. Use `store-ops` for drift,
reindex, reset, or cleanup. Use `implement-custom-model` when the model type is new. Use
`numereng-experiment-ops` after a champion is finalized and submission handoff is needed.

## Use when
- the user wants experiment strategy, round design, scout-to-scale decisions, plateau logic, or
  champion selection
- the user wants help deciding what configs to run next or how to interpret completed rounds
- the user wants experiment reporting guidance after one or more numereng rounds are complete

## Do not use when
- the user needs experiment folder, manifest, schema, or template rules; use
  `numereng-experiment-ops`
- the user needs drift diagnosis, reindex, reset, or cleanup; use `store-ops`
- the user needs a new custom model type implemented; use `implement-custom-model`
- the user needs the concrete numereng submission handoff after champion selection; use `numereng-experiment-ops`

## Persistence expectation (required)

This skill is *not* complete after a single promising run. You must run experiments in **rounds**
(typically **4-5 configs per round**), synthesize results, and decide what to try next. Only
finalize when you reach a plateau and additional rounds stop improving the primary metric.

## Planning checklist (answer before running)
- State the model idea and novelty.
- Choose the initial baseline and feature scope. Keep experiments aligned with the chosen
  baseline unless the round is explicitly about changing it.
- Decide the primary metric:
  - `bmc_last_200_eras.mean`
- Decide the tie-break metric:
  - `bmc.mean`
- Decide which parameter dimensions to explore based on the core idea:
  - targets
  - model hyperparameters
  - ensemble weights
  - data settings
- Or decide that only a minimal round is needed because the change is tiny, but still run
  multiple variants unless the user explicitly requested exactly one run.

## Handling ambiguity (fast disambiguation)
If the user's request is unclear or underspecified:
1. List 2-4 plausible interpretations and keep them meaningfully different.
2. Implement quick scout runs for each interpretation using conservative compute.
3. Compare `bmc.mean` and `bmc_last_200_eras.mean`.
4. Use the best-BMC interpretation going forward, and document the choice and rationale in
   `EXPERIMENT.md`.

## Workflow
Core loop (repeat for each experiment round):
1. If the model type is new, implement it with the `implement-custom-model` skill.
2. Create or update **4-5 configs** for the current round:
   - one base
   - single-variable variants
3. Run training for each config through numereng experiment commands.
4. Wait for the whole round to finish, then synthesize results:
   - pick the current best by `bmc_last_200_eras.mean` (primary), with `bmc.mean` as a tie-breaker
   - sanity-check `corr.mean`, `mmc.mean`, and `cwmm.mean`
   - check stability and whether the improvement is consistent across eras
5. Update `EXPERIMENT.md` with:
   - what changed this round
   - the metrics table
   - the next-round decision
6. Repeat rounds until a plateau is reached (see "When to stop" below), then scale the winner.

## Scout -> Scale
1. **Use lower-cost scouts first**: Prefer the scout-friendly dataset variant and capacity choices
   defined by `numereng-experiment-ops` to save time and compute while exploring.
2. **Pick the sweep dimension that matches the core idea**: Run a focused sweep only when it
   serves the research question; otherwise run a single experiment config and evaluate.
3. **Iterate until improvements stop**: Keep sweeping on that dimension while a round produces a
   new best metric. If a round does not improve, reassess or pivot.
4. **Focus when a parameter dominates**: If one parameter clearly drives results, dedicate a full
   round to mapping its range while holding others fixed.
5. **Scale only winners**: Once a best option is determined in the scout phase, move to a scaled
   round where you increase feature scope, data scope, or model capacity only for top candidates.
6. **Run one scaled final**: Run the top config in a scaled setting and record the final metrics
   before you stop finding improvements.

## When to stop (plateau criteria)

Stop iterating only when **at least two consecutive rounds** fail to beat the current best
`bmc_last_200_eras.mean` by a meaningful margin (rule of thumb: ~`1e-4`-`3e-4`), *and* the
remaining untried knobs are either redundant with what you already swept or likely to increase
overfit or benchmark-correlation.

If you plateau on scout settings, do *one* confirmatory scale step before concluding the idea is
maxed out.

## Sweep selection by research type
Note that these are examples only. Each idea will call for different sweeps, or no sweeps. These
are guidelines, but use judgment to determine the best experiments to run to answer the core
question of whether this idea can produce a model with high BMC.
- **New target, label, or feature engineering**: Sweep target variants or preprocessing settings;
  skip broad hyperparameter sweeps unless performance is unstable.
- **New model architecture**: Run a hyperparameter sweep over capacity, learning rate,
  regularization, and related controls.
- **Ensemble, blend, or stacking idea**: Sweep combination weights, blend rules, or stacker
  settings.
- **Training-procedure change**: Sweep procedure-specific parameters such as profile, loading mode,
  or prediction-stage transforms.
- **Data change**: Sweep dataset variant, feature scope, or other data settings.

## Sweep design guidance
- Use one-variable-at-a-time changes for each run in the chosen sweep dimension.
- Build a base config per round, then create variants that change a single parameter or variant.
- Take time to design each round based on last-round results, model type, and known sensitivities.
- If scaling depth, width, leaves, estimators, or related parameters, consider whether learning
  rate or regularization should move with them.
- Track and compare per-round results; keep the best model and document why it won.

## Baseline alignment
- Declare which baseline the model is aiming to improve on.
- Keep comparisons aligned with the chosen baseline for fair comparisons.
- Default to the repo's standard benchmark reference and report against the same baseline across
  the experiment unless the round is intentionally testing a baseline change.

## Experiment organization
- Keep related runs under a single, well-named folder in `experiments/`.
- One experiment folder = one line of inquiry.
  - `configs/` for configs
  - `EXPERIMENT.md` for summary and decisions
  - `run_plan.csv` only when ordering a planned sweep
- Include a **baseline row** in result tables for comparisons.
- Name configs to reflect the single variable change.
- Use `numereng-experiment-ops` for the exact path, template, and manifest contract.

## Reporting expectations
- Run experiments in **rounds** and wait for the round to finish so you do not report prematurely.
- Once you complete the research and stop finding improvements, write a report for the user. It
  should describe learnings, what worked and what did not, and include the final stats table.
- Always report:
  - `bmc.mean`
  - `bmc_last_200_eras.mean`
  - `corr.mean`
  - `mmc.mean`
  - `cwmm.mean`
- Use consistent markdown tables and update `EXPERIMENT.md` after each round.
- Include a cohesive plan and story, finishing with a final result that combines learnings from all
  experiments.

## Dataset handling
- Prefer the scout-friendly dataset variant and contract defaults from `numereng-experiment-ops`
  for quick iteration.
- Only scale dataset scope or feature scope after a clear signal from earlier rounds.
- If you override the default dataset variant for cost or debugging reasons, record the reason in
  `EXPERIMENT.md`.

## Useful entry points
- `numereng experiment create|details|train|report|promote`
- `numereng ensemble build|list|details`
- `numereng hpo create`
- `numereng run train`
- `numereng store doctor`

## Deployment (after experiments complete)
Once you have finalized your best model and have a submission-ready run artifact or predictions
source using the `numereng-experiment-ops` skill:

1. **Offer deployment**: Ask the user if they want to submit the champion through numereng.
2. **Deployment options**:
   - submit from a live-eligible winning run artifact
   - submit from an explicit predictions file
3. **Follow the `numereng-experiment-ops` skill** for the final refit and submission workflow.

This allows the full research-to-submission workflow to happen in a single session.

## Output Contract

Return:
- the current experiment question being answered
- the recommended next round, pivot, scale step, or stop decision
- the current winner and why it leads
- the key metrics used to justify the recommendation
- any explicit risk or uncertainty that should affect the next decision

## Reference Loading Guide

Load these references only when the task needs deeper guidance beyond the base workflow.

| When the task involves... | Load this reference |
|---|---|
| Hypothesis framing, round planning, stop criteria, reporting | `references/research-strategy.md` |
| Manual sweeps and config-variant comparisons | `references/tuning-and-optimization.md` |
| Prediction-stage neutralization decisions | `references/feature-neutralization.md` |
| Blend strategy, candidate selection, and weight choices | `references/ensemble-building.md` |
| Seed variance reduction strategy | `references/seed-ensembling.md` |

If a task spans multiple domains, load each relevant reference and avoid unrelated files.
