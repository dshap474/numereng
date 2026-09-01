---
name: experiment-kaggle-gm-workflow
description: End-to-end Kaggle Grandmaster workflow for building tournament-ready Numerai models. 7-phase process from EDA through submission.
user-invocable: true
---

# Kaggle GM Workflow

Opinionated 7-phase playbook mapping the NVIDIA Kaggle Grandmaster Playbook onto the numereng codebase. Prescribes _what to build, in what order, and when to stop_.

Delegates execution to `experiment-design` (experiment lifecycle) and `numerai-model-upload` (submission). This skill adds the strategic layer.

Compatibility note:
- The current numereng CLI in this repo does not expose `orchestrator`, `optimize`, or `ensemble` command families.
- Training/HPO configs are JSON-only in the current runtime (`.json`); treat `.yaml` assets in this skill as reference templates to translate.
- Execute phases with `experiment-design` using current commands (`experiment create|train|report|promote`, `run train`, `run submit`) plus manual sweep/blend workflows where needed.

Run from:
- `<repo>`

## Reference Loading Guide

**Read the matching reference file BEFORE executing that phase.** This SKILL.md is a routing index.

| Phase | When to load | Reference file |
|-------|-------------|----------------|
| 1 - EDA & Baseline | Starting a new campaign | `references/phase1-eda-baseline.md` |
| 2 - Diversity Campaign | Training diverse model pool | `references/phase2-diversity-campaign.md` |
| 3 - HPO Refinement | Tuning top performers | `references/phase3-hpo-refinement.md` |
| 4 - Seed Ensembling | Reducing variance | `references/phase4-seed-ensembling.md` |
| 5 - Ensemble Construction | Building final blend | `references/phase5-ensemble-construction.md` |
| 6 - Post-Processing | Neutralization & calibration | `references/phase6-postprocessing.md` |
| 7 - Validation & Submission | Final eval & deployment | `references/phase7-validation-submission.md` |
| Cost planning | Any phase | `references/cost-estimator.md` |
| Avoiding mistakes | Any phase | `references/anti-patterns.md` |

If a task spans phases, load all relevant references.

## 7-Phase Overview

```
Phase 1: EDA & Baseline          ~1 hour    | local      | free
Phase 2: Diversity Campaign       ~4-8 hrs  | cloud      | $5-15
Phase 3: HPO Refinement           ~2-4 hrs  | cloud      | $3-10
Phase 4: Seed Ensembling          ~2-6 hrs  | cloud      | $3-12
Phase 5: Ensemble Construction    ~1-2 hrs  | local      | free
Phase 6: Post-Processing          ~30 min   | local      | free
Phase 7: Validation & Submission  ~30 min   | local      | free
                                  ─────────────────────────────
                                  Total: ~12-24 hrs, $10-40
```

### Phase 1: EDA & Baseline

**Goal:** Establish floor metrics with a single LightGBM on payout target.

- Download data, verify freshness
- Train single LGBM on `target_ender_20` using scout data (`data.dataset_variant: downsampled`)
- Record baseline Sharpe, corr_mean, max_drawdown
- Create experiment via `experiment-design` skill

**Gate:** Baseline Sharpe > 0.3, no data issues. Proceed to Phase 2.

### Phase 2: Diversity Campaign

**Goal:** Train 12-20+ models across algorithm, target, and feature axes.

- 3+ algorithms: LightGBM, XGBoost, CatBoost, Ridge
- 4+ targets: ender_20, cyrusd_20, teager2b_20, xerxes_20, etc.
- Feature set: medium (default) or all
- Scout on downsampled data (`data.dataset_variant: downsampled`), then scale winners on full data (`non_downsampled`)
- Correlation analysis to verify diversity (pairwise < 0.85)

**Gate:** 8+ models with Sharpe > 0.3, at least 3 algorithm types, pairwise correlation matrix shows diversity. Proceed to Phase 3.

### Phase 3: HPO Refinement

**Goal:** Tune top 3-5 performers with manual config sweeps (current CLI contract).

- 25-50 trials per model on downsampled scout data (`training.engine.profile: purged_walk_forward`)
- Per-algorithm search spaces (see `assets/hpo-study-*.yaml`)
- Keep `colsample_bytree=0.1` FIXED
- Stop early if best trial < 5% improvement over default

**Gate:** Top performers re-evaluated on full data, improvement confirmed. Proceed to Phase 4.

### Phase 4: Seed Ensembling

**Goal:** Reduce variance via multi-seed averaging.

- 5 seeds for GBDTs (42, 123, 456, 789, 1011)
- 3 seeds for linear models (42, 123, 456)
- Per-seed rank-normalize, arithmetic mean, final rank
- CV(Sharpe) across seeds < 0.30

**Gate:** Seed-averaged models show lower variance than singles. Proceed to Phase 5.

### Phase 5: Ensemble Construction

**Goal:** Combine diverse seed-averaged models into final blend.

- Correlation scan across all candidates
- Forward selection on payout metric (hillclimb eras)
- Evaluate on holdout eras
- Optional: stacking (Ridge meta-learner on OOF) if forward selection shows >50% weight concentration
- Overfitting guardrail: hillclimb-to-holdout degradation < 60%

**Gate:** Ensemble Sharpe > 0.5 on holdout, degradation < 60%, 3+ effective models. Proceed to Phase 6.

### Phase 6: Post-Processing

**Goal:** Apply neutralization and validate prediction quality.

- Rank normalization (automatic in engine)
- Neutralization sweep: [0.0, 0.3, 0.5, 0.7]
- Feature exposure check: max < 0.10
- Prediction distribution: verify uniform-ish on [0, 1]

**Gate:** Feature exposure < 0.10, predictions pass distribution check. Proceed to Phase 7.

### Phase 7: Validation & Submission

**Goal:** Final holdout evaluation and live deployment.

- Full holdout evaluation with extended metrics
- Overfitting diagnostics (hillclimb vs holdout vs full)
- Package/submit predictions via `numerai-model-upload` skill (or `official-numerai-ops` for direct official API workflows)
- Validate submission source/readiness, then run live submission (requires user confirmation)
- Post-submission monitoring plan

**Gate:** Live submission confirmed, monitoring plan in place.

## Decision Matrix

| Situation | Action |
|-----------|--------|
| Baseline Sharpe < 0.3 | Check data/config issues before proceeding |
| Few models beat baseline | Expand target diversity first, then algorithms |
| HPO gives < 5% improvement | Skip HPO, move to seed ensembling |
| CV(Sharpe) across seeds already < 0.15 | Use 3 seeds instead of 5 |
| Forward selection puts >50% on one model | Consider stacking as alternative |
| Hillclimb-to-holdout degradation > 60% | Use equal-weight blend instead |
| Feature exposure > 0.10 after neutralization | Increase neutralization proportion |
| Budget constrained (< $10) | Skip Phase 3 HPO, use 3 seeds, scout/downsampled-only |

## Numerai Constants (Quick Reference)

| Constant | Value |
|----------|-------|
| Payout target | `target_ender_20` |
| Payout formula | `0.75 * CORR + 2.25 * BMC` |
| Feature set sizes | small (~42), medium (~700), all (~2,376) |
| Purge gap (20-day targets) | 8 eras |
| Purge gap (60-day targets) | 16 eras |
| colsample_bytree | 0.1 (FIXED, never tune) |
| Rank predictions | Always rank to [0, 1] before submission |

## Campaign Lifecycle

A "campaign" is one full pass through all 7 phases. Use the `experiment-design` skill to manage the experiment lifecycle:

```bash
# Create campaign experiment
uv run numereng experiment create --id 2026-02-22_gm-campaign-001 --name "GM Campaign 1" \
  --hypothesis "Full GM workflow on payout target" --tags "gm-workflow,campaign"

# Use the campaign template for logging
# Copy assets/gm-campaign-template.md to .numereng/experiments/2026-02-22_gm-campaign-001/EXPERIMENT.md
```

Track phase transitions in the experiment's EXPERIMENT.md using the phase checklist (`assets/phase-checklist.md`).

## Related Skills

| Skill | Delegation |
|-------|-----------|
| `experiment-design` | Experiment create/train/report/promote commands plus manual tuning and blend strategy |
| `numerai-model-upload` | Numereng submission flow (run artifact or predictions file) |
| `official-numerai-ops` | Official MCP/GraphQL/NumerAPI workflows (diagnostics/model ops) |
| `utility-store-ops` | Store/DB operations if needed |
| `numerai-model-implementation` | Only if adding a new model type (rare) |

## Assets

| Asset | Purpose |
|-------|---------|
| `assets/gm-campaign-template.md` | Extended EXPERIMENT.md for campaigns |
| `assets/diversity-matrix.yaml` | Target x Algorithm x Feature model catalog |
| `assets/hpo-study-lgbm.yaml` | Starter HPO config: LightGBM |
| `assets/hpo-study-xgboost.yaml` | Starter HPO config: XGBoost |
| `assets/hpo-study-catboost.yaml` | Starter HPO config: CatBoost |
| `assets/stacking-config.yaml` | Starter stacking ensemble config |
| `assets/phase-checklist.md` | Gate checklist per phase transition |
