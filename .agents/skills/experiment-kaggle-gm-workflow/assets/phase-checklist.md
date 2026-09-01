# Phase Transition Checklist

Use these checklists to verify gate criteria before advancing to the next phase.

## Phase 1 -> Phase 2 (EDA to Diversity)

- [ ] Data downloaded and verified (correct version, no stale data)
- [ ] Baseline LGBM trained on `target_ender_20`
- [ ] Baseline Sharpe > 0.3 on validation
- [ ] No data quality issues (NaN rates normal, era coverage complete)
- [ ] Floor metrics recorded in EXPERIMENT.md
- [ ] Experiment created and status set to active

## Phase 2 -> Phase 3 (Diversity to HPO)

- [ ] 8+ models trained with Sharpe > 0.3
- [ ] At least 3 algorithm types represented (e.g., LGBM + XGBoost + Ridge)
- [ ] At least 3 different targets represented
- [ ] Pairwise correlation matrix reviewed
- [ ] No redundant pairs (correlation > 0.85) kept
- [ ] At least 4 models with pairwise correlation < 0.70
- [ ] Scout results documented in EXPERIMENT.md

## Phase 3 -> Phase 4 (HPO to Seed Ensembling)

- [ ] Top 3-5 models selected for HPO (or HPO skipped with justification)
- [ ] Optuna studies completed (25-50 trials each)
- [ ] Best configs exported
- [ ] Improvement > 5% confirmed on at least one model (or skip documented)
- [ ] Tuned configs validated on full data (if applicable)
- [ ] HPO results documented in EXPERIMENT.md

## Phase 4 -> Phase 5 (Seed Ensembling to Ensemble Construction)

- [ ] All seed variants trained (5 per GBDT, 3 per linear)
- [ ] CV(Sharpe) < 0.30 for each model config
- [ ] Seed averages created for each model config
- [ ] Seed averages show lower variance than individual seeds
- [ ] Seed variance documented in EXPERIMENT.md

## Phase 5 -> Phase 6 (Ensemble to Post-Processing)

- [ ] Correlation scan across all candidates completed
- [ ] Forward selection run on payout metric
- [ ] Holdout evaluation completed
- [ ] Hillclimb-to-holdout degradation < 60%
- [ ] Equal-weight sanity check performed
- [ ] 3+ effective models in final ensemble (no single model > 60% weight)
- [ ] Ensemble Sharpe > 0.5 on holdout
- [ ] Final ensemble choice documented in EXPERIMENT.md

## Phase 6 -> Phase 7 (Post-Processing to Submission)

- [ ] Predictions rank-normalized to [0, 1]
- [ ] Neutralization sweep completed ([0.0, 0.3, 0.5, 0.7])
- [ ] Neutralization proportion selected and justified
- [ ] Feature exposure < 0.10 after neutralization
- [ ] Prediction distribution validated (approximately uniform)
- [ ] No missing predictions (all stock IDs present)
- [ ] Post-processing config documented in EXPERIMENT.md

## Phase 7 -> Submit (Validation to Live)

- [ ] Full holdout evaluation with extended metrics
- [ ] Overfitting diagnostics reviewed
- [ ] Benchmark comparison documented
- [ ] Submission source validated (`--run-id` xor `--predictions`)
- [ ] **User confirmed live submission** (mandatory)
- [ ] Live submission completed and accepted
- [ ] Monitoring plan documented
- [ ] Experiment concluded with verdict and notes
