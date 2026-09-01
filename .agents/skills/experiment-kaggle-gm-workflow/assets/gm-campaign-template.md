# GM Campaign: <CAMPAIGN_ID>

**Created:** <DATE>
**Status:** active
**Hypothesis:** Full GM workflow — diverse models, HPO, seed ensembling, blend on payout target

## Phase 1: EDA & Baseline

**Baseline Run:** `<run_id>`
**Date:** <DATE>

| Metric | Value |
|--------|-------|
| Sharpe | |
| corr_mean | |
| max_drawdown | |
| feature_exposure | |

**Notes:**

**Gate:** [ ] Sharpe > 0.3, no data issues

---

## Phase 2: Diversity Campaign

**Date Started:** <DATE>

### Scout Results (Downsampled)

| Run ID | Algorithm | Target | Sharpe | corr_mean | Status |
|--------|-----------|--------|--------|-----------|--------|
| | lgbm | ender_20 | | | |
| | lgbm | cyrusd_20 | | | |
| | xgboost | ender_20 | | | |
| | catboost | teager2b_20 | | | |
| | ridge | ender_20 | | | |

### Correlation Matrix Summary

| Model A | Model B | Correlation |
|---------|---------|-------------|
| | | |

### Scaled to Full Data

| Run ID | Sharpe (scout/downsampled) | Sharpe (full/non_downsampled) | Status |
|--------|---------------|---------------|--------|
| | | | |

**Gate:** [ ] 8+ models, 3+ algorithms, correlation diversity confirmed

---

## Phase 3: HPO Refinement

**Date Started:** <DATE>

### HPO Studies

| Study Name | Algorithm | Target | Trials | Best Sharpe | Default Sharpe | Improvement |
|------------|-----------|--------|--------|-------------|----------------|-------------|
| | | | | | | |

### Key Parameter Findings

| Algorithm | Important Parameters | Optimal Range |
|-----------|---------------------|---------------|
| | | |

**Gate:** [ ] Improvement > 5% confirmed (or skip justification)

---

## Phase 4: Seed Ensembling

**Date Started:** <DATE>

### Seed Variance

| Model Config | Seed 42 | Seed 123 | Seed 456 | Seed 789 | Seed 1011 | CV(Sharpe) |
|-------------|---------|----------|----------|----------|-----------|------------|
| | | | | | | |

### Seed Averages

| Model Config | Avg Sharpe | Single Best | Improvement |
|-------------|-----------|-------------|-------------|
| | | | |

**Gate:** [ ] CV < 0.30, seed averages show lower variance

---

## Phase 5: Ensemble Construction

**Date Started:** <DATE>

### Forward Selection Results

| Step | Model Added | Weight | Cumulative Sharpe (Hillclimb) |
|------|------------|--------|-------------------------------|
| 1 | | | |
| 2 | | | |
| 3 | | | |

### Holdout Evaluation

| Blend | Hillclimb Sharpe | Holdout Sharpe | Degradation |
|-------|-----------------|----------------|-------------|
| Forward-selected | | | |
| Equal-weight | | | |
| Stacking (optional) | | | |

**Selected Blend:** <description>

**Gate:** [ ] Holdout Sharpe > 0.5, degradation < 60%, 3+ effective models

---

## Phase 6: Post-Processing

**Date:** <DATE>

### Neutralization Sweep

| Proportion | Sharpe | Feature Exposure | MMC |
|------------|--------|-----------------|-----|
| 0.0 | | | |
| 0.3 | | | |
| 0.5 | | | |
| 0.7 | | | |

**Selected Proportion:** <value>
**Justification:** <reason>

**Gate:** [ ] Feature exposure < 0.10, distribution valid

---

## Phase 7: Validation & Submission

**Date:** <DATE>

### Final Metrics

| Metric | Hillclimb | Holdout | Full Val |
|--------|-----------|---------|----------|
| Sharpe | | | |
| corr_mean | | | |
| BMC mean | | | |
| Payout score | | | |
| max_drawdown | | | |
| feature_exposure | | | |

### Submission

- **Model Name:** <numerai_model_name>
- **Round:** <round_number>
- **Submission Date:** <date>
- **Status:** <submitted/accepted/rejected>

### Monitoring Plan

- Weekly CORR/BMC check
- Alert if CORR < 0.005 or BMC < 0 for 3+ rounds
- Re-evaluate after 50 rounds

---

## Conclusion

**Verdict:** <success/failure/inconclusive>
**Final Ensemble:** <description>
**Key Learnings:**
-
-
-
