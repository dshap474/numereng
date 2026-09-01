> **Compatibility Note (Current numereng CLI):**
> This phase may include legacy command examples (for example `orchestrator`, `optimize`, `ensemble`, `compare`, `neutralize-sweep`, `experiment summarize`, `experiment conclude`).
> Treat legacy commands and `.yaml` config examples as historical reference only.
> Execute this phase through `experiment-design` using current commands (`experiment create|train|report|promote`, `run train`, `run submit`) and JSON configs.

# Phase 5: Ensemble Construction

**Goal:** Combine diverse seed-averaged models into a final blended prediction.

## Prerequisites

- Phase 4 complete: seed-averaged models available
- At least 4 diverse model candidates with pairwise correlation < 0.85

## Steps

### 5.1 Prepare Candidates

List all seed-averaged models (and any strong singles) as ensemble candidates:

```bash
uv run numereng experiment summarize --id gm-campaign-<NNN> --metric sharpe
```

Filter to candidates with:
- Sharpe > 0.3 on validation
- Unique contribution (not redundant with a better model)

### 5.2 Correlation Scan

```bash
uv run numereng ensemble correlations --runs <candidate1,candidate2,...>
```

Review the correlation matrix:
- Pairs with correlation > 0.85: keep the better one, drop the other
- Ideal ensemble: 4-8 models with mean pairwise correlation 0.40-0.70
- Always include Ridge (typically 0.12-0.21 correlation with tree models)

### 5.3 Forward Selection (Primary Method)

Run forward selection on the payout metric using hillclimb eras:

```bash
uv run numereng ensemble build \
  --runs <candidates> \
  --method forward \
  --metric payout
```

Or via experiment integration:

```bash
uv run numereng experiment build-ensemble \
  --id gm-campaign-<NNN> \
  --method forward_selection \
  --metric payout
```

### 5.4 Evaluate on Holdout

Evaluate the forward-selected ensemble on holdout eras:

```bash
uv run numereng ensemble evaluate \
  --runs <selected_models> \
  --weights <w1,w2,w3,...>
```

Check the hillclimb-to-holdout degradation:
- degradation = 1 - (holdout_sharpe / hillclimb_sharpe)
- **< 40%:** Good generalization
- **40-60%:** Acceptable, monitor in live
- **> 60%:** Overfitting risk — use equal-weight blend instead

### 5.5 Equal-Weight Sanity Check

Always compute the equal-weight blend for comparison:

```bash
uv run numereng ensemble evaluate \
  --runs <selected_models> \
  --weights equal
```

If equal-weight holdout Sharpe is within 10% of forward-selected holdout Sharpe, prefer equal-weight (more robust out of sample).

### 5.6 Stacking (Optional)

Consider stacking only if:
- Forward selection puts > 50% weight on a single model
- You have 5+ diverse base models
- Budget allows additional compute

Stacking uses a Ridge meta-learner trained on OOF predictions:

```bash
uv run numereng orchestrator run \
  --config .numereng/experiments/gm-campaign-<NNN>/configs/stacking.yaml \
  --local
```

See `assets/stacking-config.yaml` for the starter config. The stacking config maps to the `StackingConfig` dataclass in `src/numereng/ensemble/stacking.py`:

```yaml
model:
  type: stacking
  params:
    level1_models:
      - type: lgbm
        params: { ... }  # Best LGBM config
      - type: xgboost
        params: { ... }  # Best XGBoost config
      - type: catboost
        params: { ... }  # Best CatBoost config
      - type: ridge
        params: { alpha: 1.0 }
    per_model_targets:
      0: target_ender_20
      1: target_cyrusd_20
      2: target_teager2b_20
      3: target_ender_20
    level2_model:
      type: ridge
      params: { alpha: 0.1 }
    n_folds: 5
    embargo_eras: 8
    per_era_rank: true
```

### 5.7 Final Ensemble Selection

Compare all candidates:
1. Forward-selected blend (optimized weights)
2. Equal-weight blend (robust baseline)
3. Stacking ensemble (if computed)

Select the one with best holdout performance AND acceptable degradation.

## Gate Criteria

- [ ] Correlation scan completed
- [ ] Forward selection run on payout metric
- [ ] Holdout evaluation completed
- [ ] Hillclimb-to-holdout degradation < 60%
- [ ] Equal-weight sanity check performed
- [ ] 3+ effective models in final ensemble (no single model > 60% weight)
- [ ] Ensemble Sharpe > 0.5 on holdout
- [ ] Stacking evaluated if weight concentration > 50% (optional)
- [ ] Final ensemble choice documented in EXPERIMENT.md

## Common Issues

- **Forward selection picks only 1-2 models:** Low diversity in candidates. Consider going back to Phase 2 for more targets/algorithms.
- **Degradation > 60%:** Use equal-weight blend. Alternatively, use fewer iterations in forward selection.
- **Stacking underperforms forward selection:** Normal for small candidate pools. Stick with forward selection.
- **Ridge excluded by forward selection:** Force include Ridge at 10-15% weight for diversity insurance.

## Compute Estimate

| Task | Backend | Duration | Cost |
|------|---------|----------|------|
| Correlation scan + forward selection | Local | ~30 min | Free |
| Equal-weight evaluation | Local | ~15 min | Free |
| Stacking (optional, 5-fold OOF) | Local/Cloud | ~1-2 hrs | $0-3 |
