> **Compatibility Note (Current numereng CLI):**
> This phase may include legacy command examples (for example `orchestrator`, `optimize`, `ensemble`, `compare`, `neutralize-sweep`, `experiment summarize`, `experiment conclude`).
> Treat legacy commands and `.yaml` config examples as historical reference only.
> Execute this phase through `experiment-design` using current commands (`experiment create|train|report|promote`, `run train`, `run submit`) and JSON configs.

# Phase 2: Diversity Campaign

**Goal:** Train 12-20+ models across three diversity axes: algorithm, target, and feature set.

## Prerequisites

- Phase 1 complete: baseline Sharpe > 0.3
- Cloud backend configured (RunPod recommended): `uv run numereng runpod setup`

## Diversity Axes

1. **Algorithm:** LightGBM, XGBoost, CatBoost, Ridge
2. **Target:** target_ender_20, target_cyrusd_20, target_teager2b_20, + others
3. **Feature set:** medium (default), all (for select models)

See `assets/diversity-matrix.yaml` for the full model catalog with tiered priorities.

## Steps

### 2.1 Scout Phase (Downsampled Scout)

Train Tier 1 models (8 must-have) first on downsampled scout data to validate configs before scaling.

For each model in the diversity matrix:

```bash
uv run numereng orchestrator run \
  --config .numereng/experiments/gm-campaign-<NNN>/configs/<model_id>.yaml \
  --local
```

Or batch via cloud for faster iteration:

```bash
# RunPod (sequential - one at a time to avoid parallel training)
uv run numereng orchestrator run \
  --config .numereng/experiments/gm-campaign-<NNN>/configs/<model_id>.yaml \
  --tier rtx4090
```

### 2.2 Evaluate Scouts

After all scouts complete:

```bash
uv run numereng experiment summarize --id gm-campaign-<NNN> --metric sharpe
```

Filter to models with Sharpe > 0.3. Drop obvious failures.

### 2.3 Correlation Analysis

Check pairwise prediction correlations:

```bash
uv run numereng ensemble correlations --runs <run1,run2,run3,...>
```

**Key threshold:** Pairwise correlation < 0.85. Models with correlation > 0.85 are redundant — keep the better performer and drop the other.

What counts as diversity:
- Same algorithm, different target: typically 0.40-0.70 correlation (good)
- Different algorithm, same target: typically 0.70-0.85 correlation (moderate)
- Same algorithm, same target: typically 0.90+ correlation (redundant)

### 2.4 Scale Winners

Promote top performers (Sharpe > 0.3 AND low correlation with others) to full training:

```bash
uv run numereng orchestrator run \
  --config .numereng/experiments/gm-campaign-<NNN>/configs/<model_id>_full.yaml \
  --tier rtx4090
```

Create `_full.yaml` variants by switching `data.dataset_variant` from `downsampled` to `non_downsampled`.

### 2.5 Add Tier 2 and Tier 3 Models

If budget allows, train Tier 2 (should-have) and Tier 3 (nice-to-have) models from the diversity matrix. Focus on targets/algorithms not yet represented.

## Config Templates

**LightGBM on ender:**
```yaml
run_id: lgbm-ender-scout
data:
  target: target_ender_20
  feature_set: medium
  dataset_variant: downsampled
model:
  type: lgbm
  params:
    n_estimators: 2000
    learning_rate: 0.01
    max_depth: 5
    num_leaves: 31
    colsample_bytree: 0.1
training:
  engine:
    profile: purged_walk_forward
```

**XGBoost on cyrusd:**
```yaml
run_id: xgb-cyrusd-scout
data:
  target: target_cyrusd_20
  feature_set: medium
  dataset_variant: downsampled
model:
  type: xgboost
  params:
    n_estimators: 1000
    learning_rate: 0.01
    max_depth: 5
    colsample_bytree: 0.1
    subsample: 0.8
training:
  engine:
    profile: purged_walk_forward
```

**Ridge on ender:**
```yaml
run_id: ridge-ender-scout
data:
  target: target_ender_20
  feature_set: medium
  dataset_variant: downsampled
model:
  type: ridge
  params:
    alpha: 1.0
training:
  engine:
    profile: purged_walk_forward
```

**CatBoost on teager:**
```yaml
run_id: catboost-teager-scout
data:
  target: target_teager2b_20
  feature_set: medium
  dataset_variant: downsampled
model:
  type: catboost
  params:
    iterations: 2000
    learning_rate: 0.01
    depth: 5
    rsm: 0.1
    l2_leaf_reg: 3.0
training:
  engine:
    profile: purged_walk_forward
```

## Gate Criteria

- [ ] 8+ models trained with Sharpe > 0.3
- [ ] At least 3 algorithm types represented
- [ ] At least 3 different targets represented
- [ ] Pairwise correlation matrix reviewed
- [ ] No pair with correlation > 0.85 kept (redundancy removed)
- [ ] At least 4 models with pairwise correlation < 0.70
- [ ] Scout results logged in EXPERIMENT.md
- [ ] Top performers promoted to full training (or full training planned)

## Common Issues

- **All models underperform:** Check data version. Ensure payout target is correct.
- **High pairwise correlation:** Expand target diversity. Add Ridge (always low-corr with trees).
- **CatBoost OOM:** Reduce feature set to medium or use higher-memory cloud tier.

## Compute Estimate

| Scenario | Models | Backend | Duration | Cost |
|----------|--------|---------|----------|------|
| Minimum (8 scouts, downsampled) | 8 | Local | ~4 hrs | Free |
| Standard (12 scouts + 6 full) | 18 | RunPod rtx4090 | ~6 hrs | $5-10 |
| Full GM (20 scouts + 12 full) | 32 | RunPod rtx4090 | ~10 hrs | $10-15 |
