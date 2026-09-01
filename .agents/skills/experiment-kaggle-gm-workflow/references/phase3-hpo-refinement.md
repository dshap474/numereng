> **Compatibility Note (Current numereng CLI):**
> This phase may include legacy command examples (for example `orchestrator`, `optimize`, `ensemble`, `compare`, `neutralize-sweep`, `experiment summarize`, `experiment conclude`).
> Treat legacy commands and `.yaml` config examples as historical reference only.
> Execute this phase through `experiment-design` using current commands (`experiment create|train|report|promote`, `run train`, `run submit`) and JSON configs.

# Phase 3: HPO Refinement

**Goal:** Tune the top 3-5 performing models with Optuna to squeeze out additional performance.

## Prerequisites

- Phase 2 complete: 8+ diverse models with Sharpe > 0.3
- Top 3-5 performers identified for tuning

## Key Rules

1. **NEVER tune `colsample_bytree`** — keep fixed at 0.1. Numerai's correlated features require heavy subsampling.
2. **Use downsampled scout data for HPO trials** — full data is too expensive for 25-50 trials.
3. **Stop early if < 5% improvement** — diminishing returns on obfuscated features.

## Steps

### 3.1 Select Candidates

Pick the top 3-5 models from Phase 2 by:
- Validation Sharpe (primary)
- Low correlation with other top models (secondary)
- Different algorithm types preferred (diversity)

### 3.2 Create HPO Study Configs

Use the starter configs from `assets/hpo-study-*.yaml` as templates. Copy and customize per model.

```bash
# Copy starter config
cp .agents/skills/experiment-kaggle-gm-workflow/assets/hpo-study-lgbm.yaml \
   .numereng/experiments/gm-campaign-<NNN>/configs/hpo-lgbm-ender.yaml

# Edit to set correct target, run_id prefix, etc.
```

### 3.3 Run Optuna Studies

Use the `experiment-design` skill's optimization commands:

```bash
uv run numereng optimize run \
  --config .numereng/experiments/gm-campaign-<NNN>/configs/hpo-lgbm-ender.yaml
```

For cloud execution (faster):

```bash
uv run numereng optimize run \
  --config .numereng/experiments/gm-campaign-<NNN>/configs/hpo-lgbm-ender.yaml \
  --remote --tier r7i.4xlarge
```

### 3.4 Analyze Results

```bash
uv run numereng optimize results --study-name <study_name>
```

Check:
- Best trial Sharpe vs default config Sharpe
- Parameter importance (which params matter most)
- Convergence (are trials still improving?)

### 3.5 Export Best Configs

```bash
uv run numereng optimize export \
  --study-name <study_name> \
  --output .numereng/experiments/gm-campaign-<NNN>/configs/<model>-tuned.yaml
```

### 3.6 Validate on Full Data

Re-train tuned configs on full data to confirm improvement:

```bash
uv run numereng orchestrator run \
  --config .numereng/experiments/gm-campaign-<NNN>/configs/<model>-tuned.yaml \
  --tier rtx4090
```

Compare tuned vs default on full-data validation metrics.

## Search Spaces by Algorithm

### LightGBM (see `assets/hpo-study-lgbm.yaml`)

| Parameter | Range | Type |
|-----------|-------|------|
| n_estimators | 1000-20000 | int (log) |
| learning_rate | 0.001-0.05 | float (log) |
| max_depth | 4-10 | int |
| num_leaves | 16-512 | int (log) |
| subsample | 0.5-1.0 | float |
| reg_alpha | 0.0-5.0 | float |
| reg_lambda | 0.0-5.0 | float |
| min_child_samples | 5-100 | int |

**FIXED:** `colsample_bytree: 0.1`

### XGBoost (see `assets/hpo-study-xgboost.yaml`)

| Parameter | Range | Type |
|-----------|-------|------|
| n_estimators | 500-10000 | int (log) |
| learning_rate | 0.001-0.05 | float (log) |
| max_depth | 4-10 | int |
| subsample | 0.5-1.0 | float |
| reg_alpha | 0.0-5.0 | float |
| reg_lambda | 0.0-5.0 | float |

**FIXED:** `colsample_bytree: 0.1`

### CatBoost (see `assets/hpo-study-catboost.yaml`)

| Parameter | Range | Type |
|-----------|-------|------|
| iterations | 1000-10000 | int (log) |
| learning_rate | 0.001-0.05 | float (log) |
| depth | 4-10 | int |
| l2_leaf_reg | 0.1-10.0 | float (log) |

**FIXED:** `rsm: 0.1` (CatBoost's colsample_bytree)

## Gate Criteria

- [ ] Top 3-5 models selected for HPO
- [ ] Optuna studies completed (25-50 trials each)
- [ ] Best configs exported
- [ ] Improvement > 5% confirmed on at least one model (otherwise skip HPO)
- [ ] Tuned configs validated on full data
- [ ] Results logged in EXPERIMENT.md

**Skip condition:** If no model shows > 5% improvement from HPO, proceed to Phase 4 with default configs. HPO on obfuscated features often yields marginal gains.

## Common Issues

- **All trials similar performance:** Search space may be too narrow, or defaults are already near-optimal. This is common with Numerai.
- **Best trial much better than median:** Check for overfitting to scout/downsampled eras. Validate on full data.
- **Study fails to converge:** Increase trials to 50, or narrow search space based on parameter importance.

## Compute Estimate

| Scenario | Models | Trials | Backend | Duration | Cost |
|----------|--------|--------|---------|----------|------|
| Minimum (3 models, 25 trials) | 3 | 75 | Local | ~3 hrs | Free |
| Standard (4 models, 30 trials) | 4 | 120 | EC2 r7i.4xlarge | ~3 hrs | $3-5 |
| Full GM (5 models, 50 trials) | 5 | 250 | EC2 r7i.4xlarge | ~5 hrs | $5-10 |
