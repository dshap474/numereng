> **Compatibility Note (Current numereng CLI):**
> This phase may include legacy command examples (for example `orchestrator`, `optimize`, `ensemble`, `compare`, `neutralize-sweep`, `experiment summarize`, `experiment conclude`).
> Treat legacy commands and `.yaml` config examples as historical reference only.
> Execute this phase through `experiment-design` using current commands (`experiment create|train|report|promote`, `run train`, `run submit`) and JSON configs.

# Phase 1: EDA & Baseline

**Goal:** Establish floor metrics with a single LightGBM on the payout target.

## Prerequisites

- Data downloaded and fresh: `uv run numereng download`
- numereng status checks pass: `uv run numereng status`
- Sufficient local RAM (~8GB free for downsampled scout runs)

## Steps

### 1.1 Download & Verify Data

```bash
uv run numereng download
uv run numereng status
```

Confirm data version and round number. If data is stale (>7 days), re-download.

### 1.2 Create Campaign Experiment

```bash
uv run numereng experiment create \
  --id gm-campaign-<NNN> \
  --name "GM Campaign <NNN>" \
  --hypothesis "Full GM workflow: diverse models -> HPO -> seed ensemble -> blend on payout target" \
  --tags "gm-workflow,campaign"
```

Copy `assets/gm-campaign-template.md` to the experiment's EXPERIMENT.md for structured logging.

### 1.3 Train Baseline

Train a single LightGBM on the payout target with scout settings (`data.dataset_variant: downsampled`, `training.engine.profile: purged_walk_forward`):

```bash
uv run numereng orchestrator run \
  --config .numereng/experiments/gm-campaign-<NNN>/configs/baseline.yaml \
  --local
```

Baseline config (create at `.numereng/experiments/gm-campaign-<NNN>/configs/baseline.yaml`):

```yaml
run_id: baseline-ender-lgbm
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
    subsample: 0.8
    reg_alpha: 0.1
    reg_lambda: 0.1
    min_child_samples: 20
training:
  engine:
    profile: purged_walk_forward
```

### 1.4 Record Floor Metrics

```bash
uv run numereng experiment summarize --id gm-campaign-<NNN>
```

Record in EXPERIMENT.md:
- Sharpe (validation)
- corr_mean
- max_drawdown
- Feature exposure

### 1.5 Quick EDA (Optional)

```bash
uv run numereng eda run --experiment gm-campaign-<NNN>
uv run numereng eda features --top 20 --sort-by psi_val
```

## Gate Criteria

- [ ] Data downloaded and verified
- [ ] Baseline trained successfully
- [ ] Baseline Sharpe > 0.3 on validation
- [ ] No data quality issues (NaN rates, era coverage)
- [ ] Floor metrics recorded in EXPERIMENT.md
- [ ] Experiment created and active

**If Sharpe < 0.3:** Check config for issues (wrong target, incorrect params). The standard LGBM scout config should exceed 0.3. Do not proceed until baseline is healthy.

## Compute Estimate

| Resource | Duration | Cost |
|----------|----------|------|
| Local (scout/downsampled) | ~15-30 min | Free |
