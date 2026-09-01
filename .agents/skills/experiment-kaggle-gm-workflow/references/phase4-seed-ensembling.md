> **Compatibility Note (Current numereng CLI):**
> This phase may include legacy command examples (for example `orchestrator`, `optimize`, `ensemble`, `compare`, `neutralize-sweep`, `experiment summarize`, `experiment conclude`).
> Treat legacy commands and `.yaml` config examples as historical reference only.
> Execute this phase through `experiment-design` using current commands (`experiment create|train|report|promote`, `run train`, `run submit`) and JSON configs.

# Phase 4: Seed Ensembling

**Goal:** Reduce prediction variance by averaging multiple random seeds per model config.

## Prerequisites

- Phase 3 complete (or skipped): final model configs established
- Cloud backend available for multi-seed training

## Why Seed Ensembling

GBDTs use random subsampling (features, data) during training. Different seeds produce different models with:
- Same expected performance
- Different error patterns
- Averaging reduces variance without sacrificing bias

Research shows: 5 seeds typically captures 80%+ of available variance reduction for GBDTs (ρ ≈ 0.85-0.95 between seeds). Diminishing returns after K=5.

## Pre-Committed Seed List

Use these seeds consistently across all models:

| Seed Slot | Value | Used For |
|-----------|-------|----------|
| Seed 1 | 42 | All models (default) |
| Seed 2 | 123 | All models |
| Seed 3 | 456 | All models |
| Seed 4 | 789 | GBDTs only |
| Seed 5 | 1011 | GBDTs only |

**Linear models (Ridge):** Use 3 seeds (42, 123, 456). Ridge is deterministic given the same data, but seed affects any randomized preprocessing or CV splits.

## Steps

### 4.1 Generate Seed Configs

For each model config from Phase 2/3, create seed variants:

```yaml
# lgbm-ender-s42.yaml (seed 42 - often already exists from Phase 2)
run_id: lgbm-ender-s42
model:
  type: lgbm
  params:
    random_state: 42
    # ... other params from tuned config

# lgbm-ender-s123.yaml
run_id: lgbm-ender-s123
model:
  type: lgbm
  params:
    random_state: 123
    # ... same params, different seed
```

### 4.2 Train All Seeds

```bash
# Train each seed variant sequentially
for seed in 42 123 456 789 1011; do
  uv run numereng orchestrator run \
    --config .numereng/experiments/gm-campaign-<NNN>/configs/lgbm-ender-s${seed}.yaml \
    --tier rtx4090
done
```

Or use the experiment-design seed-ensemble workflow if available.

### 4.3 Verify Seed Variance

Check that seed variance is within expected bounds:

```bash
# Compare seed variants
uv run numereng experiment summarize --id gm-campaign-<NNN> --metric sharpe
```

Compute CV(Sharpe) = std(Sharpe_across_seeds) / mean(Sharpe_across_seeds):
- CV < 0.15: Low variance (3 seeds may suffice)
- CV 0.15-0.30: Normal (use 5 seeds)
- CV > 0.30: High variance (investigate — may indicate unstable config)

### 4.4 Create Seed Averages

For each model config, combine its seed variants:

```bash
# Average seed predictions
uv run numereng ensemble build \
  --runs lgbm-ender-s42,lgbm-ender-s123,lgbm-ender-s456,lgbm-ender-s789,lgbm-ender-s1011 \
  --method equal_weight
```

The process:
1. Per-seed: rank-normalize predictions to [0, 1]
2. Arithmetic mean across seeds
3. Final rank normalization

### 4.5 Evaluate Seed Averages vs Singles

Compare the seed-averaged model against individual seeds:

```bash
uv run numereng compare lgbm-ender-seed-avg lgbm-ender-s42
```

Expected: seed average has lower Sharpe variance (more stable) with similar or slightly better mean Sharpe.

## Gate Criteria

- [ ] All seed variants trained (5 per GBDT, 3 per linear)
- [ ] CV(Sharpe) < 0.30 for each model config
- [ ] Seed averages created for each model config
- [ ] Seed averages show lower variance than singles
- [ ] Results logged in EXPERIMENT.md

## Common Issues

- **One seed dramatically different:** Check for training failure or data issue. Remove outlier seed and re-average.
- **CV(Sharpe) > 0.30:** Config may be unstable. Consider reducing model complexity (fewer trees, higher regularization).
- **Linear models identical across seeds:** Expected — Ridge is deterministic. Only 1 seed needed for pure Ridge with no preprocessing randomness.

## Compute Estimate

| Scenario | Models | Seeds/Model | Total Runs | Backend | Duration | Cost |
|----------|--------|-------------|------------|---------|----------|------|
| Minimum (4 models, 3 seeds) | 4 | 3 | 12 | RunPod rtx4090 | ~3 hrs | $3-5 |
| Standard (6 models, 5 seeds) | 6 | 5 | 30 | RunPod rtx4090 | ~6 hrs | $5-10 |
| Full GM (8 models, 5 seeds) | 8 | 5 | 40 | RunPod rtx4090 | ~8 hrs | $8-12 |

Note: Linear models need fewer seeds (3 vs 5), so actual counts are slightly lower.
