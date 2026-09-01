> **Compatibility Note (Current numereng CLI):**
> This phase may include legacy command examples (for example `orchestrator`, `optimize`, `ensemble`, `compare`, `neutralize-sweep`, `experiment summarize`, `experiment conclude`).
> Treat legacy commands and `.yaml` config examples as historical reference only.
> Execute this phase through `experiment-design` using current commands (`experiment create|train|report|promote`, `run train`, `run submit`) and JSON configs.

# Phase 6: Post-Processing

**Goal:** Apply neutralization and validate prediction quality before submission.

## Prerequisites

- Phase 5 complete: final ensemble selected
- Ensemble predictions available for post-processing

## Steps

### 6.1 Rank Normalization

Rank normalization is applied automatically by the numereng engine. Verify predictions are ranked to [0, 1]:

```bash
# Check prediction distribution
uv run numereng compare <ensemble_run_id>
```

Predictions should be approximately uniform on [0, 1] with:
- Min ≈ 0.0
- Max ≈ 1.0
- Mean ≈ 0.5
- Std ≈ 0.29 (uniform distribution)

### 6.2 Neutralization Sweep

Test multiple neutralization proportions:

```bash
uv run numereng neutralize-sweep \
  --model .numereng/runs/<ensemble_run_id>/artifacts/model/model.pkl \
  --proportions 0.0,0.3,0.5,0.7 \
  --threshold 0.10
```

Compare results across proportions:

| Proportion | Sharpe | Feature Exposure | MMC |
|------------|--------|-----------------|-----|
| 0.0 | Highest CORR | Highest exposure | Lowest MMC |
| 0.3 | Moderate | Moderate | Moderate |
| 0.5 | Balanced | Low | Good |
| 0.7 | Lower CORR | Lowest | Highest MMC |

**Default recommendation:** Start with 0.5. Use 0.3 if CORR matters more. Use 0.7 only if targeting high MMC.

**Warning:** Proportion 0.7 is aggressive. While it improved backtested metrics in Phase 7 experiments, it can remove real signal. Monitor live CORR after submission.

### 6.3 Feature Exposure Check

After neutralization, verify feature exposure is below threshold:

```bash
uv run numereng eda features --top 20 --sort-by exposure
```

**Threshold:** Max feature exposure < 0.10. If any feature exposure > 0.10 after neutralization at your chosen proportion, increase the proportion.

### 6.4 Prediction Distribution Validation

Final checks on the neutralized predictions:

1. **Uniform-ish distribution:** Histogram should be roughly flat, not peaked
2. **Per-era consistency:** Predictions should have similar distribution across eras
3. **No extreme values:** After rank normalization, values strictly in [0, 1]
4. **Coverage:** No missing predictions (all stock IDs present)

### 6.5 Select Final Post-Processing

Record the chosen neutralization proportion and feature exposure in EXPERIMENT.md. This becomes the production configuration.

```yaml
postprocess:
  neutralize:
    enabled: true
    proportion: 0.5  # or your chosen value
    threshold: 0.10
```

## Gate Criteria

- [ ] Predictions rank-normalized to [0, 1]
- [ ] Neutralization sweep completed
- [ ] Neutralization proportion selected and justified
- [ ] Feature exposure < 0.10 after neutralization
- [ ] Prediction distribution validated (uniform-ish)
- [ ] No missing predictions
- [ ] Post-processing config recorded in EXPERIMENT.md

## Common Issues

- **Feature exposure remains > 0.10:** Increase neutralization proportion. If still high at 0.7, the signal may be feature-dominated — consider retraining with different features.
- **Sharpe drops too much with neutralization:** The model may rely heavily on common signal. Consider using a lower proportion (0.3) and accepting some feature exposure.
- **Predictions peaked at 0.5:** May indicate insufficient signal or over-neutralization. Check base model predictions before neutralization.

## Compute Estimate

| Task | Backend | Duration | Cost |
|------|---------|----------|------|
| Neutralization sweep | Local | ~15 min | Free |
| Distribution checks | Local | ~5 min | Free |
