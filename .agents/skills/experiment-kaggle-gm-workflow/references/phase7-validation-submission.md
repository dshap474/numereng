> **Compatibility Note (Current numereng CLI):**
> This phase may include legacy command examples (for example `orchestrator`, `optimize`, `ensemble`, `compare`, `neutralize-sweep`, `experiment summarize`, `experiment conclude`).
> Treat legacy commands and `.yaml` config examples as historical reference only.
> Execute this phase through `experiment-design` using current commands (`experiment create|train|report|promote`, `run train`, `run submit`) and JSON configs.

# Phase 7: Validation & Submission

**Goal:** Final holdout evaluation, overfitting diagnostics, and live deployment.

## Prerequisites

- Phase 6 complete: post-processed predictions ready
- Numerai credentials configured: `NUMERAI_PUBLIC_ID`, `NUMERAI_SECRET_KEY` in `.env`

## Steps

### 7.1 Full Holdout Evaluation

Evaluate the final ensemble on all validation splits:

```bash
uv run numereng experiment summarize --id gm-campaign-<NNN> --metric payout
uv run numereng experiment show --id gm-campaign-<NNN> --plot
```

Record extended metrics:

| Metric | Hillclimb | Holdout | Full Val |
|--------|-----------|---------|----------|
| Sharpe | | | |
| corr_mean | | | |
| BMC mean | | | |
| Payout score | | | |
| max_drawdown | | | |
| feature_exposure | | | |

### 7.2 Overfitting Diagnostics

Compute degradation ratios:

- **Hillclimb-to-holdout:** (1 - holdout_sharpe / hillclimb_sharpe)
  - < 40%: Good
  - 40-60%: Acceptable
  - > 60%: Concerning — consider equal-weight blend

- **Validation-to-live expectation:** Expect 20-40% further degradation from validation to live performance (based on Numerai community experience).

### 7.3 Comparison with Benchmark

Compare your ensemble against Numerai benchmark models:

| Model | Val Sharpe | Your Ensemble Sharpe | Delta |
|-------|-----------|---------------------|-------|
| v51_lgbm_cyrusd20 | 1.70 | | |
| v51_lgbm_teager2b20 | 1.77 | | |

Your ensemble should outperform benchmarks on the payout metric (not just Sharpe) to justify deployment.

### 7.4 Prepare Submission Source

Use the `numerai-model-upload` skill for numereng-native submission flow (or `official-numerai-ops` for direct official API workflows).

Resolve exactly one submission source:
- `--run-id <final_ensemble_run_id>` from your final run artifacts, or
- `--predictions <path/to/predictions.csv|parquet>`

### 7.5 Live Submission

**Requires explicit user confirmation before proceeding.**

```bash
uv run numereng run submit \
  --model-name <numerai_model_name> \
  --run-id <final_ensemble_run_id>
```

Optional pre-submit neutralization:

```bash
uv run numereng run submit \
  --model-name <numerai_model_name> \
  --run-id <final_ensemble_run_id> \
  --neutralize \
  --neutralizer-path <path/to/neutralizers.parquet> \
  --neutralization-proportion 0.5 \
  --neutralization-mode era
```

After submission:
- Verify submission accepted on Numerai dashboard
- Note the round number
- Record submission details in EXPERIMENT.md

### 7.6 Post-Submission Monitoring Plan

Set up monitoring for the first 4-8 weeks:

1. **Weekly:** Check CORR, BMC, and payout score on Numerai dashboard
2. **Monthly:** Compare live performance to validation expectations
3. **Alert thresholds:**
   - CORR consistently < 0.005: investigate signal decay
   - BMC turns negative: benchmark model composition may have changed
   - Payout negative for 3+ consecutive rounds: consider model update

### 7.7 Conclude Experiment

```bash
uv run numereng experiment conclude \
  --id gm-campaign-<NNN> \
  --verdict <success|failure|inconclusive> \
  --notes "Final ensemble: <description>. Deployed to <model_name>. Monitoring plan active."
```

## Gate Criteria

- [ ] Full holdout evaluation completed with extended metrics
- [ ] Overfitting diagnostics reviewed (degradation < 60%)
- [ ] Benchmark comparison documented
- [ ] Submission source validated (run artifact or predictions path)
- [ ] Live submission completed (with user confirmation)
- [ ] Submission accepted on Numerai dashboard
- [ ] Monitoring plan documented
- [ ] Experiment concluded with verdict and notes

## Common Issues

- **Submission request fails validation:** Usually wrong source flags (must provide exactly one of `--run-id` or `--predictions`) or missing neutralizer path when neutralization is enabled.
- **Submission rejected:** Check round deadline. Submissions must be in before Saturday closing time.
- **Model too large to upload:** Ensure packaging only includes predictions, not the full model.
- **Live performance significantly worse than validation:** Expected to some degree (20-40% degradation). If > 50% after 8+ rounds, consider re-running the campaign.

## Compute Estimate

| Task | Backend | Duration | Cost |
|------|---------|----------|------|
| Holdout evaluation | Local | ~15 min | Free |
| Submission source validation | Local | ~5 min | Free |
| Submission | Local | ~5 min | Free |
