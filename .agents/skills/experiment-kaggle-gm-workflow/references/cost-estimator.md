# Cost Estimator

Compute cost reference for planning GM workflow campaigns.

## Per-Phase Cost Table

| Phase | Local Hours | EC2 Cost | RunPod Cost | Notes |
|-------|-------------|----------|-------------|-------|
| 1 - EDA & Baseline | 0.5-1 hr | - | - | Always local |
| 2 - Diversity Campaign | 4-10 hr (scout/downsampled) | $2-8 | $3-12 | Main cost driver |
| 3 - HPO Refinement | 2-5 hr (scout/downsampled) | $2-5 | $2-8 | Skip if marginal gains |
| 4 - Seed Ensembling | 3-8 hr | $3-8 | $3-12 | Linear with seed count |
| 5 - Ensemble Construction | 1-2 hr | - | - | Mostly local |
| 6 - Post-Processing | 0.5 hr | - | - | Always local |
| 7 - Validation & Submission | 0.5 hr | - | - | Always local |

## Tier Selection Guide

### RunPod (Default Backend for GPU Training)

| Tier | VRAM | Spot Price | Best For |
|------|------|-----------|----------|
| rtx3090 | 24GB | ~$0.20/hr | Budget scout runs |
| rtx4090 | 24GB | ~$0.40/hr | Default for most training |
| l40 | 48GB | ~$0.70/hr | Large feature sets |
| a100-40 | 40GB | ~$0.80/hr | Production quality |
| a100-80 | 80GB | ~$1.20/hr | Very large models |

### EC2 CPU (For HPO Studies)

| Tier | RAM | Spot Price | Best For |
|------|-----|-----------|----------|
| m7i.xlarge | 16GB | ~$0.05/hr | Scout/downsampled HPO |
| r7i.2xlarge | 64GB | ~$0.15/hr | Standard training |
| r7i.4xlarge | 128GB | ~$0.30/hr | Full data + HPO (recommended) |
| r7i.8xlarge | 256GB | ~$0.60/hr | Large sweeps |

### When to Use Which

- **Scout/downsampled, single runs:** Local (free)
- **Scout/downsampled, many runs:** EC2 m7i.xlarge or RunPod rtx3090
- **Full data, GBDT training:** RunPod rtx4090 (default)
- **HPO studies (many trials):** EC2 r7i.4xlarge (cheaper per trial for CPU models)
- **Full data, large feature set:** RunPod l40 or a100

## Budget Scenarios

### Minimum Viable (~$5-8)

Tight budget. Skip HPO, use 3 seeds, scout/downsampled-only training.

| Phase | Approach | Cost |
|-------|----------|------|
| 1 | Local baseline | $0 |
| 2 | 8 scouts (downsampled, local) + 4 full (RunPod rtx3090) | $3-4 |
| 3 | Skip | $0 |
| 4 | 3 seeds x 4 models (RunPod rtx3090) | $2-3 |
| 5-7 | Local | $0 |
| **Total** | | **$5-7** |

### Standard (~$15-25)

Recommended. Moderate HPO, 5 seeds, full training for top models.

| Phase | Approach | Cost |
|-------|----------|------|
| 1 | Local baseline | $0 |
| 2 | 12 scouts (downsampled) + 8 full (RunPod rtx4090) | $5-8 |
| 3 | 3-4 HPO studies, 30 trials (EC2 r7i.4xlarge) | $3-5 |
| 4 | 5 seeds x 6 models (RunPod rtx4090) | $5-8 |
| 5-7 | Local (stacking optional, +$2) | $0-2 |
| **Total** | | **$13-23** |

### Full GM (~$30-45)

Maximum thoroughness. Extensive HPO, 5-10 seeds, full stacking.

| Phase | Approach | Cost |
|-------|----------|------|
| 1 | Local baseline + EDA | $0 |
| 2 | 20 scouts + 12 full (RunPod rtx4090) | $8-12 |
| 3 | 5 HPO studies, 50 trials (EC2 r7i.4xlarge) | $5-10 |
| 4 | 5-10 seeds x 8 models (RunPod rtx4090) | $10-15 |
| 5-7 | Local + stacking | $2-3 |
| **Total** | | **$25-40** |

## Time Estimates

| Scenario | Wall Clock (sequential) | Wall Clock (with cloud parallelism) |
|----------|------------------------|--------------------------------------|
| Minimum | ~8-12 hours | ~4-6 hours |
| Standard | ~16-24 hours | ~8-12 hours |
| Full GM | ~24-40 hours | ~12-20 hours |

Note: Wall clock includes model training time, analysis, and decision-making. Cloud runs execute sequentially (one at a time) to stay within budget and avoid issues.

## Cost Control Tips

1. **Scout on downsampled data first** — 10x cheaper than full data
2. **Kill non-performing runs early** — Don't wait for completion if metrics are clearly bad
3. **Use EC2 for CPU-bound HPO** — 50-70% cheaper than RunPod for non-GPU workloads
4. **Set budget limits:** `--budget 20.0 --timeout 7200`
5. **Check spot pricing before launching:** `uv run numereng runpod pricing` or `uv run numereng cloud pricing`
