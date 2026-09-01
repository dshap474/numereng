# Seed Ensembling Reference

Use this reference for reducing variance by averaging multiple seed runs of the same core config.

## Core Formula

```
Ensemble Variance = sigma^2 / K + (K - 1) * rho * sigma^2 / K
```

- `sigma^2`: individual model variance
- `K`: number of seeds
- `rho`: inter-seed correlation

High `rho` means diminishing returns from adding more seeds.

## Practical Seed Counts

| Model Type | Typical Seed Count |
|---|---:|
| LightGBM / XGBoost | 5 |
| Linear / Ridge | 3 |
| Neural tabular models | 8-15 |

## Rules

1. Pre-commit seed lists before training.
2. Compare strategies with paired seeds.
3. Average predictions, not per-seed scores.
4. Rank-normalize predictions before final submission.
5. Track seed variance in experiment notes.

## Suggested Workflow

1. Create per-seed configs under `experiments/<id>/configs/`.
2. Train each seed via `experiment train`.
3. Record seed-level metrics in `EXPERIMENT.md`.
4. Build an internal seed blend with `ensemble build` (preferred):

```bash
numereng ensemble build \
  --experiment-id <id> \
  --run-ids <seed_run_1,seed_run_2,seed_run_3> \
  --method rank_avg \
  --metric corr20v2_sharpe \
  --target target_ender_20 \
  --name "seed_blend" \
  --weights 0.34,0.33,0.33
```

5. Use external blend generation only when you need a custom blend not supported by current CLI.
6. Submit final blended prediction file via `run submit --predictions` when using external artifacts.

## Continue/Stop Rules for More Seeds

Add more seeds only when both hold:
1. Current seed blend improves meaningful metrics vs strongest single-seed run.
2. Seed-level variability is still high enough to justify variance reduction.

Default stop signals:
- two consecutive seed-expansion rounds with < `1e-4` gain on `bmc_last_200_eras.mean`,
- or inter-seed correlation is high and added seeds do not change blend ranking materially.

## Diagnostics

Track coefficient of variation across seed-level primary metric values:

`cv = std(metric) / mean(metric)`

- `cv > 0.30` often indicates unstable setup.
- Lower CV is preferred for production candidates.
- Track marginal gain from adding each seed group and log the continue/pivot/stop decision in `EXPERIMENT.md`.
