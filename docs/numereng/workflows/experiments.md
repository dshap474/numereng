# Experiments

Use `experiment` when you want numereng to keep one model-development project organized under one root: configs, round scoring, champion tracking, reports, HPO studies, ensembles, serving packages, and research state.

## Create An Experiment

```bash
uv run numereng experiment create \
  --id 2026-04-18_lgbm-baseline \
  --name "LGBM baseline" \
  --hypothesis "baseline before wider sweeps" \
  --tags baseline,lgbm
```

This creates `.numereng/experiments/<experiment_id>/` with:

- `experiment.json`
- `EXPERIMENT.md`
- `configs/`
- `run_plan.csv`
- `run_scripts/`

## Train A Config Inside The Experiment

```bash
uv run numereng experiment train \
  --id 2026-04-18_lgbm-baseline \
  --config .numereng/experiments/2026-04-18_lgbm-baseline/configs/r1_baseline.json
```

Useful overrides:

- `--profile <simple|purged_walk_forward|full_history_refit>`
- `--post-training-scoring <none|core|full|round_core|round_full>`
- `--workspace <path>`

## Launch One Planned Window

```bash
uv run numereng experiment run-plan \
  --id 2026-04-18_lgbm-baseline \
  --start-index 1 \
  --end-index 5
```

Use this when one experiment already has a prepared `run_plan.csv` and you want numereng to execute one contiguous window instead of launching configs one by one.

## Batch Score One Round

```bash
uv run numereng experiment score-round \
  --id 2026-04-18_lgbm-baseline \
  --round r1 \
  --stage post_training_full
```

Use this when round configs deferred scoring and you want numereng to materialize the whole round in one pass.

## Inspect, Report, And Promote

```bash
uv run numereng experiment list
uv run numereng experiment details --id 2026-04-18_lgbm-baseline
uv run numereng experiment report --id 2026-04-18_lgbm-baseline
uv run numereng experiment promote --id 2026-04-18_lgbm-baseline
uv run numereng experiment pack --id 2026-04-18_lgbm-baseline
```

`experiment pack` overwrites `EXPERIMENT.pack.md` with a compact dashboard-aligned report.

## Archive And Restore

```bash
uv run numereng experiment archive --id 2026-04-18_lgbm-baseline
uv run numereng experiment unarchive --id 2026-04-18_lgbm-baseline
```

Archived experiments move under `.numereng/experiments/_archive/<experiment_id>/`.

## High-Risk Gotchas

- keep experiment configs inside the experiment’s `configs/` folder whenever possible
- `experiment train` enforces experiment-linked output discipline; do not point it at arbitrary output roots
- `round_core` and `round_full` rely on the experiment context and config naming conventions
- `experiment score-round` only resolves finished runs that still have persisted predictions

## Read Next

- [Agentic Research](agentic-research.md)
- [Hyperparameter Optimization](optimization.md)
- [Ensembles](ensembles.md)
- [Serving & Model Uploads](serving.md)
