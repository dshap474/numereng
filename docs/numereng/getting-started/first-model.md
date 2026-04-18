# Train Your First Model

This is the shortest useful numereng path: one experiment, one config, one tracked run.

## Step 0: Prerequisites

Before training, make sure you have:

- bootstrapped the repo with `uv sync --extra dev`
- initialized `.numereng/` with `uv run numereng store init`
- populated the required Numerai datasets locally under `.numereng/datasets/<data_version>/`

If datasets are not present yet, use the [Numerai Operations](../workflows/numerai-ops.md) workflow first.

## Step 1: Create An Experiment

```bash
uv run numereng experiment create \
  --id 2026-04-18_first-model \
  --name "First Model" \
  --hypothesis "Baseline LGBM on v5.2 small features"
```

## Step 2: Add A Config

Create `.numereng/experiments/2026-04-18_first-model/configs/r1_baseline.json`:

```json
{
  "data": {
    "data_version": "v5.2",
    "dataset_variant": "non_downsampled",
    "feature_set": "small",
    "target_col": "target"
  },
  "model": {
    "type": "LGBMRegressor",
    "params": {
      "n_estimators": 2000,
      "learning_rate": 0.01,
      "num_leaves": 64,
      "colsample_bytree": 0.1,
      "random_state": 42
    }
  },
  "training": {
    "engine": {
      "profile": "purged_walk_forward"
    },
    "post_training_scoring": "core"
  }
}
```

## Step 3: Train The Config

```bash
uv run numereng experiment train \
  --id 2026-04-18_first-model \
  --config .numereng/experiments/2026-04-18_first-model/configs/r1_baseline.json
```

## Step 4: Inspect The Result

```bash
uv run numereng experiment report --id 2026-04-18_first-model
uv run numereng experiment details --id 2026-04-18_first-model
```

The resulting run lands under `.numereng/runs/<run_id>/`.

## Step 5: Re-Score Or Submit When Ready

If you deferred scoring:

```bash
uv run numereng run score --run-id <run_id>
```

When you have a candidate you want to upload:

```bash
uv run numereng run submit --model-name MY_MODEL --run-id <run_id>
```

## What To Do Next

- add more configs to the same experiment and keep comparing them
- use [Hyperparameter Optimization](../workflows/optimization.md) if you want Optuna to search the config space
- use [Agentic Research](../workflows/agentic-research.md) if you want numereng to mutate the config autonomously
- use [Serving & Model Uploads](../workflows/serving.md) if the winner should become a production package
