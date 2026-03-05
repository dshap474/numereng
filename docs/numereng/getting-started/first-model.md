# Train Your First Model

End-to-end walkthrough using current `experiment` + `run` contracts.

## Step 1: Initialize Store

```bash
uv run numereng store init
```

## Step 2: Create an Experiment

```bash
uv run numereng experiment create \
  --id 2026-02-22_first_model \
  --name "First Model" \
  --hypothesis "Baseline LGBM on v5.2"
```

## Step 3: Create `configs/run.json`

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
    }
  }
}
```

## Step 4: Train the Experiment Run

```bash
uv run numereng experiment train \
  --id 2026-02-22_first_model \
  --config configs/run.json
```

## Step 5: Inspect Results

```bash
uv run numereng experiment report --id 2026-02-22_first_model
```

Run artifacts are under `.numereng/runs/<run_id>/`.

## Step 6: Submit (Optional)

```bash
uv run numereng run submit --model-name MY_MODEL --run-id <run_id>
```

## Next Steps

- [Training Models](../workflows/training.md)
- [Experiments](../workflows/experiments.md)
- [Submissions](../workflows/submission.md)
- [Metrics](../reference/metrics.md)
