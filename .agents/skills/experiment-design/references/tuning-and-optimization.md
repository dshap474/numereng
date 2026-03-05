## Tuning and Optimization (Manual Sweep Mode)

Use this reference when the user asks for hyperparameter tuning or controlled parameter exploration.

### Current Contract

The current numereng CLI does not expose `optimize`, `baselines`, or `neutralize-sweep` command families.
Optimization is executed as explicit config variants plus repeated `experiment train` runs.

### Sweep Structure

Use one variable change per config variant.

Suggested loop:
1. Start from a base config.
2. Create 4-5 variant configs by default.
3. Train each config in the same experiment.
4. Rank all new runs with the same metric.
5. Keep top variants, then iterate.

Planning helper:
- `assets/hpo-study-template.json` (canonical shape for `numereng hpo create --study-config ...`)

Optional HPO run command:
- `uv run numereng hpo create --study-config <path.json>`

### Sweep Selection by Research Type

Choose the sweep dimension that matches the hypothesis:
- target/label idea -> sweep target/preprocessing variants first,
- model architecture idea -> sweep model hyperparameters,
- ensemble/blend idea -> sweep component selection and weight strategy,
- training-procedure idea -> sweep procedure controls (profile, loading/scoring mode, neutralization settings),
- data-scope idea -> sweep feature scope and dataset variant (`downsampled` for scout vs `non_downsampled` for scale).

Do not run broad unfocused sweeps. Each round should answer one concrete question.

### Commands

```bash
# Train variants
uv run numereng experiment train --id <id> --config <variant_a>.json
uv run numereng experiment train --id <id> --config <variant_b>.json
uv run numereng experiment train --id <id> --config <variant_c>.json

# Rank with fixed metric
uv run numereng experiment report --id <id> --metric bmc_last_200_eras.mean --limit 20 --format table

# Inspect detailed experiment state
uv run numereng experiment details --id <id> --format json
```

### Suggested Sweep Dimensions

- `model.params.learning_rate`
- `model.params.num_leaves`
- `model.params.min_child_samples`
- `model.params.subsample`
- `model.params.colsample_bytree`
- `training.engine.profile` (`simple|purged_walk_forward|submission`)
- `data.dataset_variant` (`downsampled` for scout vs `non_downsampled` for scale)

### Guardrails

- Keep one-variable-at-a-time within a round.
- Default round size is 4-5 variants; use fewer only when explicitly requested.
- Use only supported profiles: `simple`, `purged_walk_forward`, `submission`.
- Do not use legacy config fields (`training.method`, `training.strategy`, `training.cv`).
- Use scout->scale progression:
  - scout with lower-cost settings first (for example `data.dataset_variant=downsampled`),
  - scale top candidates only after repeatable scout improvements.
- Apply numeric stop gate between rounds:
  - if best `bmc_last_200_eras.mean` gain is below `1e-4` to `3e-4` for two consecutive rounds, run remaining-knobs audit before continuing.

### Verification

- `experiment report` contains new run IDs from the sweep.
- Winning config and rationale are recorded in `EXPERIMENT.md`.
- `EXPERIMENT.md` includes round-best delta and continue/pivot/stop decision with gate status.
