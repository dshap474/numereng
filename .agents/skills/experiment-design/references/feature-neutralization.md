## Feature Neutralization (When, Why, Where in Pipeline)

Use this reference when deciding whether/how to apply feature neutralization in an experiment.

### What It Is

Feature neutralization is a **post-processing step on predictions**.

- Train model normally.
- Generate predictions.
- Remove the linear component explained by selected neutralizer features (typically per-era).
- Rescale/rank predictions for downstream use.

It does **not** train a new model. It creates a new prediction vector from an existing model output.

### Defaults in numereng

Current implementation defaults:
- `mode = era`
- `proportion = 0.5`
- `rank_output = true`
- `neutralizer_cols = None` (auto-select numeric neutralizer columns, excluding `era`, `id`, `prediction`, and common target columns)

Important column-selection behavior:
- `neutralizer_cols=None` => auto-select.
- explicit empty columns are invalid (`neutralizer_cols=[]` / empty CSV input).

### Why Use It

Use neutralization to reduce feature exposure and improve regime robustness when a model appears overly tied to unstable features.

Typical objective:
- lower exposure risk and drawdown sensitivity,
- while preserving as much mean signal as possible.

Tradeoff:
- stronger neutralization can improve stability but may reduce raw mean performance.

### Where It Fits in HPO

If you will neutralize in production/submission, include neutralization in trial scoring.

Per-trial evaluation should look like:

`train -> predict -> neutralize -> score`

Best-practice options:
1. Full-loop: evaluate neutralization in every trial.
2. Compute-saving: run base HPO, then re-rank top-K trials with neutralization.

In both cases, final selection must be based on the same neutralized pipeline used at inference.

Implementation notes for HPO in numereng:
- when HPO neutralization is enabled, neutralizer path/schema/column checks run **before** trial execution (preflight validation).
- if training succeeds but a later trial step fails (for example indexing/neutralized scoring), failed trials retain `run_id` for artifact traceability.

Scout integration:
- in early rounds, test a limited neutralization grid (for example `p in {0.25, 0.50}`) before expanding.
- only expand neutralization complexity when scout rounds show repeatable benefit.

### Where It Fits in Ensembling

Preferred starting pattern:

`member predictions -> blend -> neutralize once`

Guidance:
- Start with **final-blend-only** neutralization.
- Add member-level neutralization only if diagnostics still show problematic exposure.
- Avoid heavy member neutralization + heavy final neutralization together (over-neutralization risk).

If both stages are used, apply lighter proportions and verify net benefit.

Current ensemble CLI controls:

```bash
uv run numereng ensemble build \
  --run-ids <run_a,run_b,...> \
  --method rank_avg \
  --neutralize-members \
  --neutralize-final \
  --neutralizer-path <neutralizer.parquet> \
  --neutralization-proportion <0..1> \
  --neutralization-mode <era|global> \
  --neutralizer-cols <csv> \
  --no-neutralization-rank
```

Notes:
- `--neutralizer-path` is required when using member/final neutralization.
- When `--neutralize-final` is enabled, the artifact directory also includes `predictions_pre_neutralization.parquet`.
- If ensemble persistence fails after artifact write and the artifact directory was newly created for that build, numereng cleans up that new directory.

### Practical Knobs

- Neutralizer set:
  - all features,
  - risky subset,
  - high-exposure subset.
- Proportion `p` (common starting grid): `0.25`, `0.50`, `0.75`.
- Granularity: prefer per-era neutralization over global neutralization.

### Hard Contracts (numereng)

- Neutralization is prediction-stage only; it does not retrain models.
- `neutralizer_path` is required whenever neutralization is enabled.
- Neutralizer join keys (`era`, `id`) must be unique.
- Joins normalize key formats internally (`era`, `id`) for matching.
- Output preserves the original prediction key formatting from the source predictions.

### Decision Rules

Good candidate for neutralization:
- high/max feature exposure,
- unstable era-level performance,
- acceptable mean drop for meaningful sharpe/consistency gain.

Avoid/limit neutralization when:
- model already low exposure and stable,
- mean signal collapses materially,
- improvement appears only on a single validation slice.

Stop/pivot tie-in:
- if neutralized variants fail to improve `bmc_last_200_eras.mean` beyond the round gate threshold (default `1e-4` to `3e-4`) for two consecutive rounds, pivot to other hypothesis dimensions.

### Recommended Starting Setup

Start simple and only add complexity if diagnostics justify it:

1. Ensemble first, then neutralize once at final blend.
2. Use `mode=era`, `proportion=0.5`, `rank_output=true`.
3. Keep neutralization settings identical between validation/HPO scoring and submission-time inference.

### Contract Notes for numereng

- Current CLI does not expose `neutralize-sweep`; perform neutralization experiments as explicit external prediction-stage variants.
- Keep neutralization decisions and rationale in `EXPERIMENT.md`.
