# Validation

Numereng uses profile-driven era-aware validation. There is no standalone `validation` config block.

## Why Era-Aware Validation

Numerai data is temporal (`era` buckets). Validation must preserve time ordering and embargo gaps to prevent leakage.

## Training Profiles

Validation behavior is controlled by `training.engine.profile`:

| Profile | Behavior | Notes |
|---------|----------|-------|
| `simple` | Train on train eras, validate on validation eras | Holdout split, no embargo |
| `purged_walk_forward` | Purged walk-forward CV over train+validation eras | Recommended evaluation mode |
| `submission` | One-shot fit on full history | Final fit mode, no validation metrics |

## Embargo and Horizon

For `purged_walk_forward`:

- Walk-forward chunk size is fixed at `156` eras.
- Embargo is horizon-based: `20d -> 8`, `60d -> 16`.
- Horizon resolution uses `data.target_horizon` first, then `target_col` naming.
- If neither yields a clear horizon, training fails with `training_engine_target_horizon_ambiguous`.

## Overfitting Signals

| Signal | Meaning |
|--------|---------|
| High `corr20v2_sharpe` + low `mmc_mean` | Limited meta-model contribution upside |
| Proxy metrics far exceed full-history expectations | Possible selection bias |
| High mean + high std | Unstable era-level performance |
| Very high sharpe on tiny data slices | Often overfit |

## High-Risk Gotchas

- `submission` does not produce validation metrics.
- Training does not apply row-level subsampling; use dataset-level downsampling when reducing training size.
- Validation behavior is controlled only by `training.engine.profile`; legacy engine parameters are not supported.
