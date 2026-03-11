# Targets & Scoring

Numereng keeps target/scoring behavior aligned with persisted run provenance.

## Targets

Training config target keys:

- `data.target_col`: native training target used for model fit and native scoring outputs.
- `data.scoring_targets` (optional): explicit list of scoring targets to evaluate in post-run scoring.

Scoring target resolution:

1. use `data.scoring_targets` when present
2. otherwise score the native target plus `target_ender_20`
3. dedupe while preserving order

The native target always keeps the unsuffixed metric keys (`corr`, `fnc`, `mmc`). Additional scoring targets use aliased keys such as `corr_ender20`, `fnc_ender20`, and `mmc_ender20`.

## Canonical Metric Families

Run scoring produces:

- `corr` / `corr_<alias>`
- `fnc` / `fnc_<alias>`
- `mmc` / `mmc_<alias>`
- `cwmm`
- `bmc`
- `bmc_last_200_eras`
- `feature_exposure`
- `max_feature_exposure`

`bmc` and feature-exposure metrics are diagnostics. Numereng does not emit payout estimate fields.

## FNC Semantics

Canonical FNC is:

1. neutralize predictions to `fncv3_features`
2. correlate the neutralized submission against the scoring target being evaluated

That means:

- `fnc` uses the native run target
- `fnc_<alias>` uses the corresponding aliased scoring target

## Provenance

`score_provenance.json` records source fingerprints, fixed scoring policy, and join coverage. In particular:

- `joins.predictions_rows`
- `joins.benchmark_source_rows`
- `joins.benchmark_overlap_rows`
- `joins.benchmark_missing_rows`
- `joins.benchmark_missing_eras`
- `joins.meta_overlap_rows`
- `joins.meta_source_rows`
- `policy.fnc_feature_set`
- `policy.fnc_target_policy`
- `policy.benchmark_min_overlap_ratio`
- `policy.include_feature_neutral_metrics`

Feature-neutral metrics can be disabled via `include_feature_neutral_metrics=false`; when disabled, FNC/exposure outputs and their FNC-specific provenance fields are omitted.
