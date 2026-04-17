# Targets & Scoring

Numereng keeps target/scoring behavior aligned with persisted run provenance.

## Targets

Training config target keys:

- `data.target_col`: native training target used for model fit and native scoring outputs.
- `data.scoring_targets` (optional): explicit list of scoring targets to evaluate in deferred run scoring.

Scoring target resolution:

1. use `data.scoring_targets` when present to define the explicit extra scoring targets
2. otherwise score the native target plus opportunistic `target_ender_20` for `corr` / `fnc`
3. dedupe while preserving order

Unsuffixed key policy:

- `corr` and `fnc` keep native-target semantics
- `mmc`, `bmc`, and `bmc_last_200_eras` keep payout-target semantics
- additional explicit targets use aliased keys such as `corr_ender20`, `fnc_ender20`, `mmc_target`, and `bmc_target`

## Canonical Metric Families

Run scoring produces:

- `corr` / `corr_<alias>`
- `fnc` / `fnc_<alias>`
- `mmc` / `mmc_<alias>`
- `cwmm`
- `bmc` / `bmc_<alias>`
- `bmc_last_200_eras` / `bmc_last_200_eras_<alias>`

`bmc` is a diagnostic metric. Numereng does not emit payout estimate fields.

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
- `columns.corr_target_cols`
- `columns.fnc_target_cols`
- `columns.contribution_target_cols`
- `columns.requested_scoring_target_cols`
- `columns.scoring_targets_explicit`
- `policy.fnc_feature_set`
- `policy.fnc_target_policy`
- `policy.benchmark_min_overlap_ratio`

Feature-neutral diagnostics are stage-driven: `post_training_core` omits FNC outputs, while `post_training_full` and `all` include them.
