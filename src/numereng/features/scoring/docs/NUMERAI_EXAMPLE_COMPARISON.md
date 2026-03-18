# Scoring Comparison: Numereng vs `numerai-example`

This note is intentionally narrow. It exists only to capture the current
high-level relationship between Numereng scoring math and the vendored
`numerai-example` snapshot in this repo.

## What Still Matches

For the shared metric surface, Numereng and `numerai-example` still match on
the core math:

- `corr`
- `mmc`
- `bmc`
- `bmc_last_200_eras`
- `avg_corr_with_benchmark`
- summary stats: `mean`, `std`, `sharpe`, `max_drawdown`

Both codepaths ultimately delegate the shared scoring formulas to
`numerai_tools.scoring`, especially:

- `numerai_corr`
- `correlation_contribution`

## What Numereng Adds

Numereng's scoring package is broader than the example package and now also
owns:

- target-family scoring such as `corr_ender20`, `mmc_ender20`, `bmc_ender20`
- feature-neutral diagnostics
- score provenance
- persisted scoring artifacts under `artifacts/scoring/`
- rescoring of existing runs
- configurable benchmark prediction sources via the active benchmark contract

So the fair comparison is not package size or total code volume. The useful
comparison is only whether the shared formulas still align.

## Current Conclusion

On the shared metric surface, Numereng remains mathematically aligned with the
vendored `numerai-example` implementation.

Differences are primarily in:

- package structure
- validation and join handling
- persistence/provenance
- benchmark-source configuration
- additional diagnostic metric families

This file is not the canonical scoring contract. The source of truth is:

- [SCORING_CONTRACT.md](/Users/daniel/Developer/numereng/src/numereng/features/scoring/docs/SCORING_CONTRACT.md)
