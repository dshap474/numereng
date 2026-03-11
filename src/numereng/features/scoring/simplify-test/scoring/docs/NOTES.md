# Scoring Notes: Feature Exposure NaN / Divide Warnings

## Summary

We investigated repeated post-run scoring warnings of the form:

`RuntimeWarning: invalid value encountered in divide`

The warning source is local Numereng feature-exposure diagnostics, not model
training and not the core CORR/BMC/MMC scoring paths for the affected runs.

The key conclusion is:

- the warning is produced by the local `feature_exposure` calculation
- the underlying math is expected when a correlation input has zero variance
- the current implementation should be hardened locally rather than ignored or
  globally warning-suppressed

## What Triggered the Warning

The warning is emitted from:

- `src/numereng/features/scoring/metrics.py`
- function: `_feature_exposure_stats_for_frame(...)`
- current line behavior:
  - rank-normalize prediction
  - rank-normalize all exposure features
  - compute `ranked_features.corrwith(ranked_pred).abs()`

This calls into pandas/NumPy correlation machinery. If a feature is constant
within an era, the correlation with the prediction is undefined and NumPy emits
the divide warning while returning `NaN`.

This is distinct from:

- `corr`
- `fnc`
- `bmc`
- `mmc`
- `cwmm`

for the specific `target_ralph_20` / `target_rowan_20` warning runs we traced.

There was also a separate local finding for `target_ralph_60`:

- some eras have zero target variance
- plain correlation is undefined for those eras too

That is a separate edge case from the feature-exposure warning storm.

## Local Dataset Findings

During local replay against run artifacts:

- feature-exposure warnings reproduced immediately from the local
  `per_era_feature_exposure(...)` path
- the main correlation-family paths replayed clean for the representative
  `target_ralph_20` warning run
- at least two FNC reference features were constant within many eras:
  - `feature_stable_shier_stagger`
  - `feature_nephric_unboned_mandalay`

Observed local fact:

- the training feature set for the run was `small`
- the scoring/neutralization feature set was `fncv3_features`

That feature-set mismatch is expected for local diagnostics and is not itself
the bug. The bug is that the current local feature-exposure implementation asks
the correlation routine to evaluate undefined era-feature pairs and only drops
`NaN` after the warning has already been emitted.

## Official Numerai / Public Implementation Findings

Reviewed sources included:

- current Numerai tournament scoring docs
- current `numerai-tools` public code and tests
- staff/community forum posts on dangerous features, diagnostics, and release
  handling of missing/degenerate features
- primary NumPy / pandas / SciPy docs for correlation semantics

What those sources support:

- constant-input correlation is mathematically undefined and should produce
  `NaN`
- official/public Numerai materials are explicit about core formulas and
  submission validation, but not explicit about this exact diagnostic edge case
- Numerai is strict about invalid submission inputs
  - no NaNs
  - non-zero prediction standard deviation
- Numerai/forum guidance around dangerous or degenerate features leans toward
  explicit handling, retraining, exclusion, or deliberate imputation, not
  blanket warning suppression

What those sources do not clearly publish:

- a canonical official policy for per-era feature-exposure correlations that
  become undefined because a feature is constant within that era

## Interpretation

The correct local policy is:

- keep the metric defined against the intended reference feature set
  (`fncv3_features`)
- do not change the semantic meaning of feature exposure by redefining the
  feature universe per era
- do treat constant/degenerate era-feature pairs as undefined local inputs
- exclude those undefined pairs explicitly before aggregation
- preserve observability that exclusion happened

This means:

- the warning is not proof that the metric formula is wrong
- leaving the implementation unchanged keeps noisy logs and implicit behavior
- global warning suppression would hide real future failures in other
  correlation paths
- local prefiltering of invalid era-feature pairs is the smallest correct fix

## Current Code Properties That Support a Small Fix

Relevant current behavior:

- `score_summary(...)` already drops `NaN` before computing summary statistics
- `summarize_scores(...)` already works correctly with partially missing
  per-era values
- the warning source is isolated to `_feature_exposure_stats_for_frame(...)`
- both materialized and era-stream scoring backends call the same
  `per_era_feature_exposure(...)` helper

This means a local fix in the feature-exposure helper should be sufficient and
should not require redesigning the wider scoring contract.

## Recommended Direction

Recommended implementation policy:

- keep `feature_exposure`
- keep `max_feature_exposure`
- keep `fncv3_features` as the scoring feature set
- in feature-exposure scoring only:
  - identify zero-variance or near-zero-variance features within the current
    era frame
  - exclude them before calling `corrwith(...)`
  - if no valid features remain for the era, emit `NaN` for that era
- continue using existing nan-aware summary behavior
- avoid blanket warning suppression

Recommended non-goals:

- do not change official-style metric semantics
- do not silently substitute a numeric value for undefined exposure
- do not broaden changes across unrelated score families unless separately
  justified

## Open Questions

These are policy/contract questions, not blockers to the core fix direction:

- exact-zero only vs near-constant threshold
- whether skipped-feature counts should be persisted in provenance
- whether an era with zero valid exposure features should remain `NaN` or be
  called out explicitly in artifacts/logging
- whether the same hardening pattern should later be applied to future local
  diagnostics built on correlation helpers

## Planned Update Sequence

1. Update `_feature_exposure_stats_for_frame(...)` to prefilter invalid
   era-feature pairs before `corrwith(...)`.
2. Preserve existing `NaN` output when no valid exposure features remain for an
   era.
3. Keep `score_summary(...)` / `summarize_scores(...)` unchanged.
4. Add or update tests covering:
   - constant feature within an era
   - all features invalid within an era
   - materialized scoring path stability
   - era-stream scoring path stability
5. Decide whether to add observability for skipped invalid features in
   provenance or keep the fix silent but deterministic.

## Confidence

Current confidence in this direction: `9.6 / 10`

Reasoning:

- strong support from local replay
- strong support from primary statistical-library behavior
- strong support from Numerai public validation philosophy
- only limited by absence of an official Numerai statement for this exact
  diagnostic edge case
