# Feature Exposure NaN / Divide Warnings

## Issue

Local post-run scoring emitted repeated warnings of the form:

`RuntimeWarning: invalid value encountered in divide`

The warnings were traced to local feature-exposure diagnostics, not model
training and not the core CORR/BMC/MMC score families for the representative
warning runs.

## Root Cause

The warning source is:

- `src/numereng/features/scoring/metrics.py`
- `_feature_exposure_stats_for_frame(...)`

The previous implementation:

1. rank-normalized the prediction series
2. rank-normalized all configured FNC reference features for the era
3. called `corrwith(...)` between the ranked prediction and the ranked feature
   matrix

If a feature was constant within the era, that correlation was mathematically
undefined. Pandas/NumPy then emitted the divide warning and returned `NaN`.

## Local Findings

Confirmed locally during investigation:

- the warning storm reproduced from the local `feature_exposure` path
- the main correlation-family paths replayed clean for the representative
  `target_ralph_20` / `target_rowan_20` warning runs
- some FNC reference features were constant within many eras
- the run training feature set being `small` while diagnostics used
  `fncv3_features` was expected and not itself the bug

Important distinction:

- using `fncv3_features` for diagnostics is intentional local scoring policy
- the bug was allowing undefined era-feature pairs into `corrwith(...)`

## Public / Official Context

Reviewed during investigation:

- current Numerai scoring docs
- current public `numerai-tools` code/tests
- staff/community forum posts about dangerous features, feature exposure, and
  release-specific missing/degenerate feature handling
- primary NumPy / pandas / SciPy docs

Supported conclusions:

- constant-input correlation is mathematically undefined and should yield `NaN`
- Numerai is strict about invalid submission inputs
- public Numerai materials are not explicit about this exact diagnostic edge
  case
- Numerai/community guidance around dangerous or degenerate features leans
  toward explicit handling, retraining, exclusion, or deliberate imputation,
  not blanket warning suppression

## Implemented Local Policy

Chosen local policy for this issue:

- keep the metric defined against the canonical scoring feature set
  (`fncv3_features`)
- do not silently change the metric semantics by swapping to an ad hoc per-era
  feature universe
- do explicitly exclude zero-variance era-feature pairs before correlation
- if no valid features remain for an era, preserve `NaN` for that era
- keep existing nan-aware score summarization behavior
- do not globally suppress warnings

This policy is intentionally reversible if Numerai later recommends a different
approach.

## Code Change

The local hardening for this issue is intentionally narrow:

- update `_feature_exposure_stats_for_frame(...)`
- after rank-normalizing features, remove feature columns whose ranked
  per-era standard deviation is zero
- compute exposure only on remaining valid feature columns
- preserve `NaN` output if the filtered feature set is empty

No broader scoring-family redesign is part of this issue.

## Why This Change

Why not do nothing:

- leaves warning spam in logs
- keeps undefined-pair handling implicit
- makes it harder to notice future real correlation problems

Why not globally suppress warnings:

- hides unrelated failures in other correlation paths
- does not improve metric semantics

Why this fix:

- smallest local change
- preserves intended metric meaning
- matches expected statistical behavior
- works with existing nan-aware summary behavior
- easy to revert if external guidance changes

## Tests Added

This issue should remain covered by targeted unit tests for:

- mixed valid + constant features within an era
- all features constant within an era
- existing normal feature-exposure behavior unchanged

## Follow-Up

If Numerai confirms a different preferred policy, this change can be revised or
reverted cleanly because:

- the implementation change is localized to one helper
- the behavior change is documented here
- no broader scoring contract changes are required for this issue
