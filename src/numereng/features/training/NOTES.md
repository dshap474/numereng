# Training Pipeline Debug Notes (2026-02-23)

## Context
- Goal: move to lazy-load-friendly training behavior without benchmark-model failures and lower peak RAM.
- Initial failure looked like a lazy pipeline issue but was traced to benchmark data coverage alignment.

## Root Cause Found
- The pipeline was attempting benchmark joins in a way that blocked runs when benchmark coverage did not fully align with prediction IDs/eras.
- Failing symptom observed previously:
  - `training_benchmark_no_overlapping_ids` / `training_benchmark_partial_id_overlap`
- This was primarily a config/data coverage mismatch, not a core lazy-loading bug.

## Key Decisions
- Benchmark models are now **metrics-only** in training workflows.
- Benchmark models are **not** used as training features (`X`) anymore.
- Metrics scoring is separated into a **post-run phase** (after predictions/artifacts are persisted) to reduce peak training RAM.
- Current scoring policy is **tolerant partial overlap** for benchmark metrics:
  - compute on overlapping `(id, era)` rows
  - fail on zero overlap or era misalignment

## What Changed
- Removed benchmark feature usage from training feature-group behavior.
- Added post-run scoring flow in training service:
  - train + save predictions first
  - finalize run artifacts
  - then score metrics from saved predictions parquet
  - rewrite results/metrics with final scoring outputs
- Moved scoring internals into a dedicated module boundary:
  - `src/numereng/features/training/scoring/service.py` (orchestration)
  - `src/numereng/features/training/scoring/metrics.py` (metric engines)
  - `src/numereng/features/training/metrics.py` now acts as a compatibility re-export shim
- Updated metric-time benchmark join to allow partial overlap in scoring paths.
- Fixed `era_stream` scoring to handle sparse overlap by chunk:
  - do not fail a chunk when benchmark/meta overlap is zero for that chunk
  - aggregate scores on overlapping chunks
  - fail only when global overlap is zero for benchmark or meta model
- Added run-local live training log output:
  - each run writes `runs/<run_id>/run.log` during execution
  - includes stage updates, run start/completion, and failure/error markers
  - `run.json` now records `artifacts.log = "run.log"`
  - log writing is fail-open (never blocks training)

## Validation Performed
- Unit tests (training service + metrics) passed.
- `ender_medium_official` rerun completed successfully:
  - Run ID: `59db908f1b4f`
  - Status: `FINISHED`
  - Artifacts include predictions, results, metrics, score provenance.
- Sequential experiment validation runs:
  - `335d56fbd245` (`ender_small_official_60d.json`): FINISHED, materialized scoring OK.
  - `1415588606dd` (`ender_medium_official_era_stream.json`): FINISHED, era-stream scoring OK after fix.
  - `f07b4c17d2bd` (`ender_small_official_60d_era_stream.json`): FINISHED, era-stream scoring OK after fix.

## Current Overlap State (Run `59db908f1b4f`)
- Predictions eras: `0157..1202` (`6,038,476` rows)
- Benchmark eras: `0158..1202` (`6,033,459` rows)
- Missing benchmark coverage only for era `0157` (`5,017` rows)
- Effective benchmark-metrics overlap: very high, with one missing era.

## Current Operating Guidance
- Keep benchmark as metrics-only.
- Keep post-run scoring for memory efficiency.
- Keep tolerant overlap if the objective is robust execution on near-complete official coverage.
- If strict official-style enforcement is preferred, add strict mode to fail on any partial overlap.

## Follow-Up Candidates
- Add preflight coverage validation to surface overlap issues before fold execution.
- Optional strict/tolerant toggle for benchmark metric joins.

---

# RAM Optimization Update (2026-03-02)

## Implemented This Round
- Materialized scoring now avoids duplicate full predictions reads:
  - predictions file is read once and reused for corr/FNC/MMC/CWMM/BMC setup.
- Benchmark/meta loads now use projected reads (only required columns), instead of loading full tables.
- CSV column discovery now uses header-only reads (`nrows=0`) to avoid materializing full csv files for schema checks.
- Training service now drops more large references before heavy phases:
  - before post-run scoring (non-full-history path)
  - before finalize/index steps (all modes)
  - includes `full`, lazy profile frame, joins/baseline refs, loader refs, and prediction frame handles.
- CV serial path now processes folds incrementally (no intermediate `fold_results` list in serial mode), reducing peak fold aggregation memory.

## Validation
- Unit tests passed:
  - `tests/unit/numereng/features/training/test_metrics.py`
  - `tests/unit/numereng/features/training/test_cv.py`
  - `tests/unit/numereng/features/training/test_service.py`
- Integration RAM-efficiency suite passed:
  - `tests/integration/test_training_ram_efficiency.py`

## Next RAM Improvement Notes (Shortlist)
- Add optional fold-level spill-to-disk for OOF assembly when fold count/size is large.
- Replace fold-lazy profile-frame materialization with metadata/row-count + era-only scan path where possible.
- Add optional dtype downcast pass for non-feature numeric helper columns after load.
- Add instrumentation snapshots around phase boundaries (pre-train, pre-score, post-score) to track RSS deltas per run.
