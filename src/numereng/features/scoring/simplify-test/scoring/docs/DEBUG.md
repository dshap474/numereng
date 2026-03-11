# Scoring Notes: Benchmark Coverage vs Official Numerai Behavior

## Summary

We investigated repeated post-run scoring failures with:

`training_benchmark_partial_id_overlap:predictions=6038476,benchmark=6033459,overlap=6033459`

The failure is happening in numereng's local post-run scoring path, not in model fitting or artifact persistence.

The key conclusion from the investigation was:

- Era `0157` is valid and expected under official Numerai walk-forward validation.
- The benchmark coverage gap at era `0157` appears to come from benchmark data availability, not from an invalid training split.
- At investigation time, our local scoring policy was stricter than what official Numerai docs explicitly require.

## What Failed

In local post-run scoring, numereng previously required exact `(id, era)` overlap between saved predictions and benchmark predictions.

Current strict enforcement lives in:

- `src/numereng/features/scoring/models.py`
- `src/numereng/features/scoring/metrics.py`

Relevant behavior:

- before this change, default scoring policy set `benchmark_overlap_policy="strict"`
- benchmark scoring preflight calls `validate_join_source_coverage(...)`
- any partial benchmark overlap raises `TrainingDataError`
- the training run itself still completes because post-run scoring failures are caught and serialized into `results.json`

Observed mismatch during failing runs:

- prediction rows: `6,038,476`
- benchmark rows: `6,033,459`
- overlap rows: `6,033,459`
- missing benchmark rows: `5,017`

Those missing rows are entirely from era `0157`.

## Local Dataset Facts

Verified locally:

- `full.parquet` covers era `0157`
- `full_benchmark_models.parquet` starts at era `0158`
- benchmark coverage is otherwise aligned after that point

So the strict local failure is caused by:

- predictions include era `0157`
- benchmark parquet does not include era `0157`
- numereng previously treated that partial overlap as a hard scoring failure

## Official Numerai Findings

Official sources reviewed:

- `https://docs.numer.ai/numerai-tournament/models`
- `https://docs.numer.ai/numerai-tournament/data`
- `https://docs.numer.ai/numerai-tournament/scoring`
- `https://docs.numer.ai/numerai-tournament/scoring/definitions`
- `https://docs.numer.ai/numerai-tournament/scoring/meta-model-contribution-mmc`
- supporting release/forum context:
  - `https://forum.numer.ai/t/v5-atlas-data-release/7576`
  - `https://forum.numer.ai/t/benchmark-models/6754`

Confirmed from official docs:

- Benchmark models are built with walk-forward validation in 156-era chunks.
- For `20D` targets, purge is `8` eras.
- For `60D` targets, purge is `16` eras.
- The documented first example window is:
  - train: `1..148`
  - purge: `149..156`
  - validation: `157..312`
- Historical eras are weekly.
- `20D` and `60D` targets are forward-looking and overlapping.
- `MMC` is contribution relative to the meta model.
- `BMC` is contribution relative to benchmark models.
- BMC is informational and diagnostics-oriented in the official scoring docs context.

## Important Interpretation

Official Numerai docs support the following:

- era `157` is a legitimate validation era under the official benchmark walk-forward setup
- purge windows are the right tool for overlap control
- benchmark-based scoring exists and is meaningful

Official Numerai docs do not clearly state that local historical benchmark diagnostics must hard-fail whenever benchmark coverage is missing for a small portion of rows or eras.

That exact-failure behavior appears to be a numereng policy choice, not a documented official Numerai requirement.

## Conclusion

The bug is not:

- the WFCV schedule
- the presence of era `0157`
- the purge logic itself

The bug is the combination of:

- valid predictions including era `0157`
- benchmark parquet beginning at `0158`
- numereng previously enforcing strict benchmark overlap in local post-run scoring

In short:

`0157` should remain in the training/scoring window, and the benchmark-scoring path should not fail the entire run just because benchmark diagnostics are unavailable for that boundary era.

## Implemented Direction

Keep unchanged:

- official walk-forward split logic
- purge behavior
- prediction generation over era `0157`

Change in numereng:

- local benchmark-based scoring should tolerate partial benchmark overlap when the overlap is otherwise high and aligned
- benchmark metrics should be computed on overlapping rows and eras only
- missing benchmark coverage should not fail the entire run if core metrics remain computable

Reasoning:

- this matches the official walk-forward framing
- it preserves valid OOF predictions
- it avoids throwing away a correct boundary era just to satisfy a local benchmark diagnostic constraint
- it treats benchmark diagnostics as secondary to core train/run success

## Non-Conclusion

We did not confirm from official docs that Numerai itself mandates tolerant overlap for local benchmark diagnostics.

What we confirmed is narrower and sufficient:

- official docs validate era `157`
- official docs validate purge-based WFCV
- official docs do not justify numereng's current strict local benchmark-overlap failure as an official requirement
