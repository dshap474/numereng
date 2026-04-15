# Local training failures

## Symptoms

- `numereng run train` fails before or during lifecycle bootstrap
- run directory exists but `metrics.json` or `run.json` is missing
- viz shows stale or failed lifecycle state for a recent local run

## First checks

1. Run `just test` if the failure happened during local development changes.
2. Inspect the run log under `.numereng/runs/<run_id>/run.log`.
3. Inspect `.numereng/runs/<run_id>/runtime.json`.
4. Inspect the current lifecycle row through the CLI/API store tooling.

## Contract reminders

- `run_lifecycles` and `runs/<run_id>/runtime.json` are the current-truth lifecycle surfaces.
- Training requires successful pre-finalization `index_run`.
- Public training entrypoints must bind launch metadata before work starts.

## Common recovery paths

- If lifecycle state looks stale, use the store repair workflow rather than editing artifacts manually.
- If a run is partially written, prefer deterministic cleanup/reset through store tooling before retrying.
- If a config fails validation, fix the JSON config rather than bypassing the loader.

## Escalate when

- multiple runs disagree between filesystem and sqlite state
- scoring artifacts are missing after a supposedly finished run
- cloud-pulled artifacts fail extraction or indexing
