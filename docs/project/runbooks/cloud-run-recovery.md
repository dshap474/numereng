# Cloud run recovery

## Symptoms

- managed AWS or Modal runs do not reconcile back into the local store
- pulled artifacts exist under `.numereng/cloud/` but do not appear under `.numereng/runs/`
- cloud state JSON exists but the CLI/API cannot resume or inspect the job cleanly

## First checks

1. Confirm the state file path is under `.numereng/cloud/*.json`.
2. Inspect the pull/extract step separately; download and extraction are different contracts.
3. Inspect warnings about skipped archive members or unsafe keys.
4. Re-index extracted runs through store tooling after successful extraction.

## Contract reminders

- `cloud aws train pull` is download-only.
- `cloud aws train extract` is the step that materializes `runs/<run_id>/*`.
- Extraction must reject unsafe tar members and invalid run identities.
- Cloud runtime profiles affect packaging, not the training device contract.

## Common recovery paths

- Re-run extraction only after confirming the pulled archive is the expected one.
- If extraction fails on safety checks, fix the artifact source instead of bypassing the checks.
- If the store is stale after valid extraction, use store indexing/repair workflows rather than manual sqlite edits.
