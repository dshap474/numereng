---
name: numerai-submissions-ops
description: Manage numereng submitted-model snapshots under .numereng/submissions. Use when Codex needs to inspect or refresh an individual submitted model's identity, upload provenance, live-round parquet, submission metadata, or dashboard visibility. Use numerai-live-calibration-sync instead for local-versus-live calibration analysis or research-memory integration.
---

# Submissions Ops

Use this skill for submitted-model snapshot mechanics. Use `numerai-live-calibration-sync` for calibration refresh and interpretation, `numerai-api-ops` for account or write operations, and `experiment-ops` for experiment/package lineage.

Run from the numereng repo root.

## Operating Contract

- Store each submitted model at `.numereng/submissions/<model_name>/`.
- Keep the structure flat; do not add strategy, calibration, benchmark, or nested model folders.
- Treat a model slot, experiment, run, package, hosted pickle, and live-round observation as distinct objects.
- Treat local submission artifacts as snapshots; Numerai remains the live source of truth.
- Never perform a Numerai write without confirmation of the exact operation.

## Artifact Contract

Each model folder contains only:

```text
.numereng/submissions/<model_name>/
  submission.json
  live_rounds.parquet
```

`submission.json` stores identity, experiment/package/upload provenance, offline snapshot, latest live summary, refresh metadata, and warnings.

`live_rounds.parquet` stores one row per live round/score observation, including round state, close and resolve dates, payout fields, multipliers, live metrics, source, pull timestamp, and estimate flags when available.

Do not duplicate the round table into markdown or JSON unless the user explicitly requests an export.

## Workflow

1. Resolve model name and Numerai model ID from `submission.json` or `numerai-api-ops`.
2. Verify available lineage: experiment ID, package ID, upload ID, and hosted-pickle label.
3. Refresh the requested model snapshot through the supported submissions command or Numerai read API.
4. Write only `submission.json` and `live_rounds.parquet`.
5. Verify row counts, latest scored/resolved rounds, pull timestamp, and dashboard discovery.

For multi-model calibration, correlation, lane comparisons, resolved-round gating, or memory updates, stop and invoke `numerai-live-calibration-sync`.

## Safety

- Do not create slots, upload or assign pickles, trigger compute, submit predictions, replace models, or stake from this skill.
- Keep offline metrics separate from live scores.
- Do not overwrite source experiment or package artifacts.
- Never store credentials or private keys in submission artifacts.

## Output

Report:

- model name and model ID;
- API source used;
- lineage fields present or missing;
- files updated;
- live row count and latest scored/resolved rounds;
- pull timestamp and warnings.

For write intent, route to `numerai-api-ops` and state the exact confirmation required.
