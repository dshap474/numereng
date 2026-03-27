---
name: telemetry-triage
description: "Diagnose numereng lifecycle, monitor, and telemetry context issues."
---

# Telemetry Triage

Use this skill when a run, experiment, or dashboard issue looks like a telemetry or lifecycle contract problem.

## Canonical sources

- `src/numereng/features/telemetry/context.py`
- `src/numereng/features/telemetry/lifecycle.py`
- `.factory/rules/observability.md`
- `runbooks/local-training-failures.md`
- `runbooks/viz-monitoring.md`

## Checklist

1. Confirm launch metadata was bound by the public entrypoint.
2. Compare `run_lifecycles` against `runs/<run_id>/runtime.json`.
3. Treat `run_job_events`, `run_job_logs`, and `run_job_samples` as history, not current truth.
4. Inspect monitor snapshot or `/healthz` failures before changing frontend code.

## Rule

- Preserve backward compatibility for existing lifecycle readers and persisted monitor artifacts.
