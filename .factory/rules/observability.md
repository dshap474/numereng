# Observability rules

- Preserve the local telemetry lifecycle contract for public training entrypoints
- Treat `run_lifecycles` and `runs/<run_id>/runtime.json` as the canonical current-truth surfaces
- Keep `run_job_events`, `run_job_logs`, and `run_job_samples` append-only history
- Do not move lifecycle ownership into the frontend; progress/state are backend-owned
- Maintain stable correlation fields such as run ID, job ID, experiment ID, launch source, operation type, job type, and terminal reason when touching telemetry paths
- Keep `/healthz` and monitor snapshot flows lightweight and read-only
- Prefer additive observability changes that preserve existing artifact and index contracts

## Telemetry context contract

- Public training entrypoints must bind launch metadata before model or dataset work starts
- Current required launch metadata fields are:
  - `source`
  - `operation_type`
  - `job_type`
- When expanding telemetry context, prefer additive optional fields and preserve backward compatibility with existing sqlite and artifact readers
- Any change to launch metadata semantics, lifecycle state semantics, or monitor snapshot shape should be treated as a contract change and reviewed alongside the repo memory/rules
