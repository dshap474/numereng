# telemetry feature notes

- This slice owns the canonical local run lifecycle control plane: `run_lifecycles` in sqlite plus `runs/<run_id>/runtime.json`.
- Local public training entrypoints must bind launch metadata via `bind_launch_metadata(...)`; missing metadata is a hard failure at lifecycle bootstrap.
- Lifecycle bootstrap is mandatory for local public training and must create canonical run linkage (`run_id`, `run_dir`, `job_id`, `logical_run_id`, `attempt_id`) before data/model work starts.
- After bootstrap, event/log/sample writes remain additive and fail-open so transient telemetry persistence issues do not silently kill training.
- Append-only history lives in `run_job_events`, `run_job_logs`, and `run_job_samples`; current truth lives in `run_lifecycles` + `runtime.json`.
- Cooperative cancel is keyed by `run_id`; local training checks safe checkpoints and exits through canonical `canceled` terminalization.
- Keep status vocabulary aligned with viz expectations: `queued`, `starting`, `running`, `completed`, `failed`, `canceled`, `stale`.
