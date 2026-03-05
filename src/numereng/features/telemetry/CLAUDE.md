# telemetry feature notes

- Local telemetry is additive and fail-open: training must continue if telemetry persistence fails.
- This slice writes run-operation metadata into SQLite tables consumed by `features.viz`.
- Launch metadata is opt-in via `bind_launch_metadata(...)`; absent metadata means no telemetry session for that run.
- Keep status vocabulary aligned with viz expectations: `queued`, `starting`, `running`, `completed`, `failed`, `canceled`, `stale`.
