# viz

- Read [`viz/docs/llms.txt`](./docs/llms.txt) first, then [`viz/docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md).
- `viz/api/numereng_viz` is the authoritative backend package. `viz/api.py` is only the launcher shim.
- Mission control is federated: local store + zero or more SSH remote stores. The UI must never assume `experiment_id` is globally unique.
- Use `/api/experiments/overview` as the mission-control source of truth. Do not rebuild the experiments page by stitching other endpoints together.
- The experiments page must avoid overlapping overview polls. Slow SSH snapshots can make older successful responses look newer if the client allows concurrent refreshes.
- `generated_at` on the merged overview is the canonical page pulse. Do not derive pulse only from experiment timestamps.
- Remote monitoring is read-only. Never sync or share full `.numereng` stores between machines.

## Notes

- Keep source-aware keys in the Svelte experiments page:
  - experiments: `source_kind:source_id:experiment_id`
  - live experiments: `source_kind:source_id:experiment_id`
  - recent activity: include source plus run/job ids
- Remote source `state = live` means the remote snapshot fetch succeeded. It does not imply that the remote currently has live runs.
