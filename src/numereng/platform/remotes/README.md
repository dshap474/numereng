# Remote Monitoring Profiles

Remote machine inventory for numereng lives here.

Tracked files in this folder define:
- the YAML schema shape
- loader behavior
- examples
- local-user instructions

Real machine profiles belong in:
- `src/numereng/platform/remotes/profiles/*.yaml`

Those profile files are gitignored so each clone stays clean for OSS users.

## Topology modes

Numereng supports two first-class deployment shapes. Neither is a code fork — they differ only in whether you populate `profiles/`.

- **Single-machine** (default, zero profiles) — training, storage, and viz all run on one host reading `.numereng/` directly. No SSH, no config. This is how a fresh clone behaves.
- **Distributed** (one or more profiles) — training + storage live on a remote compute host (workstation, GPU box, etc.) and viz runs locally as a federated read-only dashboard that SSHes into each enabled profile. Local `.numereng/` still exists but is typically empty or thin; each remote source is keyed by `source_kind:source_id:experiment_id` in the UI so nothing collides with local rows.

Do not mix the modes by syncing a remote's full `.numereng/` back to local — the federation is the intended bridge.

## Default behavior

- Default profile dir: `src/numereng/platform/remotes/profiles`
- Override with env: `NUMERENG_REMOTE_PROFILES_DIR`
- Supported v1 target kind: `ssh`
- Supported remote shells:
  - `posix` for Linux/macOS SSH targets
  - `powershell` for Windows SSH targets
- Default remote Python command: `uv run python`

## How to add one machine

1. Copy the right template into `profiles/<id>.yaml`:
   - `examples/ssh.example.yaml` for Linux/macOS remotes (`shell: posix`)
   - `examples/windows.example.yaml` for Windows remotes (`shell: powershell`)
2. Fill in your SSH host alias or explicit host
3. Set any referenced env vars for username/key path if needed
4. Make sure the remote machine can run:
   - `numereng monitor snapshot --workspace <workspace_root> --json`
   - This implies a working `uv`-managed environment on the remote. First invocation installs deps, so bump `command_timeout_seconds` on the profile (default 15) or run `uv sync` on the remote once before relying on the viz probe.

## Remote ops

The tracked remote inventory is also used by the public remote ops surface:

- `numereng remote list`
- `numereng remote bootstrap-viz`
- `numereng remote doctor --target <id>`
- `numereng remote repo sync --target <id>`
- `numereng remote experiment sync --target <id> --experiment-id <id>`
- `numereng remote config push --target <id> --config <path.json>`
- `numereng remote run train --target <id> --config <path.json>`

Sync rules:

- `remote repo sync` mirrors the local git-visible working tree only.
- `remote experiment sync` mirrors experiment authoring files only.
- Do not sync the full `.numereng` store.
- Do not sync `profiles/*.yaml` to remotes.

## Viz bootstrap

- `enabled: true` means the remote participates in both monitoring and viz bootstrap.
- `numereng remote bootstrap-viz` runs repo sync in auto mode and then `remote doctor` for every enabled target.
- Installed workspaces launch the dashboard with `numereng viz`.
- Remotes stay read-only from the dashboard point of view: no remote viz server is started on the target host.
- If one remote is unreachable, local viz still starts and mission control marks that source as unavailable/degraded.

## Shell selection

- Linux/macOS remotes should use `shell: posix`
- Windows remotes should use `shell: powershell`
- Remote ops reuse the same shell selection for sync and detached launch helpers.
- The SSH monitor always stays read-only; it only runs the remote numereng snapshot command.

## Notes

- V1 monitoring is read-only and numereng-owned only.
- SSH monitoring polls the remote store through an official numereng CLI entrypoint.
- Do not put raw secrets in YAML. Use env refs such as `user_env` and `identity_file_env`.
