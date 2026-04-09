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

## Default behavior

- Default profile dir: `src/numereng/platform/remotes/profiles`
- Override with env: `NUMERENG_REMOTE_PROFILES_DIR`
- Supported v1 target kind: `ssh`
- Supported remote shells:
  - `posix` for Linux/macOS SSH targets
  - `powershell` for Windows SSH targets
- Default remote Python command: `uv run python`

## How to add one machine

1. Copy `examples/ssh.example.yaml` into `profiles/<id>.yaml`
2. Fill in your SSH host alias or explicit host
3. Set any referenced env vars for username/key path if needed
4. Make sure the remote machine can run:
   - `uv run numereng monitor snapshot --workspace <store_root> --json`

## Remote ops

The tracked remote inventory is also used by the public remote ops surface:

- `uv run numereng remote list`
- `uv run numereng remote bootstrap-viz`
- `uv run numereng remote doctor --target <id>`
- `uv run numereng remote repo sync --target <id>`
- `uv run numereng remote experiment sync --target <id> --experiment-id <id>`
- `uv run numereng remote config push --target <id> --config <path.json>`
- `uv run numereng remote run train --target <id> --config <path.json>`

Sync rules:

- `remote repo sync` mirrors the local git-visible working tree only.
- `remote experiment sync` mirrors experiment authoring files only.
- Do not sync the full `.numereng` store.
- Do not sync `profiles/*.yaml` to remotes.

## Viz bootstrap

- `enabled: true` means the remote participates in both monitoring and viz bootstrap.
- `uv run numereng remote bootstrap-viz` runs repo sync in auto mode and then `remote doctor` for every enabled target.
- `make viz` runs that bootstrap step before starting the local FastAPI + Vite stack.
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
