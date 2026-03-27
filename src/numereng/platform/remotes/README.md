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

## How to add one machine

1. Copy `examples/ssh.example.yaml` into `profiles/<id>.yaml`
2. Fill in your SSH host alias or explicit host
3. Set any referenced env vars for username/key path if needed
4. Make sure the remote machine can run:
   - `uv run numereng monitor snapshot --store-root <store_root> --json`

## Notes

- V1 monitoring is read-only and numereng-owned only.
- SSH monitoring polls the remote store through an official numereng CLI entrypoint.
- Do not put raw secrets in YAML. Use env refs such as `user_env` and `identity_file_env`.
