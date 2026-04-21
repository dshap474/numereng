# Public Repo Boundary

`numereng` is open-sourced as a repo-clone workspace.

Public support contract:

- clone the repo
- run `uv sync`
- keep runtime state under local gitignored paths like `.numereng/`, `.env`, and real remote profile YAMLs

Non-contract surfaces:

- public PyPI-style distribution
- tracked machine-specific remote targets
- tracked credentials or `.env` files
- tracked generated forum/cache exports

## Retained Corpus Inventory

| Surface | Decision | Rationale | Notes |
| --- | --- | --- | --- |
| `docs/numerai/` | Keep in repo, exclude from build artifacts | Repo-local docs reader depends on a synced upstream mirror | Official docs mirror from `https://github.com/numerai/docs`; mirrored docs remain upstream Numerai content; governed by `docs/numerai/SYNC_POLICY.md`; `docs/numerai/forum/` remains local-only generated output |
| `vendor/example-scripts`, `vendor/numerai-cli`, `vendor/numerai-predict`, `vendor/numerai-tools`, `vendor/numerapi`, `vendor/numerblox`, `vendor/opensignals` | Keep in repo, exclude from build artifacts | Local operator research and compatibility references | Each retained vendor root currently carries an upstream `LICENSE*` file and should stay source-faithful |
| `docs/assets/*.png` | Keep in repo | First-party screenshots for the README and docs | Treat as maintained repo docs assets |
| `.codex/` | Keep in repo for now | Repo-native agent workflow surface and source for shipped skill curation | Audit as normal tracked content; trim in a separate pass if the public operator story changes |

## Local-Only Surfaces

- `.numereng/`
- `.env` and nested `.env.*` files, except `.env.example`
- `src/numereng/platform/remotes/profiles/*.yaml` and `*.yml`
- `docs/numerai/forum/`
- `docs/numerai/.sync-meta.json`
- `viz/*.pid`

## Internal Build Note

Internal wheel builds still exist for cloud package-transfer and serving smoke flows. They are not part of public OSS readiness and must not be treated as the primary distribution contract for `numereng`.
