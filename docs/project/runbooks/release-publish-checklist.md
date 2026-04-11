# Release Publish Checklist

Use this runbook before publishing a new numereng release.

## Preflight

1. Sync packaged docs and shipped skills from canonical sources:
   - `./scripts/sync_packaged_assets.sh`
2. Run the fast gate:
   - `just test`
3. Run the install smoke:
   - `./scripts/release_smoke_install.sh`
4. If serving/model-upload changes are included, run the serving release gate:
   - `docs/project/runbooks/serve-release-gate.md`

## Optional Artifact Hygiene Check

Inspect the built wheel for content that should never ship:

```bash
uv build
wheel=$(ls -t dist/numereng-*.whl | head -n 1)
unzip -l "$wheel" | rg 'pc.yaml|forum_scraper_state|docs/numerai'
```

Expected result:
- no machine-local remote profiles
- no forum scraper state
- no packaged `docs/numerai` tree

## Publish

1. Build release artifacts:
   - `uv build`
2. Confirm the changelog or release notes are ready.
3. Publish with the maintainer workflow you normally use.
