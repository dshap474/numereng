# Release Publish Checklist

Use this runbook before publishing a new numereng release.

## Preflight

1. Sync packaged docs and shipped skills from canonical sources:
   - `./scripts/sync_packaged_assets.sh`
2. Confirm local secrets are still untracked and out of release artifacts:
   - verify `.env` and similar auth files are ignored
   - verify no secrets are staged in the publish diff
3. Run the fast gate:
   - `just test`
4. Run the install smoke:
   - `./scripts/release_smoke_install.sh`
5. If serving/model-upload changes are included, run the serving release gate:
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

1. Bump `version` in `pyproject.toml`.
2. Confirm the changelog or release notes are ready.
3. Configure pending or active Trusted Publishers on TestPyPI and PyPI for:
   - repo: `dshap474/numereng`
   - workflow: `.github/workflows/release.yml`
   - environments: `testpypi` and `pypi`
   - keep `pypi` environment approval-gated
   - remember a pending publisher does not reserve `numereng` until the first successful trusted publish
4. Build release artifacts:
   - `uv build`
5. Run the GitHub `Release` workflow manually to publish to TestPyPI.
6. Smoke the published package from a clean temp workspace:
   - `uvx --from numereng --default-index https://test.pypi.org/simple --index https://pypi.org/simple numereng init --workspace /tmp/numereng-testpypi-smoke`
7. Push the production tag:
   - `git tag vX.Y.Z`
   - `git push origin vX.Y.Z`
8. Verify the published-user install path against real PyPI:
   - `uvx --from numereng numereng init --workspace /tmp/numereng-pypi-smoke`
   - `uvx --from numereng numereng workspace sync --workspace /tmp/numereng-pypi-smoke --runtime-source pypi`
9. Keep the public share language explicit:
   - community-built
   - self-supported
   - not affiliated with or endorsed by Numerai
10. If serving/model-upload changes shipped, finish the serving release gate before announcing the release.
