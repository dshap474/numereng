# TODO

## Current Snapshot (2026-02-21)
- Built modular `cloud ec2` CLI commands for full lifecycle: `init-iam`, `setup-data`, `provision`, `package build-upload`, `config upload`, `push`, `install`, `train start/poll`, `logs`, `pull`, `terminate`, `s3 ls/cp/rm`.
- Added AWS adapter injection seams to support hermetic testing.
- Added hermetic boto3 `Stubber` tests for EC2/S3/SSM/IAM adapters.
- Expanded cloud API and CLI test coverage.
- Status: cloud-focused tests are green (`118 passed`), full pytest is green (`176 passed`).

## Next
- Fix existing non-cloud strict mypy errors in training modules (`cv`, `metrics`, `service`) so strict type gates pass.
- Decide CI gate strategy: cloud-only type gate now vs full-repo strict gate.
- Add one command-chain smoke test that validates agent-style CLI chaining end-to-end (mocked/hermetic).
- Add concise docs for recommended command chains (train run + artifact pull + cleanup).
