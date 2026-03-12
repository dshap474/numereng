# TODO

## Current Snapshot (2026-03-10)
- 2026-03-10 23:17:03 CDT: Created a tiny SageMaker CUDA LightGBM smoke config at `.numereng/experiments/testing/configs/ender20_small_lgbm_cuda_sagemaker_smoke.json` using `feature_set=small`, `target_ender_20`, `profile=simple`, and `model.device=cuda`.
- 2026-03-10 23:17:03 CDT: Verified SageMaker training quota behavior in `us-east-2`: `ml.g5.2xlarge` training remains unavailable in this account, while `ml.g5.4xlarge` is approved and launchable.
- 2026-03-10 23:17:03 CDT: Confirmed the first `ml.g5.4xlarge` SageMaker attempt failed before container startup because the prior ECR tag lacked a usable `linux/amd64` manifest for SageMaker.
- 2026-03-10 23:17:03 CDT: Researched the container requirement path, validated the CUDA base image manifest, rebuilt the SageMaker CUDA image explicitly for `linux/amd64`, and pushed `699475917808.dkr.ecr.us-east-2.amazonaws.com/numereng-training:testing-ender20-small-cuda-smoke-g54-amd64`.
- 2026-03-10 23:17:03 CDT: Verified the rebuilt ECR tag exposes `linux/amd64`, resubmitted the smoke run on `ml.g5.4xlarge`, and observed the SageMaker job move through `Downloading` -> `Training` -> `Completed`.
- 2026-03-10 23:17:03 CDT: Pulled and extracted the managed outputs locally; the successful managed run is `testing-ender20-small-cuda-smoke-g54-amd64`, SageMaker job `neng-sm-testing-ender20-small-cuda-smoke-g54-amd64-1773201846`, and extracted numereng run ID `8950171a807b`.
- 2026-03-10 23:17:03 CDT: Updated `.numereng/experiments/testing/EXPERIMENT.md` with the failed `g5.2xlarge` quota attempt, the failed bad-manifest `g5.4xlarge` attempt, and the working `amd64 + ml.g5.4xlarge + lgbm-cuda` route.

## Next
- 2026-03-10 23:17:03 CDT: If this SageMaker CUDA smoke path should become standard, promote the working image build and submit commands into durable cloud docs or a scripted smoke target rather than relying on experiment-local notes.
- 2026-03-10 23:17:03 CDT: Decide whether the `linux/amd64` publish requirement should be enforced in the AWS image tooling or runtime profiles to prevent future SageMaker manifest mismatches.
- 2026-03-10 23:17:03 CDT: If desired, publish the documentation-only checkpoint with `github-pr-scoped`.

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
