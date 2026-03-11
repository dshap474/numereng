# Cloud Training

Run training on EC2, AWS managed services, or Modal.

## Local Baseline

Always validate locally first:

```bash
uv run numereng run train --config configs/run.json
```

## EC2 Workflow (Modular)

```bash
# 1) Bootstrap IAM + data staging
uv run numereng cloud ec2 init-iam
uv run numereng cloud ec2 setup-data --data-version v5.2

# 2) Provision + persist state
uv run numereng cloud ec2 provision --run-id run-0001 --state-path .numereng/cloud/run-0001.json

# 3) Upload package/config and prep host
uv run numereng cloud ec2 package build-upload --state-path .numereng/cloud/run-0001.json
uv run numereng cloud ec2 config upload --config configs/run.json --state-path .numereng/cloud/run-0001.json
uv run numereng cloud ec2 push --instance-id i-abc123 --state-path .numereng/cloud/run-0001.json
uv run numereng cloud ec2 install --instance-id i-abc123 --state-path .numereng/cloud/run-0001.json

# 4) Train + collect artifacts
uv run numereng cloud ec2 train start --instance-id i-abc123 --state-path .numereng/cloud/run-0001.json
uv run numereng cloud ec2 train poll --instance-id i-abc123 --state-path .numereng/cloud/run-0001.json
uv run numereng cloud ec2 pull --instance-id i-abc123 --state-path .numereng/cloud/run-0001.json
uv run numereng cloud ec2 terminate --instance-id i-abc123 --state-path .numereng/cloud/run-0001.json
```

## AWS Managed Training (SageMaker/Batch)

```bash
# Build and push image
uv run numereng cloud aws image build-push --context-dir .

# Submit managed training
uv run numereng cloud aws train submit \
  --backend sagemaker \
  --config configs/run.json

# Status/logs/cancel/pull/extract
uv run numereng cloud aws train status --backend sagemaker --run-id <run_id>
uv run numereng cloud aws train logs --backend sagemaker --run-id <run_id>
uv run numereng cloud aws train cancel --backend sagemaker --run-id <run_id>
uv run numereng cloud aws train pull --run-id <run_id>
uv run numereng cloud aws train extract --run-id <run_id>
```

### AWS Constraints

- `cloud aws train submit --backend` supports only `sagemaker|batch`.
- `--spot` and `--on-demand` are mutually exclusive.
- SageMaker CUDA requires a config that resolves to `model.device=cuda`, `cloud aws image build-push --runtime-profile lgbm-cuda`, and a GPU instance type such as `ml.g5.*` or `ml.p*`.
- `runtime_profile` selects packaging only; it does not override the training config device.
- `cloud aws train extract` only promotes `runs/<run_id>/*` artifacts from pulled tarballs.
- Archive extraction hard-fails on unsafe members (absolute paths, traversal, links, invalid run ids) and run-dir hash conflicts.
- Managed `--state-path` must resolve under `<store_root>/cloud/*.json`.

### SageMaker CUDA LightGBM

```bash
# Build the CUDA-specific SageMaker image
uv run numereng cloud aws image build-push \
  --context-dir . \
  --runtime-profile lgbm-cuda

# Submit with a CUDA config and GPU instance type
uv run numereng cloud aws train submit \
  --backend sagemaker \
  --config configs/run-cuda.json \
  --runtime-profile lgbm-cuda \
  --instance-type ml.g5.2xlarge
```

- The standard SageMaker image uses `docker/Dockerfile.sagemaker`.
- The CUDA image uses `docker/Dockerfile.sagemaker-lgbm-cuda`.
- SageMaker submit rejects CUDA configs unless the runtime profile and instance type are CUDA-compatible.
- Batch is not currently accepted for CUDA-configured LightGBM runs.

### EC2 CUDA LightGBM

```bash
# Provision a GPU tier first
uv run numereng cloud ec2 provision --run-id run-0001 --tier gpu --state-path .numereng/cloud/run-0001.json

# Install the CUDA LightGBM runtime on that worker
uv run numereng cloud ec2 install \
  --instance-id i-abc123 \
  --runtime-profile lgbm-cuda \
  --state-path .numereng/cloud/run-0001.json
```

- EC2 CUDA install requires GPU instance state and fails fast on CPU workers.
- The remote install reinstalls `lightgbm` with CUDA build flags instead of relying on the default CPU wheel.

## Modal Workflow

```bash
# Deploy remote function
uv run numereng cloud modal deploy --ecr-image-uri <registry>/<repository>:<tag>

# Sync required dataset files from local .numereng/datasets
uv run numereng cloud modal data sync --config configs/run.json --volume-name numereng-data

# Submit and monitor training
uv run numereng cloud modal train submit --config configs/run.json
uv run numereng cloud modal train status
uv run numereng cloud modal train logs
uv run numereng cloud modal train pull
```

### Modal Constraints

- `cloud modal deploy` requires full ECR URI format `<registry>/<repository>:<tag>`.
- `cloud modal data sync` requires config-required dataset files under local `.numereng/datasets`.
- `cloud modal train status|logs|cancel|pull` can resolve persisted call state via `--call-id` or `--state-path`.
- Modal `--state-path` must resolve to `.numereng/cloud/*.json`.

## Tips

- Use `--state-path` consistently to make recovery and polling deterministic.
- Keep `--state-path` inside `.numereng/cloud/*.json` for all cloud providers.
- Prefer explicit `--run-id`/`--instance-id` arguments when debugging state drift.
