# Cloud Training

Numereng supports three cloud paths:

- modular EC2 orchestration
- managed AWS training on SageMaker or Batch
- Modal deployment and remote training

Validate the config locally first:

```bash
numereng run train --config configs/run.json
```

By default, numereng now writes transient cloud state under `.numereng/cache/cloud/<provider>/...` and keeps durable run provenance under `.numereng/runs/<run_id>/run.json -> execution`. Legacy `.numereng/cloud/*.json` state paths remain read-compatible during migration, but new writes should use the cache-backed layout.

## EC2 Workflow

```bash
STATE=.numereng/cache/cloud/aws/runs/run-0001/state.json

uv run numereng cloud ec2 init-iam
uv run numereng cloud ec2 setup-data --data-version v5.2

uv run numereng cloud ec2 provision \
  --run-id run-0001 \
  --state-path "$STATE"

uv run numereng cloud ec2 package build-upload --state-path "$STATE"
uv run numereng cloud ec2 config upload --config configs/run.json --state-path "$STATE"
uv run numereng cloud ec2 push --instance-id i-abc123 --state-path "$STATE"
uv run numereng cloud ec2 install --instance-id i-abc123 --state-path "$STATE"

uv run numereng cloud ec2 train start --instance-id i-abc123 --state-path "$STATE"
uv run numereng cloud ec2 train poll --instance-id i-abc123 --state-path "$STATE"
uv run numereng cloud ec2 pull --instance-id i-abc123 --state-path "$STATE"
uv run numereng cloud ec2 terminate --instance-id i-abc123 --state-path "$STATE"
```

Use `cloud ec2 logs`, `cloud ec2 status`, and `cloud ec2 s3 ls|cp|rm` for monitoring and recovery.

## AWS Managed Workflow

```bash
uv run numereng cloud aws image build-push --context-dir .

uv run numereng cloud aws train submit \
  --backend sagemaker \
  --config configs/run.json

uv run numereng cloud aws train status --backend sagemaker --run-id <run_id>
uv run numereng cloud aws train logs --backend sagemaker --run-id <run_id>
uv run numereng cloud aws train cancel --backend sagemaker --run-id <run_id>
uv run numereng cloud aws train pull --run-id <run_id>
uv run numereng cloud aws train extract --run-id <run_id>
```

Current constraints:

- managed backends are `sagemaker|batch`
- `--spot` and `--on-demand` are mutually exclusive
- `runtime_profile` controls packaging, not the training-config device
- SageMaker resolves a checked-in default ECR image when `--image-uri` is omitted
- default SageMaker aliases are `numereng-training:sagemaker-standard-current` and `numereng-training:sagemaker-lgbm-cuda-current`
- `cloud aws train pull` stages archives under `.numereng/cache/cloud/aws/runs/<run_id>/pull`
- `cloud aws train extract` is the step that unpacks pulled archives into `.numereng/runs/*`, stamps durable execution provenance into `run.json`, and indexes the extracted runs
- if extracted artifacts match an already-materialized run hash, `extract` skips moving files but still refreshes the existing run's durable cloud execution provenance from managed state

Refresh the default SageMaker images with:

```bash
uv run numereng cloud aws image build-push --context-dir . --runtime-profile standard
uv run numereng cloud aws image build-push --context-dir . --runtime-profile lgbm-cuda
```

If you need a one-off image instead of the stable profile alias, pass `--image-tag <tag>` or `--image-uri <uri>` explicitly.

### CUDA LightGBM On SageMaker

```bash
uv run numereng cloud aws image build-push \
  --context-dir . \
  --runtime-profile lgbm-cuda

uv run numereng cloud aws train submit \
  --backend sagemaker \
  --config configs/run-cuda.json \
  --runtime-profile lgbm-cuda \
  --instance-type ml.g5.2xlarge
```

## Modal Workflow

```bash
uv run numereng cloud modal deploy --ecr-image-uri <registry>/<repository>:<tag>
uv run numereng cloud modal data sync --config configs/run.json --volume-name numereng-data
uv run numereng cloud modal train submit --config configs/run.json
uv run numereng cloud modal train status
uv run numereng cloud modal train logs
uv run numereng cloud modal train pull
```

Current constraints:

- `cloud modal deploy` requires a full ECR image URI
- `cloud modal data sync` expects the config-required datasets to exist locally first
- `status|logs|cancel|pull` can resolve persisted call state via `--call-id` or `--state-path`

## State Path Rules

- omit `--state-path` for normal usage when the command already has a stable run/job identity; numereng defaults transient state under `.numereng/cache/cloud/<provider>/...`
- if you pass `--state-path`, reuse the same path across provision, submit, monitor, and pull steps
- explicit state-path overrides must resolve under `<store_root>/cache/cloud/**/*.json`; legacy `<store_root>/cloud/*.json` paths remain accepted during migration
- pulled archives under `cache/cloud` are staging, not durable run storage
- materialized run provenance lives in `.numereng/runs/<run_id>/run.json -> execution`
- prefer explicit `--run-id`, `--instance-id`, `--training-job-name`, or `--batch-job-id` when recovering from partial state
