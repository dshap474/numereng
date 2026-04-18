# Cloud Training

Numereng supports three cloud paths:

- modular EC2 orchestration
- managed AWS training on SageMaker or Batch
- Modal deploy/data/train workflows

Use cloud workflows when you want numereng to keep the repo-local runtime model but push execution or packaging to remote infrastructure.

## EC2 Workflow

```bash
STATE=.numereng/cache/cloud/aws/runs/run-0001/state.json

uv run numereng cloud ec2 init-iam
uv run numereng cloud ec2 setup-data --data-version v5.2
uv run numereng cloud ec2 provision --run-id run-0001 --state-path "$STATE"
uv run numereng cloud ec2 package build-upload --state-path "$STATE"
uv run numereng cloud ec2 config upload --config configs/run.json --state-path "$STATE"
uv run numereng cloud ec2 push --instance-id i-abc123 --state-path "$STATE"
uv run numereng cloud ec2 install --instance-id i-abc123 --state-path "$STATE"
uv run numereng cloud ec2 train start --instance-id i-abc123 --state-path "$STATE"
uv run numereng cloud ec2 train poll --instance-id i-abc123 --state-path "$STATE"
uv run numereng cloud ec2 pull --instance-id i-abc123 --state-path "$STATE"
uv run numereng cloud ec2 terminate --instance-id i-abc123 --state-path "$STATE"
```

## Managed AWS Workflow

```bash
uv run numereng cloud aws image build-push --context-dir .

uv run numereng cloud aws train submit \
  --backend sagemaker \
  --config configs/run.json

uv run numereng cloud aws train status --backend sagemaker --run-id <run_id>
uv run numereng cloud aws train logs --backend sagemaker --run-id <run_id>
uv run numereng cloud aws train pull --run-id <run_id>
uv run numereng cloud aws train extract --run-id <run_id>
```

Current constraints:

- managed backends are `sagemaker|batch`
- `--spot` and `--on-demand` are mutually exclusive
- `cloud aws train pull` stages archives under `.numereng/cache/cloud/aws/runs/<run_id>/pull`
- `cloud aws train extract` is the step that materializes the durable local run under `.numereng/runs/<run_id>/`

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

## State Model

Transient cloud state lives under `.numereng/cache/cloud/<provider>/...`.

Durable run provenance still belongs under `.numereng/runs/<run_id>/run.json`.

## Read Next

- [Remote Operations](remote-ops.md)
- [Runtime Artifacts & Paths](../reference/runtime-artifacts.md)
