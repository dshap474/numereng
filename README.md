# numereng

Architecture-standard package scaffold for the `numereng` refactor.

This package currently contains structure, contracts, and quality gates only.
Core runtime behavior from `numereng-old` has not been migrated yet.

## Quick Start

```bash
uv sync --extra dev
make test
make test-all
uv build
```

Optional extras:

- `uv sync --extra training`
- `uv sync --extra mlops` (for MLflow integration)

## AWS Managed Training

The package includes managed AWS cloud commands in addition to EC2 flows:

- `numereng cloud aws image build-push ...`
- `numereng cloud aws train submit ...`
- `numereng cloud aws train status ...`
- `numereng cloud aws train logs ...`
- `numereng cloud aws train cancel ...`
- `numereng cloud aws train pull ...`

Managed image builds expect a Dockerfile in the build context. This repo now provides a root `Dockerfile`
whose entrypoint runs `numereng run train` inside SageMaker and stages required dataset files from S3.

### Prerequisites

1. Install and configure AWS credentials (AWS CLI or shared credentials files).
2. Ensure your principal has access to ECR, S3, SageMaker, CloudWatch Logs, and optionally Batch.
3. Set required env vars:

```bash
export NUMERENG_AWS_REGION=us-east-2
export NUMERENG_S3_BUCKET=numereng-artifacts
export NUMERENG_AWS_ECR_REPOSITORY=numereng-training
export NUMERENG_AWS_SAGEMAKER_ROLE_ARN=arn:aws:iam::<account-id>:role/<role-name>
```

Optional for Batch backend:

```bash
export NUMERENG_AWS_BATCH_JOB_QUEUE=<job-queue-name>
export NUMERENG_AWS_BATCH_JOB_DEFINITION=<job-definition-name>
```

### Example Flow

```bash
numereng cloud aws image build-push --run-id run-123 --context-dir . --state-path .numereng/run-123.cloud.json

numereng cloud aws train submit \
  --run-id run-123 \
  --config configs/train.yaml \
  --image-uri <account>.dkr.ecr.us-east-2.amazonaws.com/numereng-training:run-123 \
  --role-arn "$NUMERENG_AWS_SAGEMAKER_ROLE_ARN" \
  --state-path .numereng/run-123.cloud.json

numereng cloud aws train status --state-path .numereng/run-123.cloud.json
numereng cloud aws train logs --state-path .numereng/run-123.cloud.json --lines 200
numereng cloud aws train pull --state-path .numereng/run-123.cloud.json
```

Credential resolution follows standard AWS SDK behavior (`boto3`): env vars, shared credential files, profiles, then role credentials.

The submitted image must contain the actual training entrypoint logic. Numereng handles job submission, state tracking, metadata indexing, and artifact pullback.

### Optional MLflow Tracking

Install MLflow extra and set:

```bash
export NUMERENG_MLFLOW_ENABLED=1
export NUMERENG_MLFLOW_TRACKING_URI=http://localhost:5000
export NUMERENG_MLFLOW_EXPERIMENT=numereng
```

When enabled, training logs params/metrics/artifacts to MLflow as a best-effort observer; training success does not depend on MLflow availability.

## Viz Dashboard (Ported)

Run the dashboard API + web app from this repo:

```bash
make viz
```

- API: `http://127.0.0.1:8502`
- Web: `http://127.0.0.1:5173`
- Stop servers: `make kill-viz`

Optional store-root override:

```bash
NUMERENG_STORE_ROOT=/abs/path/to/.numereng make viz
```

Phase 1 behavior:

- Read routes are backward-compatible under `/api/*`.
- Write/control routes are intentionally disabled (HTTP `501`).

## Architecture Docs

- Read first: `docs/llms.txt`
- Deep map: `docs/ARCHITECTURE.md`
- Public entrypoints: `src/numereng/api.py`, `src/numereng/cli.py`
- Custom model docs: `docs/CUSTOM_MODELS.md`
- Cloud contract: `dev/CLOUD_AWS_CONTRACT.md`
