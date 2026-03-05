# Installation

Set up numereng for local development and CLI usage.

## Prerequisites

- Python 3.11+
- `uv`
- Git
- Optional: Node.js 20+ and npm (only if you run viz frontend)

## Install Dependencies

From repo root:

```bash
uv sync --extra dev
```

## Verify CLI

```bash
uv run numereng
```

This runs bootstrap checks and prints a JSON payload.

## Configure Credentials

Create `.env` from the template and set at least Numerai API auth:

```bash
cp .env.example .env
```

`.env` is local-only and must not be committed.

```dotenv
NUMERAI_PUBLIC_ID=your_public_id
NUMERAI_SECRET_KEY=your_secret_key
```

## Initialize Store

```bash
uv run numereng store init
```

Default store root is `.numereng`.

## Download Dataset Files (Optional Prewarm)

```bash
uv run numereng numerai datasets list
uv run numereng numerai datasets download --filename v5.2/features.json --dest-path .numereng/datasets
uv run numereng numerai datasets download --filename v5.2/train.parquet --dest-path .numereng/datasets
uv run numereng numerai datasets download --filename v5.2/validation.parquet --dest-path .numereng/datasets
```

## Optional Cloud Setup

### EC2

```bash
uv run numereng cloud ec2 init-iam
uv run numereng cloud ec2 setup-data --data-version v5.2
```

### AWS Managed

```bash
uv run numereng cloud aws image build-push --context-dir .
```

### Modal

```bash
uv run numereng cloud modal deploy --ecr-image-uri <registry>/<repository>:<tag>
```

## Optional Viz

```bash
make viz
```

Frontend: `http://127.0.0.1:5173`  
API: `http://127.0.0.1:8502`
