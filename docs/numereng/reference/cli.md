# CLI Commands

This page maps the public `numereng` CLI surface to the workflow pages where each family is explained in detail.

## Working Rule

- run commands from the repo root
- prefer `uv run numereng ...`
- use `--workspace <path>` only when you intentionally want to target another checkout

## Top-Level

- `numereng`
- `numereng --help`
- `numereng docs sync numerai [--workspace <path>]`
- `numereng viz [--workspace <path>] [--host <host>] [--port <port>]`

## Workflow Families

### Runs

- `numereng run train --config <path.json> [...]`
- `numereng run score --run-id <run_id> [...]`
- `numereng run submit --model-name <name> (--run-id <id> | --predictions <path>) [...]`
- `numereng run cancel --run-id <run_id>`

See:
- [Training Models](../workflows/training.md)
- [Submissions](../workflows/submission.md)

### Baselines

- `numereng baseline build --run-ids <id1,id2,...> --name <baseline_name> [--default-target <target>] [--description <text>] [--promote-active] [--workspace <path>]`

See:
- [Baselines & Active Benchmark](../workflows/baselines.md)

### Experiments

- `numereng experiment create --id <id> [--name <text>] [--hypothesis <text>] [--tags <csv>] [--workspace <path>]`
- `numereng experiment list [--status <draft|active|complete|archived>] [--format <table|json>] [--workspace <path>]`
- `numereng experiment details --id <id> [--format <table|json>] [--workspace <path>]`
- `numereng experiment train --id <id> --config <path.json> [...]`
- `numereng experiment run-plan --id <id> --start-index <n> --end-index <n> [...]`
- `numereng experiment score-round --id <id> --round <rN> --stage <post_training_core|post_training_full> [--workspace <path>]`
- `numereng experiment promote --id <id> [--run <run_id>] [--metric <metric_key>] [--workspace <path>]`
- `numereng experiment report --id <id> [--metric <metric_key>] [--limit <n>] [--format <table|json>] [--workspace <path>]`
- `numereng experiment pack --id <id> [--workspace <path>]`
- `numereng experiment archive --id <id> [--workspace <path>]`
- `numereng experiment unarchive --id <id> [--workspace <path>]`

See:
- [Experiments](../workflows/experiments.md)

### Agentic Research

- `numereng research program list [--format <table|json>] [--workspace <path>]`
- `numereng research program show --program <id> [--format <table|json>] [--workspace <path>]`
- `numereng research init --experiment-id <id> --program <id> [--workspace <path>]`
- `numereng research status --experiment-id <id> [--format <table|json>] [--workspace <path>]`
- `numereng research run --experiment-id <id> [--max-rounds <n>] [--max-paths <n>] [--workspace <path>]`

See:
- [Agentic Research](../workflows/agentic-research.md)

### Hyperparameter Optimization

- `numereng hpo create (--study-config <path.json> | (--study-id <id> --study-name <name> --config <path.json> --search-space <json|path>)) [...]`
- `numereng hpo list [--experiment-id <id>] [--status <running|completed|failed>] [--limit <n>] [--offset <n>] [--format <table|json>] [--workspace <path>]`
- `numereng hpo details --study-id <id> [--format <table|json>] [--workspace <path>]`
- `numereng hpo trials --study-id <id> [--format <table|json>] [--workspace <path>]`

See:
- [Hyperparameter Optimization](../workflows/optimization.md)

### Ensembles

- `numereng ensemble build --run-ids <id1,id2,...> [...]`
- `numereng ensemble select --experiment-id <id> [...]`
- `numereng ensemble list [--experiment-id <id>] [--limit <n>] [--offset <n>] [--format <table|json>] [--workspace <path>]`
- `numereng ensemble details --ensemble-id <id> [--format <table|json>] [--workspace <path>]`

See:
- [Ensembles](../workflows/ensembles.md)

### Serving And Model Uploads

- `numereng serve package create --experiment-id <id> --package-id <id> --components <json|path> [...]`
- `numereng serve package inspect --experiment-id <id> --package-id <id> [--workspace <path>]`
- `numereng serve package list [--experiment-id <id>] [--format <table|json>] [--workspace <path>]`
- `numereng serve package score --experiment-id <id> --package-id <id> [...]`
- `numereng serve package sync-diagnostics --experiment-id <id> --package-id <id> [--no-wait] [--workspace <path>]`
- `numereng serve live build --experiment-id <id> --package-id <id> [--workspace <path>]`
- `numereng serve live submit --experiment-id <id> --package-id <id> --model-name <name> [--workspace <path>]`
- `numereng serve pickle build --experiment-id <id> --package-id <id> [--docker-image <image>] [--workspace <path>]`
- `numereng serve pickle upload --experiment-id <id> --package-id <id> --model-name <name> [...]`

See:
- [Serving & Model Uploads](../workflows/serving.md)

### Neutralization

- `numereng neutralize apply (--run-id <id> | --predictions <path>) --neutralizer-path <path> [...]`

See:
- [Features](../concepts/features.md)
- [Submissions](../workflows/submission.md)

### Dataset Tools

- `numereng dataset-tools build-downsampled-full [--data-version <v>] [--data-dir <path>] [--downsample-eras-step <n>] [--downsample-eras-offset <n>] [--rebuild]`

See:
- [Dataset Tools](../workflows/dataset-tools.md)

### Store

- `numereng store init [--workspace <path>]`
- `numereng store index --run-id <run_id> [--workspace <path>]`
- `numereng store rebuild [--workspace <path>]`
- `numereng store doctor [--workspace <path>] [--fix-strays]`
- `numereng store backfill-run-execution (--run-id <run_id> | --all) [--workspace <path>]`
- `numereng store repair-run-lifecycles (--run-id <run_id> | --all) [--workspace <path>]`
- `numereng store materialize-viz-artifacts --kind <scoring-artifacts|per-era-corr> (--run-id <id> | --experiment-id <id> | --all) [--workspace <path>]`

See:
- [Store Operations](../workflows/store-ops.md)

### Monitor And Viz

- `numereng monitor snapshot [--workspace <path>] [--no-refresh-cloud] [--json]`
- `numereng viz [--workspace <path>] [--host <host>] [--port <port>]`

See:
- [Dashboard & Monitor](../workflows/dashboard.md)

### Remote Operations

- `numereng remote list [--format <table|json>]`
- `numereng remote bootstrap-viz [--workspace <path>]`
- `numereng remote doctor --target <id>`
- `numereng remote repo sync --target <id> [--workspace <path>]`
- `numereng remote experiment sync --target <id> --experiment-id <id> [--workspace <path>]`
- `numereng remote experiment launch --target <id> --experiment-id <id> [...]`
- `numereng remote experiment status --target <id> --experiment-id <id> [...]`
- `numereng remote experiment maintain --target <id> --experiment-id <id> [...]`
- `numereng remote experiment stop --target <id> --experiment-id <id> [...]`
- `numereng remote experiment pull --target <id> --experiment-id <id> [--workspace <path>]`
- `numereng remote config push --target <id> --config <path.json> [--workspace <path>]`
- `numereng remote run train --target <id> --config <path.json> [...]`

See:
- [Remote Operations](../workflows/remote-ops.md)

### Cloud

- `numereng cloud ec2 ...`
- `numereng cloud aws ...`
- `numereng cloud modal ...`

See:
- [Cloud Training](../workflows/cloud-training.md)

### Numerai Operations

- `numereng numerai datasets list [...]`
- `numereng numerai datasets download --filename <path> [...]`
- `numereng numerai models [list] [...]`
- `numereng numerai round current [...]`
- `numereng numerai forum scrape [...]`

See:
- [Numerai Operations](../workflows/numerai-ops.md)
