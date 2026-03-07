# CLI Commands

Reference for the supported `numereng` CLI surface.

## Top-Level

- `numereng [--fail]`: run bootstrap health checks.
- `numereng --help`: print usage.

## `run`

- `numereng run train --config <path.json> [--output-dir <path>] [--profile <simple|purged_walk_forward|full_history_refit>]`
- `numereng run submit --model-name <name> (--run-id <id> | --predictions <path>) [--store-root <path>] [--tournament <classic|signals|crypto>] [--neutralize --neutralizer-path <path> [--neutralization-proportion <0..1>] [--neutralization-mode <era|global>] [--neutralizer-cols <csv>] [--no-neutralization-rank]]`
- `numereng run score --run-id <id> [--store-root <path>]`

Training profile notes:

- `full_history_refit` is final-fit only and emits no validation metrics.
- `simple` and `purged_walk_forward` persist post-run scoring outputs including `payout_estimate_mean` and `score_provenance.json`.
- `payout_estimate_mean` follows Numerai Classic 2026 payout semantics and is populated only for `target_ender_20`.
- `run score` recomputes metrics from persisted predictions and refreshes `results.json`, `metrics.json`, `score_provenance.json`, and run index rows.

## `experiment`

- `numereng experiment create --id <id> [--name <text>] [--hypothesis <text>] [--tags <csv>] [--store-root <path>]`
- `numereng experiment list [--status <draft|active|complete|archived>] [--format <table|json>] [--store-root <path>]`
- `numereng experiment details --id <id> [--format <table|json>] [--store-root <path>]`
- `numereng experiment train --id <id> --config <path.json> [--output-dir <path>] [--profile <simple|purged_walk_forward|full_history_refit>] [--store-root <path>]`
- `numereng experiment promote --id <id> [--run <run_id>] [--metric <metric_key>] [--store-root <path>]`
- `numereng experiment report --id <id> [--metric <metric_key>] [--limit <n>] [--format <table|json>] [--store-root <path>]`

Experiment training uses the same profile semantics as `run train`: canonical profiles are `simple|purged_walk_forward|full_history_refit`, and `full_history_refit` is refit-only with no validation metrics.

## `hpo`

- `numereng hpo create (--study-config <path.json> | (--study-name <name> --config <path.json>)) [--experiment-id <id>] [--metric <metric_key>] [--direction <maximize|minimize>] [--n-trials <n>] [--sampler <tpe|random>] [--seed <n>] [--search-space <json|path>] [--neutralize --neutralizer-path <path> [--neutralization-proportion <0..1>] [--neutralization-mode <era|global>] [--neutralizer-cols <csv>] [--no-neutralization-rank]] [--store-root <path>]`
- `numereng hpo list [--experiment-id <id>] [--status <running|completed|failed>] [--limit <n>] [--offset <n>] [--format <table|json>] [--store-root <path>]`
- `numereng hpo details --study-id <id> [--format <table|json>] [--store-root <path>]`
- `numereng hpo trials --study-id <id> [--format <table|json>] [--store-root <path>]`

## `ensemble`

- `numereng ensemble build --run-ids <id1,id2,...> [--experiment-id <id>] [--method <rank_avg>] [--metric <metric_key>] [--target <target_col>] [--name <text>] [--ensemble-id <id>] [--weights <w1,w2,...>] [--optimize-weights] [--include-heavy-artifacts] [--selection-note <text>] [--regime-buckets <n>] [--neutralize-members] [--neutralize-final] [--neutralizer-path <path>] [--neutralization-proportion <0..1>] [--neutralization-mode <era|global>] [--neutralizer-cols <csv>] [--no-neutralization-rank] [--store-root <path>]`
- `numereng ensemble list [--experiment-id <id>] [--limit <n>] [--offset <n>] [--format <table|json>] [--store-root <path>]`
- `numereng ensemble details --ensemble-id <id> [--format <table|json>] [--store-root <path>]`

## `neutralize`

- `numereng neutralize apply (--run-id <id> | --predictions <path>) --neutralizer-path <path> [--neutralization-proportion <0..1>] [--neutralization-mode <era|global>] [--neutralizer-cols <csv>] [--output-path <path>] [--no-neutralization-rank] [--store-root <path>]`

## `dataset-tools`

- `numereng dataset-tools build-full-datasets [--data-version <v>] [--data-dir <path>] [--downsample-eras-step <n>] [--downsample-eras-offset <n>] [--skip-downsample] [--rebuild]`

## `store`

- `numereng store init [--store-root <path>]`
- `numereng store index --run-id <id> [--store-root <path>]`
- `numereng store rebuild [--store-root <path>]`
- `numereng store doctor [--store-root <path>] [--fix-strays]`

## `cloud`

### EC2

- `cloud ec2 init-iam`
- `cloud ec2 setup-data`
- `cloud ec2 provision`
- `cloud ec2 package build-upload`
- `cloud ec2 config upload`
- `cloud ec2 push`
- `cloud ec2 install`
- `cloud ec2 train start|poll`
- `cloud ec2 logs`
- `cloud ec2 pull`
- `cloud ec2 terminate`
- `cloud ec2 status`
- `cloud ec2 s3 ls|cp|rm`

### AWS Managed (SageMaker/Batch)

- `cloud aws image build-push`
- `cloud aws train submit|status|logs|cancel|pull`

### Modal

- `cloud modal deploy`
- `cloud modal data sync`
- `cloud modal train submit|status|logs|cancel|pull`

## `numerai`

- `numereng numerai datasets list [--round <num>] [--tournament <classic|signals|crypto>]`
- `numereng numerai datasets download --filename <path> [--dest-path <path>] [--round <num>] [--tournament <classic|signals|crypto>]`
- `numereng numerai models [list] [--tournament <classic|signals|crypto>]`
- `numereng numerai round current [--tournament <classic|signals|crypto>]`

## Exit Codes

- `0`: success/help
- `1`: runtime/boundary error
- `2`: parse/usage error

## Removed Surface

These legacy families are no longer supported:

- `orchestrator`, `optimize`, `predict`, `submission submit`, `pipeline`, `sync-live`, `neutralize-sweep`, `download`, `status`, `runpod`, `vast`, and `db` commands.
