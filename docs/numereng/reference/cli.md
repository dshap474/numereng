# CLI Commands

Reference for the supported `numereng` CLI surface.

## Top-Level

- `numereng [--fail]`
- `numereng --help`

## `run`

- `numereng run train --config <path.json> [--output-dir <path>] [--profile <simple|purged_walk_forward|full_history_refit>] [--experiment-id <id>] [--post-training-scoring <none|core|full|round_core|round_full>]`
- `numereng run score --run-id <id> [--stage <all|run_metric_series|post_fold|post_training_core|post_training_full>] [--store-root <path>]`
- `numereng run submit --model-name <name> (--run-id <id> | --predictions <path>) [--store-root <path>] [--tournament <classic|signals|crypto>] [--allow-non-live-artifact] [--neutralize --neutralizer-path <path> [--neutralization-proportion <0..1>] [--neutralization-mode <era|global>] [--neutralizer-cols <csv>] [--no-neutralization-rank]]`

Notes:

- `full_history_refit` is final-fit only and emits no validation metrics
- `--post-training-scoring` overrides the config `training.post_training_scoring` policy for that run
- `round_core` and `round_full` are accepted at parse time but rejected at runtime for `run train`; they require the experiment workflow
- `run score` recomputes metrics from saved predictions and refreshes `results.json`, `metrics.json`, `score_provenance.json`, and store index rows
- `run score` also refreshes canonical scoring artifacts under `artifacts/scoring/` and updates `artifacts/scoring/manifest.json`

## `experiment`

- `numereng experiment create --id <YYYY-MM-DD_slug> [--name <text>] [--hypothesis <text>] [--tags <csv>] [--store-root <path>]`
- `numereng experiment list [--status <draft|active|complete|archived>] [--format <table|json>] [--store-root <path>]`
- `numereng experiment details --id <id> [--format <table|json>] [--store-root <path>]`
- `numereng experiment archive --id <id> [--store-root <path>]`
- `numereng experiment unarchive --id <id> [--store-root <path>]`
- `numereng experiment train --id <id> --config <path.json> [--output-dir <path>] [--profile <simple|purged_walk_forward|full_history_refit>] [--post-training-scoring <none|core|full|round_core|round_full>] [--store-root <path>]`
- `numereng experiment score-round --id <id> --round <rN> --stage <post_training_core|post_training_full> [--store-root <path>]`
- `numereng experiment promote --id <id> [--run <run_id>] [--metric <metric_key>] [--store-root <path>]`
- `numereng experiment report --id <id> [--metric <metric_key>] [--limit <n>] [--format <table|json>] [--store-root <path>]`
- `numereng experiment pack --id <id> [--store-root <path>]`

Notes:

- `experiment create` scaffolds `.numereng/experiments/<id>/` with `experiment.json`, a rich `EXPERIMENT.md` skeleton, `configs/`, a header-only `run_plan.csv`, and `run_scripts/launch_all.py|.sh|.ps1`
- archived experiments are hidden from normal experiment and config catalogs but still readable by direct experiment ID
- `archive` moves `.numereng/experiments/<id>` to `.numereng/experiments/_archive/<id>` and updates indexed experiment status to `archived`
- `unarchive` moves the directory back to the live root and restores the pre-archive status when recorded
- archived experiments are read-only; `experiment train` and `experiment promote` hard-fail until the experiment is unarchived
- `--post-training-scoring` overrides the config policy for that experiment run
- `round_core` and `round_full` require an `rN_*` config filename because the round label is derived from the config stem
- `pack` writes `.numereng/experiments/<id>/EXPERIMENT.pack.md` with the current `EXPERIMENT.md` body plus one dashboard-aligned scalar metrics table across manifest-listed runs

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

- `numereng dataset-tools build-downsampled-full [--data-version <v>] [--data-dir <path>] [--downsample-eras-step <n>] [--downsample-eras-offset <n>] [--rebuild]`

## `store`

- `numereng store init [--store-root <path>]`
- `numereng store index --run-id <id> [--store-root <path>]`
- `numereng store rebuild [--store-root <path>]`
- `numereng store doctor [--store-root <path>] [--fix-strays]`
- `numereng store materialize-viz-artifacts --kind <per-era-corr|scoring-artifacts> (--run-id <id> | --experiment-id <id> | --all) [--store-root <path>]`

Notes:

- `store doctor --fix-strays` also runs conservative tmp cleanup for `.numereng/tmp/remote-configs/*.json`; it only deletes files older than 30 days that are not referenced by active run lifecycles
- if the store DB is missing or unreadable, tmp remote-config cleanup is skipped rather than guessed
- `materialize-viz-artifacts` backfills persisted viz artifacts for historical runs without retraining
- `--kind scoring-artifacts` is the canonical hard-refactor path and rescoring persists the full `artifacts/scoring/` bundle plus `manifest.json`
- `--kind per-era-corr` is accepted as a compatibility label for the same rescoring/backfill path, but the canonical outputs are the new `artifacts/scoring/*` parquet artifacts
- the command is idempotent and refreshes run-manifest artifact links when it materializes missing scoring artifacts

## `cloud`

### `cloud ec2`

- `numereng cloud ec2 init-iam [--region <region>] [--bucket <bucket>] [--role-name <name>] [--security-group-name <name>]`
- `numereng cloud ec2 setup-data --data-version <v> [--cache-dir <path>] [--region <region>] [--bucket <bucket>]`
- `numereng cloud ec2 provision --run-id <id> [--tier <tier>] [--spot|--on-demand] [--region <region>] [--bucket <bucket>] [--state-path <path>]`
- `numereng cloud ec2 package build-upload [--run-id <id>] [--region <region>] [--bucket <bucket>] [--state-path <path>]`
- `numereng cloud ec2 config upload --config <path.json> [--run-id <id>] [--region <region>] [--bucket <bucket>] [--state-path <path>]`
- `numereng cloud ec2 push --instance-id <id> [--run-id <id>] [--region <region>] [--bucket <bucket>] [--state-path <path>]`
- `numereng cloud ec2 install --instance-id <id> [--run-id <id>] [--region <region>] [--runtime-profile <standard|lgbm-cuda>] [--state-path <path>]`
- `numereng cloud ec2 train start --instance-id <id> [--run-id <id>] [--region <region>] [--state-path <path>]`
- `numereng cloud ec2 train poll --instance-id <id> [--run-id <id>] [--timeout-seconds <n>] [--interval-seconds <n>] [--region <region>] [--state-path <path>]`
- `numereng cloud ec2 logs --instance-id <id> [--lines <n>] [--follow] [--region <region>] [--state-path <path>]`
- `numereng cloud ec2 pull --instance-id <id> [--run-id <id>] [--output-dir <path>] [--region <region>] [--bucket <bucket>] [--state-path <path>]`
- `numereng cloud ec2 terminate --instance-id <id> [--region <region>] [--state-path <path>]`
- `numereng cloud ec2 status [--run-id <id>] [--region <region>] [--state-path <path>]`
- `numereng cloud ec2 s3 ls --prefix <prefix> [--region <region>] [--bucket <bucket>]`
- `numereng cloud ec2 s3 cp --src <path|s3://...> --dst <path|s3://...> [--region <region>] [--bucket <bucket>]`
- `numereng cloud ec2 s3 rm --uri <s3://...> [--recursive] [--region <region>] [--bucket <bucket>]`

### `cloud aws`

- `numereng cloud aws image build-push [--run-id <id>] [--context-dir <path>] [--dockerfile <path>] [--runtime-profile <standard|lgbm-cuda>] [--repository <name>] [--image-tag <tag>] [--platform <value>] [--region <region>] [--bucket <bucket>] [--state-path <path>] [--store-root <path>]`
- `numereng cloud aws train submit [--run-id <id>] [--backend <sagemaker|batch>] [--config <path.json>] [--config-s3-uri <s3://...json>] [--image-uri <uri>] [--runtime-profile <standard|lgbm-cuda>] [--role-arn <arn>] [--instance-type <name>] [--instance-count <n>] [--volume-size-gb <n>] [--max-runtime-seconds <n>] [--max-wait-seconds <n>] [--spot|--on-demand] [--checkpoint-s3-uri <s3://...>] [--output-s3-uri <s3://...>] [--batch-job-queue <name>] [--batch-job-definition <name>] [--region <region>] [--bucket <bucket>] [--state-path <path>] [--store-root <path>]`
- `numereng cloud aws train status [--backend <sagemaker|batch>] [--run-id <id>] [--training-job-name <name>] [--batch-job-id <id>] [--region <region>] [--state-path <path>] [--store-root <path>]`
- `numereng cloud aws train logs [--backend <sagemaker|batch>] [--run-id <id>] [--training-job-name <name>] [--batch-job-id <id>] [--lines <n>] [--follow] [--region <region>] [--state-path <path>] [--store-root <path>]`
- `numereng cloud aws train cancel [--backend <sagemaker|batch>] [--run-id <id>] [--training-job-name <name>] [--batch-job-id <id>] [--region <region>] [--state-path <path>] [--store-root <path>]`
- `numereng cloud aws train pull [--run-id <id>] [--output-s3-uri <s3://...>] [--output-dir <path>] [--region <region>] [--bucket <bucket>] [--state-path <path>]`
- `numereng cloud aws train extract [--run-id <id>] [--output-dir <path>] [--region <region>] [--bucket <bucket>] [--state-path <path>]`

Notes:

- `cloud aws train submit --backend sagemaker` auto-resolves a checked-in default image when `--image-uri` is omitted
- current SageMaker default aliases are `numereng-training:sagemaker-standard-current` and `numereng-training:sagemaker-lgbm-cuda-current`
- `cloud aws image build-push` refreshes that profile alias when `--image-tag` is omitted
- Batch still requires an explicit `--image-uri`

### `cloud modal`

- `numereng cloud modal deploy --ecr-image-uri <uri:tag> [--app-name <name>] [--function-name <name>] [--environment-name <name>] [--aws-profile <name>] [--timeout-seconds <n>] [--gpu <value>] [--cpu <n>] [--memory-mb <n>] [--data-volume-name <name>] [--metadata <k=v,...>] [--state-path <path>]`
- `numereng cloud modal data sync --config <path.json> --volume-name <name> [--force] [--no-create-if-missing] [--metadata <k=v,...>] [--state-path <path>]`
- `numereng cloud modal train submit --config <path.json> [--output-dir <path>] [--profile <simple|purged_walk_forward|full_history_refit>] [--app-name <name>] [--function-name <name>] [--environment-name <name>] [--metadata <k=v,...>] [--state-path <path>]`
- `numereng cloud modal train status [--call-id <id>] [--timeout-seconds <n>] [--state-path <path>]`
- `numereng cloud modal train logs [--call-id <id>] [--lines <n>] [--state-path <path>]`
- `numereng cloud modal train cancel [--call-id <id>] [--state-path <path>]`
- `numereng cloud modal train pull [--call-id <id>] [--output-dir <path>] [--timeout-seconds <n>] [--state-path <path>]`

## `numerai`

- `numereng numerai datasets list [--round <num>] [--tournament <classic|signals|crypto>]`
- `numereng numerai datasets download --filename <path> [--dest-path <path>] [--round <num>] [--tournament <classic|signals|crypto>]`
- `numereng numerai models [list] [--tournament <classic|signals|crypto>]`
- `numereng numerai round current [--tournament <classic|signals|crypto>]`
- `numereng numerai forum scrape [--output-dir <path>] [--state-path <path>] [--full]`

## Exit Codes

- `0`: success/help
- `1`: runtime or boundary failure
- `2`: parse or usage failure

## Removed Surface

These legacy families are not part of the supported current CLI:

- `orchestrator`
- `optimize`
- `predict`
- `submission submit`
- `pipeline`
- `sync-live`
- `neutralize-sweep`
- `download`
- `status`
- `runpod`
- `vast`
- `db`
