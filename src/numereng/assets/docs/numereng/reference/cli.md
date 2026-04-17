# CLI Commands

Reference for the supported `numereng` CLI surface.

## Top-Level

- `numereng [--fail]`
- `numereng --help`
- `numereng docs sync numerai [--workspace <path>]`

Notes:

- work from the repo root by default
- `--workspace` targets another checkout's `.numereng` store and repo-local docs paths; custom model and research program discovery still comes from the current source checkout
- `docs sync numerai` downloads the official Numerai docs into `docs/numerai/`

## `run`

- `numereng run train --config <path.json> [--output-dir <path>] [--profile <simple|purged_walk_forward|full_history_refit>] [--experiment-id <id>] [--post-training-scoring <none|core|full|round_core|round_full>]`
- `numereng run score --run-id <id> [--stage <all|run_metric_series|post_fold|post_training_core|post_training_full>] [--workspace <path>]`
- `numereng run submit --model-name <name> (--run-id <id> | --predictions <path>) [--workspace <path>] [--tournament <classic|signals|crypto>] [--allow-non-live-artifact] [--neutralize --neutralizer-path <path> [--neutralization-proportion <0..1>] [--neutralization-mode <era|global>] [--neutralizer-cols <csv>] [--no-neutralization-rank]]`

## `experiment`

- `numereng experiment create --id <YYYY-MM-DD_slug> [--name <text>] [--hypothesis <text>] [--tags <csv>] [--workspace <path>]`
- `numereng experiment list [--status <draft|active|complete|archived>] [--format <table|json>] [--workspace <path>]`
- `numereng experiment details --id <id> [--format <table|json>] [--workspace <path>]`
- `numereng experiment archive --id <id> [--workspace <path>]`
- `numereng experiment unarchive --id <id> [--workspace <path>]`
- `numereng experiment train --id <id> --config <path.json> [--output-dir <path>] [--profile <simple|purged_walk_forward|full_history_refit>] [--post-training-scoring <none|core|full|round_core|round_full>] [--workspace <path>]`
- `numereng experiment score-round --id <id> --round <rN> --stage <post_training_core|post_training_full> [--workspace <path>]`
- `numereng experiment promote --id <id> [--run <run_id>] [--metric <metric_key>] [--workspace <path>]`
- `numereng experiment report --id <id> [--metric <metric_key>] [--limit <n>] [--format <table|json>] [--workspace <path>]`
- `numereng experiment pack --id <id> [--workspace <path>]`

Notes:

- `experiment create` scaffolds `.numereng/experiments/<id>/`
- `archive` moves `.numereng/experiments/<id>` to `.numereng/experiments/_archive/<id>`
- `pack` writes `.numereng/experiments/<id>/EXPERIMENT.pack.md`

## `research`

- `numereng research program list [--format <table|json>] [--workspace <path>]`
- `numereng research program show --program <id> [--format <table|json>] [--workspace <path>]`
- `numereng research init --experiment-id <id> --program <id> [--workspace <path>]`
- `numereng research status --experiment-id <id> [--format <table|json>] [--workspace <path>]`
- `numereng research run --experiment-id <id> [--max-rounds <n>] [--max-paths <n>] [--workspace <path>]`

## `hpo`

- `numereng hpo create (--study-config <path.json> | (--study-id <id> --study-name <name> --config <path.json> --search-space <json|path>)) [--experiment-id <id>] [--metric <metric_key>] [--direction <maximize|minimize>] [--n-trials <n>] [--timeout-seconds <n>] [--max-completed-trials <n>] [--sampler <tpe|random>] [--seed <n>] [--neutralize --neutralizer-path <path> [--neutralization-proportion <0..1>] [--neutralization-mode <era|global>] [--neutralizer-cols <csv>] [--no-neutralization-rank]] [--workspace <path>]`
- `numereng hpo list [--experiment-id <id>] [--status <running|completed|failed>] [--limit <n>] [--offset <n>] [--format <table|json>] [--workspace <path>]`
- `numereng hpo details --study-id <id> [--format <table|json>] [--workspace <path>]`
- `numereng hpo trials --study-id <id> [--format <table|json>] [--workspace <path>]`

## `ensemble`

- `numereng ensemble build --run-ids <id1,id2,...> [--experiment-id <id>] [--method <rank_avg>] [--metric <metric_key>] [--target <target_col>] [--name <text>] [--ensemble-id <id>] [--weights <w1,w2,...>] [--optimize-weights] [--include-heavy-artifacts] [--selection-note <text>] [--regime-buckets <n>] [--neutralize-members] [--neutralize-final] [--neutralizer-path <path>] [--neutralization-proportion <0..1>] [--neutralization-mode <era|global>] [--neutralizer-cols <csv>] [--no-neutralization-rank] [--workspace <path>]`
- `numereng ensemble list [--experiment-id <id>] [--limit <n>] [--offset <n>] [--format <table|json>] [--workspace <path>]`
- `numereng ensemble details --ensemble-id <id> [--format <table|json>] [--workspace <path>]`

## `serve`

- `numereng serve package create --experiment-id <id> --package-id <id> --components <json|path> [--data-version <v>] [--blend-rule <json|path>] [--neutralization <json|path>] [--workspace <path>]`
- `numereng serve package inspect --experiment-id <id> --package-id <id> [--workspace <path>]`
- `numereng serve package list [--experiment-id <id>] [--format <table|json>] [--workspace <path>]`
- `numereng serve package score --experiment-id <id> --package-id <id> [--dataset <validation>] [--runtime <auto|pickle|local>] [--stage <post_training_core|post_training_full>] [--workspace <path>]`
- `numereng serve package sync-diagnostics --experiment-id <id> --package-id <id> [--no-wait] [--workspace <path>]`
- `numereng serve live build --experiment-id <id> --package-id <id> [--workspace <path>]`
- `numereng serve live submit --experiment-id <id> --package-id <id> --model-name <name> [--workspace <path>]`
- `numereng serve pickle build --experiment-id <id> --package-id <id> [--docker-image <image>] [--workspace <path>]`
- `numereng serve pickle upload --experiment-id <id> --package-id <id> --model-name <name> [--data-version <v>] [--docker-image <image>] [--wait-diagnostics] [--workspace <path>]`

## `neutralize`

- `numereng neutralize apply (--run-id <id> | --predictions <path>) --neutralizer-path <path> [--neutralization-proportion <0..1>] [--neutralization-mode <era|global>] [--neutralizer-cols <csv>] [--output-path <path>] [--no-neutralization-rank] [--workspace <path>]`

## `dataset-tools`

- `numereng dataset-tools build-downsampled-full [--data-version <v>] [--data-dir <path>] [--downsample-eras-step <n>] [--downsample-eras-offset <n>] [--rebuild]`

## `store`

- `numereng store init [--workspace <path>]`
- `numereng store index --run-id <id> [--workspace <path>]`
- `numereng store rebuild [--workspace <path>]`
- `numereng store doctor [--workspace <path>] [--fix-strays]`
- `numereng store materialize-viz-artifacts --kind <per-era-corr|scoring-artifacts> (--run-id <id> | --experiment-id <id> | --all) [--workspace <path>]`

## `monitor`

- `numereng monitor snapshot [--workspace <path>] [--no-refresh-cloud] [--json]`

## `remote`

- `numereng remote list [--format <table|json>]`
- `numereng remote bootstrap-viz [--workspace <path>]`
- `numereng remote doctor --target <id>`
- `numereng remote repo sync --target <id> [--workspace <path>]`
- `numereng remote experiment launch --target <id> --experiment-id <id> [--start-index <n>] [--end-index <n>] [--score-stage <post_training_core|post_training_full>] [--sync-repo <auto|always|never>] [--workspace <path>]`
- `numereng remote experiment status --target <id> --experiment-id <id> [--start-index <n>] [--end-index <n>] [--workspace <path>]`
- `numereng remote experiment maintain --target <id> --experiment-id <id> [--start-index <n>] [--end-index <n>] [--workspace <path>]`
- `numereng remote experiment stop --target <id> --experiment-id <id> [--start-index <n>] [--end-index <n>] [--workspace <path>]`
- `numereng remote experiment sync --target <id> --experiment-id <id> [--workspace <path>]`
- `numereng remote experiment pull --target <id> --experiment-id <id> [--workspace <path>]`
- `numereng remote config push --target <id> --config <path.json> [--workspace <path>]`
- `numereng remote run train --target <id> --config <path.json> [--experiment-id <id>] [--sync-repo <auto|always|never>] [--profile <simple|purged_walk_forward|full_history_refit>] [--post-training-scoring <none|core|full|round_core|round_full>] [--workspace <path>]`

## `cloud`

- `numereng cloud ec2 ...`
- `numereng cloud aws ...`
- `numereng cloud modal ...`

## `numerai`

- `numereng numerai datasets list [--round <num>] [--tournament <classic|signals|crypto>]`
- `numereng numerai datasets download --filename <path> [--dest-path <path>] [--round <num>] [--tournament <classic|signals|crypto>]`
- `numereng numerai models [list] [--tournament <classic|signals|crypto>]`
- `numereng numerai round current [--tournament <classic|signals|crypto>]`
- `numereng numerai forum scrape [--output-dir <path>] [--state-path <path>] [--full]`
