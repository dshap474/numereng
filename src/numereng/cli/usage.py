"""CLI usage text."""

USAGE = (
    "usage:\n"
    "  numereng [--fail]\n"
    "  numereng init [--workspace <path>] [--runtime-source <pypi|path>] [--runtime-path <path>] [--with-training] [--with-mlops]\n"  # noqa: E501
    "  numereng workspace sync [--workspace <path>] [--runtime-source <pypi|path>] [--runtime-path <path>] [--with-training] [--with-mlops]\n"  # noqa: E501
    "  numereng viz [--workspace <path>] [--host <host>] [--port <n>]\n"
    "  numereng run submit --model-name <name> (--run-id <id> | --predictions <path>) [--workspace <path>] [--tournament <classic|signals|crypto>] [--allow-non-live-artifact] [--neutralize --neutralizer-path <path> [--neutralization-proportion <0..1>] [--neutralization-mode <era|global>] [--neutralizer-cols <csv>] [--no-neutralization-rank]]\n"  # noqa: E501
    "  numereng run train --config <path.json> [--output-dir <path>] [--profile <simple|purged_walk_forward|full_history_refit>] [--post-training-scoring <none|core|full|round_core|round_full>] [--experiment-id <id>] [--workspace <path>]\n"  # noqa: E501
    "  numereng run cancel --run-id <id> [--workspace <path>]\n"
    "  numereng run score --run-id <id> [--stage <all|run_metric_series|post_fold|post_training_core|post_training_full>] [--workspace <path>]\n"  # noqa: E501
    "  numereng baseline build --run-ids <id1,id2,...> --name <baseline_name> [--default-target <target_col>] [--description <text>] [--promote-active] [--workspace <path>]\n"  # noqa: E501
    "  numereng experiment create --id <YYYY-MM-DD_slug> [--name <text>] [--hypothesis <text>] [--tags <csv>] [--workspace <path>]\n"  # noqa: E501
    "  numereng experiment list [--status <draft|active|complete|archived>] [--format <table|json>] [--workspace <path>]\n"  # noqa: E501
    "  numereng experiment details --id <id> [--format <table|json>] [--workspace <path>]\n"
    "  numereng experiment archive --id <id> [--workspace <path>]\n"
    "  numereng experiment unarchive --id <id> [--workspace <path>]\n"
    "  numereng experiment train --id <id> --config <path.json> [--output-dir <path>] [--profile <simple|purged_walk_forward|full_history_refit>] [--post-training-scoring <none|core|full|round_core|round_full>] [--workspace <path>]\n"  # noqa: E501
    "  numereng experiment run-plan --id <id> [--start-index <n>] [--end-index <n>] [--score-stage <post_training_core|post_training_full>] [--resume] [--workspace <path>]\n"  # noqa: E501
    "  numereng experiment score-round --id <id> --round <rN> --stage <post_training_core|post_training_full> [--workspace <path>]\n"  # noqa: E501
    "  numereng experiment promote --id <id> [--run <run_id>] [--metric <metric_key>] [--workspace <path>]\n"
    "  numereng experiment report --id <id> [--metric <metric_key>] [--limit <n>] [--format <table|json>] [--workspace <path>]\n"  # noqa: E501
    "  numereng experiment pack --id <id> [--workspace <path>]\n"
    "  numereng research program list [--format <table|json>] [--workspace <path>]\n"
    "  numereng research program show --program <id> [--format <table|json>] [--workspace <path>]\n"
    "  numereng research init --experiment-id <id> --program <id> [--workspace <path>]\n"
    "  numereng research status --experiment-id <id> [--format <table|json>] [--workspace <path>]\n"
    "  numereng research run --experiment-id <id> [--max-rounds <n>] [--max-paths <n>] [--workspace <path>]\n"  # noqa: E501
    "  numereng hpo create (--study-config <path.json> | (--study-id <id> --study-name <name> --config <path.json> --search-space <json|path>)) [--experiment-id <id>] [--metric <metric_key>] [--direction <maximize|minimize>] [--n-trials <n>] [--timeout-seconds <n>] [--max-completed-trials <n>] [--sampler <tpe|random>] [--seed <n>] [--neutralize --neutralizer-path <path> [--neutralization-proportion <0..1>] [--neutralization-mode <era|global>] [--neutralizer-cols <csv>] [--no-neutralization-rank]] [--workspace <path>]\n"  # noqa: E501
    "  numereng hpo list [--experiment-id <id>] [--status <running|completed|failed>] [--limit <n>] [--offset <n>] [--format <table|json>] [--workspace <path>]\n"  # noqa: E501
    "  numereng hpo details --study-id <id> [--format <table|json>] [--workspace <path>]\n"
    "  numereng hpo trials --study-id <id> [--format <table|json>] [--workspace <path>]\n"
    "  numereng ensemble build --run-ids <id1,id2,...> [--experiment-id <id>] [--method <rank_avg>] [--metric <metric_key>] [--target <target_col>] [--name <text>] [--ensemble-id <id>] [--weights <w1,w2,...>] [--optimize-weights] [--include-heavy-artifacts] [--selection-note <text>] [--regime-buckets <n>] [--neutralize-members] [--neutralize-final] [--neutralizer-path <path>] [--neutralization-proportion <0..1>] [--neutralization-mode <era|global>] [--neutralizer-cols <csv>] [--no-neutralization-rank] [--workspace <path>]\n"  # noqa: E501
    "  numereng ensemble select --experiment-id <id> --source-experiment-ids <id1,id2,...> --source-rules <json|path> [--selection-id <id>] [--target <target_col>] [--primary-metric <metric.path>] [--tie-break-metric <metric.path>] [--correlation-threshold <0..1>] [--top-weighted-variants <n>] [--weight-step <step>] [--required-seed-count <n>] [--require-full-seed-bundle] [--blend-variants <csv>] [--weighted-promotion-min-gain <value>] [--format <table|json>] [--workspace <path>]\n"  # noqa: E501
    "  numereng serve package create --experiment-id <id> --package-id <id> --components <json|path> [--data-version <v>] [--blend-rule <json|path>] [--neutralization <json|path>] [--workspace <path>]\n"  # noqa: E501
    "  numereng serve package inspect --experiment-id <id> --package-id <id> [--workspace <path>]\n"
    "  numereng serve package list [--experiment-id <id>] [--format <table|json>] [--workspace <path>]\n"
    "  numereng serve live build --experiment-id <id> --package-id <id> [--workspace <path>]\n"
    "  numereng serve live submit --experiment-id <id> --package-id <id> --model-name <name> [--workspace <path>]\n"
    "  numereng serve pickle build --experiment-id <id> --package-id <id> [--workspace <path>]\n"
    "  numereng serve pickle upload --experiment-id <id> --package-id <id> --model-name <name> [--data-version <v>] [--docker-image <image>] [--workspace <path>]\n"  # noqa: E501
    "  numereng neutralize apply (--run-id <id> | --predictions <path>) --neutralizer-path <path> [--neutralization-proportion <0..1>] [--neutralization-mode <era|global>] [--neutralizer-cols <csv>] [--output-path <path>] [--no-neutralization-rank] [--workspace <path>]\n"  # noqa: E501
    "  numereng monitor snapshot [--workspace <path>] [--no-refresh-cloud] [--json]\n"
    "  numereng remote list [--format <table|json>]\n"
    "  numereng remote bootstrap-viz [--workspace <path>]\n"
    "  numereng remote doctor --target <id>\n"
    "  numereng remote repo sync --target <id> [--workspace <path>]\n"
    "  numereng remote experiment launch --target <id> --experiment-id <id> [--start-index <n>] [--end-index <n>] [--score-stage <post_training_core|post_training_full>] [--sync-repo <auto|always|never>] [--workspace <path>]\n"  # noqa: E501
    "  numereng remote experiment status --target <id> --experiment-id <id> [--start-index <n>] [--end-index <n>] [--workspace <path>]\n"  # noqa: E501
    "  numereng remote experiment maintain --target <id> --experiment-id <id> [--start-index <n>] [--end-index <n>] [--workspace <path>]\n"  # noqa: E501
    "  numereng remote experiment stop --target <id> --experiment-id <id> [--start-index <n>] [--end-index <n>] [--workspace <path>]\n"  # noqa: E501
    "  numereng remote experiment sync --target <id> --experiment-id <id> [--workspace <path>]\n"
    "  numereng remote experiment pull --target <id> --experiment-id <id> [--workspace <path>]\n"
    "  numereng remote config push --target <id> --config <path.json> [--workspace <path>]\n"
    "  numereng remote run train --target <id> --config <path.json> [--experiment-id <id>] [--sync-repo <auto|always|never>] [--profile <simple|purged_walk_forward|full_history_refit>] [--post-training-scoring <none|core|full|round_core|round_full>] [--workspace <path>]\n"  # noqa: E501
    "  numereng dataset-tools build-downsampled-full [--data-version <v>] [--data-dir <path>] [--downsample-eras-step <n>] [--downsample-eras-offset <n>] [--rebuild]\n"  # noqa: E501
    "  numereng ensemble list [--experiment-id <id>] [--limit <n>] [--offset <n>] [--format <table|json>] [--workspace <path>]\n"  # noqa: E501
    "  numereng ensemble details --ensemble-id <id> [--format <table|json>] [--workspace <path>]\n"
    "  numereng store init [--workspace <path>]\n"
    "  numereng store index --run-id <id> [--workspace <path>]\n"
    "  numereng store rebuild [--workspace <path>]\n"
    "  numereng store doctor [--workspace <path>] [--fix-strays]\n"
    "  numereng store backfill-run-execution (--run-id <id> | --all) [--workspace <path>]\n"
    "  numereng store repair-run-lifecycles [--run-id <id>] [--workspace <path>] [--all]\n"
    "  numereng store materialize-viz-artifacts --kind <scoring-artifacts|per-era-corr(deprecated)> (--run-id <id> | --experiment-id <id> | --all) [--workspace <path>]\n"  # noqa: E501
    "  numereng cloud ec2 init-iam [--region <region>] [--bucket <bucket>] [--role-name <name>] [--security-group-name <name>]\n"  # noqa: E501
    "  numereng cloud ec2 setup-data --data-version <v> [--cache-dir <path>] [--region <region>] [--bucket <bucket>]\n"  # noqa: E501
    "  numereng cloud ec2 provision --run-id <id> [--tier <tier>] [--spot|--on-demand] [--region <region>] [--bucket <bucket>] [--state-path <path>]\n"  # noqa: E501
    "  numereng cloud ec2 package build-upload [--run-id <id>] [--region <region>] [--bucket <bucket>] [--state-path <path>]\n"  # noqa: E501
    "  numereng cloud ec2 config upload --config <path.json> [--run-id <id>] [--region <region>] [--bucket <bucket>] [--state-path <path>]\n"  # noqa: E501
    "  numereng cloud ec2 push --instance-id <id> [--run-id <id>] [--region <region>] [--bucket <bucket>] [--state-path <path>]\n"  # noqa: E501
    "  numereng cloud ec2 install --instance-id <id> [--run-id <id>] [--region <region>] [--runtime-profile <standard|lgbm-cuda>] [--state-path <path>]\n"  # noqa: E501
    "  numereng cloud ec2 train start --instance-id <id> [--run-id <id>] [--region <region>] [--state-path <path>]\n"  # noqa: E501
    "  numereng cloud ec2 train poll --instance-id <id> [--run-id <id>] [--timeout-seconds <n>] [--interval-seconds <n>] [--region <region>] [--state-path <path>]\n"  # noqa: E501
    "  numereng cloud ec2 logs --instance-id <id> [--lines <n>] [--follow] [--region <region>] [--state-path <path>]\n"  # noqa: E501
    "  numereng cloud ec2 pull --instance-id <id> [--run-id <id>] [--output-dir <path>] [--region <region>] [--bucket <bucket>] [--state-path <path>]\n"  # noqa: E501
    "  numereng cloud ec2 terminate --instance-id <id> [--region <region>] [--state-path <path>]\n"  # noqa: E501
    "  numereng cloud ec2 status [--run-id <id>] [--region <region>] [--state-path <path>]\n"
    "  numereng cloud ec2 s3 ls --prefix <prefix> [--region <region>] [--bucket <bucket>]\n"
    "  numereng cloud ec2 s3 cp --src <path|s3://...> --dst <path|s3://...> [--region <region>] [--bucket <bucket>]\n"  # noqa: E501
    "  numereng cloud ec2 s3 rm --uri <s3://...> [--recursive] [--region <region>] [--bucket <bucket>]\n"
    "  numereng cloud aws image build-push [--run-id <id>] [--context-dir <path>] [--dockerfile <path>] [--runtime-profile <standard|lgbm-cuda>] [--repository <name>] [--image-tag <tag>] [--platform <value>] [--region <region>] [--bucket <bucket>] [--state-path <path>] [--workspace <path>]\n"  # noqa: E501
    "  numereng cloud aws train submit [--run-id <id>] [--backend <sagemaker|batch>] [--config <path.json>] [--config-s3-uri <s3://...json>] [--image-uri <uri>] [--runtime-profile <standard|lgbm-cuda>] [--role-arn <arn>] [--instance-type <name>] [--instance-count <n>] [--volume-size-gb <n>] [--max-runtime-seconds <n>] [--max-wait-seconds <n>] [--spot|--on-demand] [--checkpoint-s3-uri <s3://...>] [--output-s3-uri <s3://...>] [--batch-job-queue <name>] [--batch-job-definition <name>] [--region <region>] [--bucket <bucket>] [--state-path <path>] [--workspace <path>]\n"  # noqa: E501
    "  numereng cloud aws train status [--backend <sagemaker|batch>] [--run-id <id>] [--training-job-name <name>] [--batch-job-id <id>] [--region <region>] [--state-path <path>] [--workspace <path>]\n"  # noqa: E501
    "  numereng cloud aws train logs [--backend <sagemaker|batch>] [--run-id <id>] [--training-job-name <name>] [--batch-job-id <id>] [--lines <n>] [--follow] [--region <region>] [--state-path <path>] [--workspace <path>]\n"  # noqa: E501
    "  numereng cloud aws train cancel [--backend <sagemaker|batch>] [--run-id <id>] [--training-job-name <name>] [--batch-job-id <id>] [--region <region>] [--state-path <path>] [--workspace <path>]\n"  # noqa: E501
    "  numereng cloud aws train pull [--run-id <id>] [--output-s3-uri <s3://...>] [--output-dir <path>] [--region <region>] [--bucket <bucket>] [--state-path <path>] [--workspace <path>]\n"  # noqa: E501
    "  numereng cloud aws train extract [--run-id <id>] [--output-dir <path>] [--region <region>] [--bucket <bucket>] [--state-path <path>] [--workspace <path>]\n"  # noqa: E501
    "  numereng cloud modal deploy --ecr-image-uri <uri:tag> [--app-name <name>] [--function-name <name>] [--environment-name <name>] [--aws-profile <name>] [--timeout-seconds <n>] [--gpu <value>] [--cpu <n>] [--memory-mb <n>] [--data-volume-name <name>] [--metadata <k=v,...>] [--state-path <path>]\n"  # noqa: E501
    "  numereng cloud modal data sync --config <path.json> --volume-name <name> [--force] [--no-create-if-missing] [--metadata <k=v,...>] [--state-path <path>]\n"  # noqa: E501
    "  numereng cloud modal train submit --config <path.json> [--output-dir <path>] [--profile <simple|purged_walk_forward|full_history_refit>] [--app-name <name>] [--function-name <name>] [--environment-name <name>] [--metadata <k=v,...>] [--state-path <path>]\n"  # noqa: E501
    "  numereng cloud modal train status [--call-id <id>] [--timeout-seconds <n>] [--state-path <path>]\n"
    "  numereng cloud modal train logs [--call-id <id>] [--lines <n>] [--state-path <path>]\n"
    "  numereng cloud modal train cancel [--call-id <id>] [--state-path <path>]\n"
    "  numereng cloud modal train pull [--call-id <id>] [--output-dir <path>] [--timeout-seconds <n>] [--state-path <path>]\n"  # noqa: E501
    "  numereng numerai datasets list [--round <num>] [--tournament <classic|signals|crypto>]\n"
    "  numereng numerai datasets download --filename <path> [--dest-path <path>] [--round <num>] [--tournament <classic|signals|crypto>]\n"  # noqa: E501
    "  numereng numerai models [list] [--tournament <classic|signals|crypto>]\n"
    "  numereng numerai round current [--tournament <classic|signals|crypto>]\n"
    "  numereng numerai forum scrape [--output-dir <path>] [--state-path <path>] [--full]"
)
