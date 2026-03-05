"""CLI usage text."""

USAGE = (
    "usage:\n"
    "  numereng [--fail]\n"
    "  numereng run submit --model-name <name> (--run-id <id> | --predictions <path>) [--store-root <path>] [--tournament <classic|signals|crypto>] [--allow-non-live-artifact] [--neutralize --neutralizer-path <path> [--neutralization-proportion <0..1>] [--neutralization-mode <era|global>] [--neutralizer-cols <csv>] [--no-neutralization-rank]]\n"  # noqa: E501
    "  numereng run train --config <path.json> [--output-dir <path>] [--profile <simple|purged_walk_forward|submission>] [--experiment-id <id>]\n"  # noqa: E501
    "  numereng experiment create --id <YYYY-MM-DD_slug> [--name <text>] [--hypothesis <text>] [--tags <csv>] [--store-root <path>]\n"  # noqa: E501
    "  numereng experiment list [--status <draft|active|complete|archived>] [--format <table|json>] [--store-root <path>]\n"  # noqa: E501
    "  numereng experiment details --id <id> [--format <table|json>] [--store-root <path>]\n"
    "  numereng experiment train --id <id> --config <path.json> [--output-dir <path>] [--profile <simple|purged_walk_forward|submission>] [--store-root <path>]\n"  # noqa: E501
    "  numereng experiment promote --id <id> [--run <run_id>] [--metric <metric_key>] [--store-root <path>]\n"
    "  numereng experiment report --id <id> [--metric <metric_key>] [--limit <n>] [--format <table|json>] [--store-root <path>]\n"  # noqa: E501
    "  numereng hpo create (--study-config <path.json> | (--study-name <name> --config <path.json>)) [--experiment-id <id>] [--metric <metric_key>] [--direction <maximize|minimize>] [--n-trials <n>] [--sampler <tpe|random>] [--seed <n>] [--search-space <json|path>] [--neutralize --neutralizer-path <path> [--neutralization-proportion <0..1>] [--neutralization-mode <era|global>] [--neutralizer-cols <csv>] [--no-neutralization-rank]] [--store-root <path>]\n"  # noqa: E501
    "  numereng hpo list [--experiment-id <id>] [--status <running|completed|failed>] [--limit <n>] [--offset <n>] [--format <table|json>] [--store-root <path>]\n"  # noqa: E501
    "  numereng hpo details --study-id <id> [--format <table|json>] [--store-root <path>]\n"
    "  numereng hpo trials --study-id <id> [--format <table|json>] [--store-root <path>]\n"
    "  numereng ensemble build --run-ids <id1,id2,...> [--experiment-id <id>] [--method <rank_avg>] [--metric <metric_key>] [--target <target_col>] [--name <text>] [--ensemble-id <id>] [--weights <w1,w2,...>] [--optimize-weights] [--include-heavy-artifacts] [--selection-note <text>] [--regime-buckets <n>] [--neutralize-members] [--neutralize-final] [--neutralizer-path <path>] [--neutralization-proportion <0..1>] [--neutralization-mode <era|global>] [--neutralizer-cols <csv>] [--no-neutralization-rank] [--store-root <path>]\n"  # noqa: E501
    "  numereng neutralize apply (--run-id <id> | --predictions <path>) --neutralizer-path <path> [--neutralization-proportion <0..1>] [--neutralization-mode <era|global>] [--neutralizer-cols <csv>] [--output-path <path>] [--no-neutralization-rank] [--store-root <path>]\n"  # noqa: E501
    "  numereng dataset-tools build-full-datasets [--data-version <v>] [--data-dir <path>] [--downsample-eras-step <n>] [--downsample-eras-offset <n>] [--skip-downsample] [--rebuild]\n"  # noqa: E501
    "  numereng ensemble list [--experiment-id <id>] [--limit <n>] [--offset <n>] [--format <table|json>] [--store-root <path>]\n"  # noqa: E501
    "  numereng ensemble details --ensemble-id <id> [--format <table|json>] [--store-root <path>]\n"
    "  numereng store init [--store-root <path>]\n"
    "  numereng store index --run-id <id> [--store-root <path>]\n"
    "  numereng store rebuild [--store-root <path>]\n"
    "  numereng store doctor [--store-root <path>] [--fix-strays]\n"
    "  numereng cloud ec2 init-iam [--region <region>] [--bucket <bucket>] [--role-name <name>] [--security-group-name <name>]\n"  # noqa: E501
    "  numereng cloud ec2 setup-data --data-version <v> [--cache-dir <path>] [--region <region>] [--bucket <bucket>]\n"  # noqa: E501
    "  numereng cloud ec2 provision --run-id <id> [--tier <tier>] [--spot|--on-demand] [--region <region>] [--bucket <bucket>] [--state-path <path>]\n"  # noqa: E501
    "  numereng cloud ec2 package build-upload [--run-id <id>] [--region <region>] [--bucket <bucket>] [--state-path <path>]\n"  # noqa: E501
    "  numereng cloud ec2 config upload --config <path.json> [--run-id <id>] [--region <region>] [--bucket <bucket>] [--state-path <path>]\n"  # noqa: E501
    "  numereng cloud ec2 push --instance-id <id> [--run-id <id>] [--region <region>] [--bucket <bucket>] [--state-path <path>]\n"  # noqa: E501
    "  numereng cloud ec2 install --instance-id <id> [--run-id <id>] [--region <region>] [--state-path <path>]\n"  # noqa: E501
    "  numereng cloud ec2 train start --instance-id <id> [--run-id <id>] [--region <region>] [--state-path <path>]\n"  # noqa: E501
    "  numereng cloud ec2 train poll --instance-id <id> [--run-id <id>] [--timeout-seconds <n>] [--interval-seconds <n>] [--region <region>] [--state-path <path>]\n"  # noqa: E501
    "  numereng cloud ec2 logs --instance-id <id> [--lines <n>] [--follow] [--region <region>] [--state-path <path>]\n"  # noqa: E501
    "  numereng cloud ec2 pull --instance-id <id> [--run-id <id>] [--output-dir <path>] [--region <region>] [--bucket <bucket>] [--state-path <path>]\n"  # noqa: E501
    "  numereng cloud ec2 terminate --instance-id <id> [--region <region>] [--state-path <path>]\n"  # noqa: E501
    "  numereng cloud ec2 status [--run-id <id>] [--region <region>] [--state-path <path>]\n"
    "  numereng cloud ec2 s3 ls --prefix <prefix> [--region <region>] [--bucket <bucket>]\n"
    "  numereng cloud ec2 s3 cp --src <path|s3://...> --dst <path|s3://...> [--region <region>] [--bucket <bucket>]\n"  # noqa: E501
    "  numereng cloud ec2 s3 rm --uri <s3://...> [--recursive] [--region <region>] [--bucket <bucket>]\n"
    "  numereng cloud aws image build-push [--run-id <id>] [--context-dir <path>] [--dockerfile <path>] [--repository <name>] [--image-tag <tag>] [--platform <value>] [--region <region>] [--bucket <bucket>] [--state-path <path>] [--store-root <path>]\n"  # noqa: E501
    "  numereng cloud aws train submit [--run-id <id>] [--backend <sagemaker|batch>] [--config <path.json>] [--config-s3-uri <s3://...json>] [--image-uri <uri>] [--role-arn <arn>] [--instance-type <name>] [--instance-count <n>] [--volume-size-gb <n>] [--max-runtime-seconds <n>] [--max-wait-seconds <n>] [--spot|--on-demand] [--checkpoint-s3-uri <s3://...>] [--output-s3-uri <s3://...>] [--batch-job-queue <name>] [--batch-job-definition <name>] [--region <region>] [--bucket <bucket>] [--state-path <path>] [--store-root <path>]\n"  # noqa: E501
    "  numereng cloud aws train status [--backend <sagemaker|batch>] [--run-id <id>] [--training-job-name <name>] [--batch-job-id <id>] [--region <region>] [--state-path <path>] [--store-root <path>]\n"  # noqa: E501
    "  numereng cloud aws train logs [--backend <sagemaker|batch>] [--run-id <id>] [--training-job-name <name>] [--batch-job-id <id>] [--lines <n>] [--follow] [--region <region>] [--state-path <path>] [--store-root <path>]\n"  # noqa: E501
    "  numereng cloud aws train cancel [--backend <sagemaker|batch>] [--run-id <id>] [--training-job-name <name>] [--batch-job-id <id>] [--region <region>] [--state-path <path>] [--store-root <path>]\n"  # noqa: E501
    "  numereng cloud aws train pull [--run-id <id>] [--output-s3-uri <s3://...>] [--output-dir <path>] [--region <region>] [--bucket <bucket>] [--state-path <path>]\n"  # noqa: E501
    "  numereng cloud aws train extract [--run-id <id>] [--output-dir <path>] [--region <region>] [--bucket <bucket>] [--state-path <path>]\n"  # noqa: E501
    "  numereng cloud modal deploy --ecr-image-uri <uri:tag> [--app-name <name>] [--function-name <name>] [--environment-name <name>] [--aws-profile <name>] [--timeout-seconds <n>] [--gpu <value>] [--cpu <n>] [--memory-mb <n>] [--data-volume-name <name>] [--metadata <k=v,...>] [--state-path <path>]\n"  # noqa: E501
    "  numereng cloud modal data sync --config <path.json> --volume-name <name> [--force] [--no-create-if-missing] [--metadata <k=v,...>] [--state-path <path>]\n"  # noqa: E501
    "  numereng cloud modal train submit --config <path.json> [--output-dir <path>] [--profile <simple|purged_walk_forward|submission>] [--app-name <name>] [--function-name <name>] [--environment-name <name>] [--metadata <k=v,...>] [--state-path <path>]\n"  # noqa: E501
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
