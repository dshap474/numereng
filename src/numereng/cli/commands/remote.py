"""Remote SSH ops command handlers."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from pydantic import ValidationError

from numereng import api
from numereng.cli.common import _parse_simple_options, _validation_error_message
from numereng.cli.usage import USAGE
from numereng.platform.errors import PackageError


def handle_remote_command(args: Sequence[str]) -> int:
    if not args:
        print(USAGE)
        return 0
    if args[0] in {"-h", "--help"}:
        print(USAGE)
        return 0

    if args[0] == "list":
        return _handle_remote_list(args[1:])
    if args[0] == "bootstrap-viz":
        return _handle_remote_bootstrap_viz(args[1:])
    if args[0] == "doctor":
        return _handle_remote_doctor(args[1:])
    if args[0] == "repo" and len(args) >= 2 and args[1] == "sync":
        return _handle_remote_repo_sync(args[2:])
    if args[0] == "experiment" and len(args) >= 2 and args[1] == "sync":
        return _handle_remote_experiment_sync(args[2:])
    if args[0] == "experiment" and len(args) >= 2 and args[1] == "pull":
        return _handle_remote_experiment_pull(args[2:])
    if args[0] == "config" and len(args) >= 2 and args[1] == "push":
        return _handle_remote_config_push(args[2:])
    if args[0] == "run" and len(args) >= 2 and args[1] == "train":
        return _handle_remote_run_train(args[2:])

    print(f"unknown arguments: remote {' '.join(args)}", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 2


def _handle_remote_list(args: Sequence[str]) -> int:
    values, _, parse_error = _parse_simple_options(args, value_flags={"--format"})
    if parse_error == "__help__":
        print(USAGE)
        return 0
    if parse_error is not None:
        print(parse_error, file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    output_format = values.get("--format", "json")
    if output_format not in {"json", "table"}:
        print("invalid value for --format: expected table|json", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    try:
        payload = api.remote_list_targets(api.RemoteTargetListRequest())
    except PackageError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if output_format == "json":
        print(payload.model_dump_json())
        return 0
    for target in payload.targets:
        tags = ",".join(target.tags)
        print(f"{target.id}\t{target.label}\t{target.shell}\t{target.repo_root}\t{target.store_root}\t{tags}")
    return 0


def _handle_remote_bootstrap_viz(args: Sequence[str]) -> int:
    values, _, parse_error = _parse_simple_options(args, value_flags={"--workspace"})
    if parse_error == "__help__":
        print(USAGE)
        return 0
    if parse_error is not None:
        print(parse_error, file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    try:
        payload = api.remote_bootstrap_viz(api.RemoteVizBootstrapRequest(workspace_root=values.get("--workspace", ".")))
    except ValidationError as exc:
        print(_validation_error_message(exc), file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    except PackageError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(payload.model_dump_json())
    return 0


def _handle_remote_doctor(args: Sequence[str]) -> int:
    values, _, parse_error = _parse_simple_options(args, value_flags={"--target"})
    if parse_error == "__help__":
        print(USAGE)
        return 0
    if parse_error is not None:
        print(parse_error, file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    target_id = values.get("--target")
    if target_id is None:
        print("missing required argument: --target", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    try:
        payload = api.remote_doctor(api.RemoteDoctorRequest(target_id=target_id))
    except ValidationError as exc:
        print(_validation_error_message(exc), file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    except PackageError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(payload.model_dump_json())
    return 0


def _handle_remote_repo_sync(args: Sequence[str]) -> int:
    values, _, parse_error = _parse_simple_options(args, value_flags={"--target", "--workspace"})
    if parse_error == "__help__":
        print(USAGE)
        return 0
    if parse_error is not None:
        print(parse_error, file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    target_id = values.get("--target")
    if target_id is None:
        print("missing required argument: --target", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    try:
        payload = api.remote_repo_sync(
            api.RemoteRepoSyncRequest(
                target_id=target_id,
                workspace_root=values.get("--workspace", "."),
            )
        )
    except ValidationError as exc:
        print(_validation_error_message(exc), file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    except PackageError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(payload.model_dump_json())
    return 0


def _handle_remote_experiment_sync(args: Sequence[str]) -> int:
    values, _, parse_error = _parse_simple_options(args, value_flags={"--target", "--experiment-id", "--workspace"})
    if parse_error == "__help__":
        print(USAGE)
        return 0
    if parse_error is not None:
        print(parse_error, file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    target_id = values.get("--target")
    experiment_id = values.get("--experiment-id")
    if target_id is None:
        print("missing required argument: --target", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    if experiment_id is None:
        print("missing required argument: --experiment-id", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    try:
        payload = api.remote_experiment_sync(
            api.RemoteExperimentSyncRequest(
                target_id=target_id,
                experiment_id=experiment_id,
                workspace_root=values.get("--workspace", "."),
            )
        )
    except ValidationError as exc:
        print(_validation_error_message(exc), file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    except PackageError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(payload.model_dump_json())
    return 0


def _handle_remote_experiment_pull(args: Sequence[str]) -> int:
    values, _, parse_error = _parse_simple_options(args, value_flags={"--target", "--experiment-id", "--workspace"})
    if parse_error == "__help__":
        print(USAGE)
        return 0
    if parse_error is not None:
        print(parse_error, file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    target_id = values.get("--target")
    experiment_id = values.get("--experiment-id")
    if target_id is None:
        print("missing required argument: --target", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    if experiment_id is None:
        print("missing required argument: --experiment-id", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    try:
        payload = api.remote_experiment_pull(
            api.RemoteExperimentPullRequest(
                target_id=target_id,
                experiment_id=experiment_id,
                workspace_root=values.get("--workspace", "."),
            )
        )
    except ValidationError as exc:
        print(_validation_error_message(exc), file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    except PackageError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(payload.model_dump_json())
    return 0


def _handle_remote_config_push(args: Sequence[str]) -> int:
    values, _, parse_error = _parse_simple_options(args, value_flags={"--target", "--config", "--workspace"})
    if parse_error == "__help__":
        print(USAGE)
        return 0
    if parse_error is not None:
        print(parse_error, file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    target_id = values.get("--target")
    config_path = values.get("--config")
    if target_id is None:
        print("missing required argument: --target", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    if config_path is None:
        print("missing required argument: --config", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    try:
        payload = api.remote_config_push(
            api.RemoteConfigPushRequest(
                target_id=target_id,
                config_path=config_path,
                workspace_root=values.get("--workspace", "."),
            )
        )
    except ValidationError as exc:
        print(_validation_error_message(exc), file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    except PackageError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(payload.model_dump_json())
    return 0


def _handle_remote_run_train(args: Sequence[str]) -> int:
    values, _, parse_error = _parse_simple_options(
        args,
        value_flags={
            "--target",
            "--config",
            "--experiment-id",
            "--sync-repo",
            "--profile",
            "--post-training-scoring",
            "--workspace",
        },
    )
    if parse_error == "__help__":
        print(USAGE)
        return 0
    if parse_error is not None:
        print(parse_error, file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    target_id = values.get("--target")
    config_path = values.get("--config")
    if target_id is None:
        print("missing required argument: --target", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    if config_path is None:
        print("missing required argument: --config", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    try:
        payload = api.remote_train_launch(
            api.RemoteTrainLaunchRequest(
                target_id=target_id,
                config_path=config_path,
                experiment_id=values.get("--experiment-id"),
                sync_repo=values.get("--sync-repo", "auto"),
                profile=values.get("--profile"),
                post_training_scoring=values.get("--post-training-scoring"),
                workspace_root=values.get("--workspace", "."),
            )
        )
    except ValidationError as exc:
        print(_validation_error_message(exc), file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    except PackageError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(payload.model_dump_json())
    return 0


__all__ = ["handle_remote_command"]
