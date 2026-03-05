"""Cloud command dispatcher."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from numereng.cli.commands.cloud_aws import handle_cloud_aws_command
from numereng.cli.commands.cloud_ec2 import handle_cloud_ec2_command
from numereng.cli.commands.cloud_modal import handle_cloud_modal_command
from numereng.cli.usage import USAGE


def handle_cloud_command(args: Sequence[str]) -> int:
    if not args:
        print(USAGE)
        return 0
    if args[0] in {"-h", "--help"}:
        print(USAGE)
        return 0
    if args[0] == "ec2":
        return handle_cloud_ec2_command(args[1:])
    if args[0] == "aws":
        return handle_cloud_aws_command(args[1:])
    if args[0] == "modal":
        return handle_cloud_modal_command(args[1:])
    print(f"unknown arguments: cloud {' '.join(args)}", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 2


__all__ = ["handle_cloud_command"]
