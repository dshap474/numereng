"""Shell-aware SSH command builders for remote numereng targets."""

from __future__ import annotations

import base64
import os
import shlex
from collections.abc import Sequence
from pathlib import PurePosixPath, PureWindowsPath

from numereng.platform.remotes.contracts import SshRemoteTargetProfile

_PYTHON_EXEC_WRAPPER = "import base64,sys; exec(base64.b64decode(sys.argv.pop(1)).decode())"


def powershell_single_quote(value: str) -> str:
    """Return a PowerShell-safe single-quoted literal body."""

    return value.replace("'", "''")


def ssh_destination(target: SshRemoteTargetProfile) -> str:
    """Resolve SSH destination from config host or explicit host + env-backed user."""

    destination = target.ssh_config_host or target.host or ""
    if target.ssh_config_host is None:
        user = _env_value(target.user_env)
        if user:
            destination = f"{user}@{destination}"
    return destination


def ssh_base_command(target: SshRemoteTargetProfile) -> list[str]:
    """Build the base SSH argv prefix for one target."""

    args = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={target.connect_timeout_seconds}",
    ]
    identity_file = _env_value(target.identity_file_env)
    if identity_file:
        args.extend(["-i", identity_file])
    if target.port is not None:
        args.extend(["-p", str(target.port)])
    args.append(ssh_destination(target))
    return args


def scp_base_command(target: SshRemoteTargetProfile) -> list[str]:
    """Build the base SCP argv prefix for one target."""

    args = [
        "scp",
        "-B",
        "-o",
        f"ConnectTimeout={target.connect_timeout_seconds}",
    ]
    identity_file = _env_value(target.identity_file_env)
    if identity_file:
        args.extend(["-i", identity_file])
    if target.port is not None:
        args.extend(["-P", str(target.port)])
    return args


def build_ssh_command(target: SshRemoteTargetProfile, remote_command: str) -> list[str]:
    """Build the full SSH argv for one remote command."""

    return [*ssh_base_command(target), remote_command]


def remote_path_join(target: SshRemoteTargetProfile, *parts: str) -> str:
    """Join path fragments using the remote host's path semantics."""

    if target.shell == "powershell":
        path = PureWindowsPath(parts[0])
        for part in parts[1:]:
            path /= part
        return str(path)

    path = PurePosixPath(parts[0])
    for part in parts[1:]:
        path /= part
    return str(path)


def build_remote_shell_command(
    target: SshRemoteTargetProfile,
    command: str,
    *,
    cwd: str | None = None,
) -> str:
    """Wrap a trusted command fragment in the target's default shell."""

    if target.shell == "powershell":
        fragments: list[str] = []
        if cwd:
            fragments.append(f"Set-Location '{powershell_single_quote(cwd)}'")
        fragments.append(command)
        body = "; ".join(fragments).replace('"', '`"')
        return f'powershell -NoProfile -Command "{body}"'

    if cwd:
        return f"cd {shlex.quote(cwd)} && {command}"
    return command


def build_monitor_snapshot_command(target: SshRemoteTargetProfile) -> str:
    """Build the trusted remote numereng monitor snapshot command."""

    if target.shell == "powershell":
        command = (
            f"{target.runner_cmd} monitor snapshot --store-root '{powershell_single_quote(target.store_root)}' --json"
        )
        return build_remote_shell_command(target, command, cwd=target.repo_root)

    command = f"{target.runner_cmd} monitor snapshot --store-root {shlex.quote(target.store_root)} --json"
    return build_remote_shell_command(target, command, cwd=target.repo_root)


def build_remote_python_command(
    target: SshRemoteTargetProfile,
    script_source: str,
    *,
    args: Sequence[str] = (),
    cwd: str | None = None,
) -> str:
    """Build a shell-aware remote Python execution command."""

    encoded = base64.b64encode(script_source.encode("utf-8")).decode("ascii")
    if target.shell == "powershell":
        arg_string = " ".join(
            [f"'{powershell_single_quote(encoded)}'"] + [f"'{powershell_single_quote(arg)}'" for arg in args]
        )
        command = f"{target.python_cmd} -c '{powershell_single_quote(_PYTHON_EXEC_WRAPPER)}' {arg_string}".rstrip()
        return build_remote_shell_command(target, command, cwd=cwd)

    arg_string = " ".join([shlex.quote(encoded)] + [shlex.quote(arg) for arg in args])
    command = f"{target.python_cmd} -c {shlex.quote(_PYTHON_EXEC_WRAPPER)} {arg_string}".rstrip()
    return build_remote_shell_command(target, command, cwd=cwd)


def _env_value(name: str | None) -> str | None:
    if not name:
        return None
    value = os.getenv(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


__all__ = [
    "build_monitor_snapshot_command",
    "build_remote_python_command",
    "build_remote_shell_command",
    "build_ssh_command",
    "scp_base_command",
    "powershell_single_quote",
    "remote_path_join",
    "ssh_base_command",
    "ssh_destination",
]
