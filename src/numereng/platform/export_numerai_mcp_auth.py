"""Print a shell-safe export command for Numerai MCP auth."""

from __future__ import annotations

import argparse
import os
import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path

from numereng.platform.errors import NumeraiMcpAuthError

NUMERAI_MCP_AUTH_KEY = "NUMERAI_MCP_AUTH"


def find_env_file(*, start_dir: Path | None = None) -> Path:
    """Find the nearest `.env` file walking up from the starting directory."""
    search_dir = (start_dir or Path.cwd()).resolve()
    for candidate_dir in (search_dir, *search_dir.parents):
        candidate = candidate_dir / ".env"
        if candidate.is_file():
            return candidate
    raise NumeraiMcpAuthError("numerai_mcp_auth_env_file_not_found")


def read_env_value(env_file: Path, *, key: str) -> str | None:
    """Read one environment variable value from a dotenv-style file."""
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        candidate_key, raw_value = line.split("=", 1)
        if candidate_key.strip() != key:
            continue
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        return value.replace("\\$", "$")
    return None


def resolve_numerai_mcp_auth(
    *,
    env_file: Path | None = None,
    environ: Mapping[str, str] | None = None,
    start_dir: Path | None = None,
) -> str:
    """Resolve Numerai MCP auth from the current environment or repo `.env`."""
    resolved_environ = os.environ if environ is None else environ
    existing_value = resolved_environ.get(NUMERAI_MCP_AUTH_KEY)
    if existing_value:
        return existing_value

    resolved_env_file = env_file or find_env_file(start_dir=start_dir)
    file_value = read_env_value(resolved_env_file, key=NUMERAI_MCP_AUTH_KEY)
    if not file_value:
        raise NumeraiMcpAuthError("numerai_mcp_auth_not_found")
    return file_value


def build_export_command(value: str) -> str:
    """Build a POSIX-shell-safe export statement for Numerai MCP auth."""
    return f"export {NUMERAI_MCP_AUTH_KEY}={shlex.quote(value)}"


def main(argv: Sequence[str] | None = None) -> int:
    """Print an export command suitable for `eval` or `source`."""
    parser = argparse.ArgumentParser(
        description="Print a shell export command for NUMERAI_MCP_AUTH.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional path to the dotenv file to read.",
    )
    args = parser.parse_args(argv)

    export_value = resolve_numerai_mcp_auth(env_file=args.env_file)
    print(build_export_command(export_value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
