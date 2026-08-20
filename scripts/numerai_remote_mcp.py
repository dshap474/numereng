#!/usr/bin/env python3
"""Call Numerai's hosted MCP endpoint without installing an MCP server.

USAGE:
    uv run python scripts/numerai_remote_mcp.py tools
    uv run python scripts/numerai_remote_mcp.py call get_current_round \
        --json-args '{"tournament": 8}'
    uv run python scripts/numerai_remote_mcp.py call create_model \
        --json-args '{"name": "example", "tournament": 8}' --confirm-write
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DEFAULT_ENDPOINT = "https://api-tournament.numer.ai/mcp"
PROTOCOL_VERSION = "2024-11-05"
WRITE_TOOLS = {"create_model"}
WRITE_OPERATIONS = {
    "manage_support_requests": {"create"},
    "run_diagnostics": {"create", "delete"},
    "upload_model": {"create", "assign", "trigger"},
}


# --------------------------------------------------------------------------- #
# Authentication
# --------------------------------------------------------------------------- #


def find_env_file(start_dir: Path | None = None) -> Path | None:
    """Find the nearest `.env` file without modifying the environment."""
    search_dir = (start_dir or Path.cwd()).resolve()
    for candidate_dir in (search_dir, *search_dir.parents):
        candidate = candidate_dir / ".env"
        if candidate.is_file():
            return candidate
    return None


def read_env_file(path: Path | None) -> dict[str, str]:
    """Read simple dotenv values needed for Numerai authentication."""
    if path is None:
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'").replace("\\$", "$")
    return values


def resolve_auth_header(
    *,
    environ: Mapping[str, str] | None = None,
    env_file: Path | None = None,
) -> str | None:
    """Resolve a Numerai Authorization value without exposing it in output."""
    environment = os.environ if environ is None else environ
    file_values = read_env_file(env_file if env_file is not None else find_env_file())

    def value(key: str) -> str | None:
        return environment.get(key) or file_values.get(key)

    combined = value("NUMERAI_MCP_AUTH") or value("NUMERAI_API_AUTH")
    if combined:
        return combined if combined.startswith(("Token ", "Bearer ")) else f"Token {combined}"

    public_id = value("NUMERAI_PUBLIC_ID")
    secret_key = value("NUMERAI_SECRET_KEY")
    if public_id and secret_key:
        return f"Token {public_id}${secret_key}"
    return None


# --------------------------------------------------------------------------- #
# MCP requests
# --------------------------------------------------------------------------- #


def is_write_call(tool_name: str, arguments: Mapping[str, Any]) -> bool:
    """Return whether a known MCP call can mutate Numerai state."""
    if tool_name in WRITE_TOOLS:
        return True
    operation = arguments.get("operation")
    if operation in WRITE_OPERATIONS.get(tool_name, set()):
        return True
    if tool_name == "graphql_query":
        query = str(arguments.get("query", "")).lstrip().lower()
        return query.startswith("mutation")
    return False


def rpc_request(
    *,
    endpoint: str,
    method: str,
    params: Mapping[str, Any],
    auth_header: str | None,
    request_id: int = 1,
) -> dict[str, Any]:
    """Send one JSON-RPC request to the hosted Numerai MCP endpoint."""
    payload = json.dumps({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}).encode()
    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
        "User-Agent": "numereng-remote-mcp/1.0",
    }
    if auth_header:
        headers["Authorization"] = auth_header
    request = urllib.request.Request(endpoint, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - fixed/explicit operator endpoint
            result = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"Numerai MCP HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Numerai MCP request failed: {exc.reason}") from exc
    if not isinstance(result, dict):
        raise RuntimeError("Numerai MCP returned a non-object response")
    return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_json_args(value: str) -> dict[str, Any]:
    """Parse a JSON object for an MCP tool call."""
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("--json-args must be a JSON object")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the remote MCP adapter argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--no-auth", action="store_true", help="Do not send discovered Numerai credentials.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("initialize", help="Initialize and inspect server capabilities.")
    subparsers.add_parser("tools", help="List live MCP tools and their input schemas.")

    call_parser = subparsers.add_parser("call", help="Call one live MCP tool.")
    call_parser.add_argument("tool_name")
    call_parser.add_argument("--json-args", type=parse_json_args, default={})
    call_parser.add_argument("--confirm-write", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the hosted Numerai MCP adapter."""
    args = build_parser().parse_args(argv)
    auth_header = None if args.no_auth else resolve_auth_header(env_file=args.env_file)

    if args.command == "initialize":
        method = "initialize"
        params = {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "numereng-remote-mcp", "version": "1.0"},
        }
    elif args.command == "tools":
        method = "tools/list"
        params = {}
    else:
        if is_write_call(args.tool_name, args.json_args) and not args.confirm_write:
            raise SystemExit("Refusing write-capable MCP call without --confirm-write")
        method = "tools/call"
        params = {"name": args.tool_name, "arguments": args.json_args}

    result = rpc_request(endpoint=args.endpoint, method=method, params=params, auth_header=auth_header)
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    raise SystemExit(main())
