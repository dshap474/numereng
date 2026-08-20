"""Test the standalone hosted Numerai MCP adapter.

USAGE:
    uv run pytest tests/unit/test_numerai_remote_mcp.py -q
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

# --------------------------------------------------------------------------- #
# Test setup
# --------------------------------------------------------------------------- #

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "numerai_remote_mcp.py"


def load_adapter() -> ModuleType:
    """Load the standalone script as a module for focused unit tests."""
    spec = importlib.util.spec_from_file_location("numerai_remote_mcp", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_resolve_auth_header_prefers_combined_token(tmp_path: Path) -> None:
    adapter = load_adapter()
    env_file = tmp_path / ".env"
    env_file.write_text("NUMERAI_MCP_AUTH=Token public\\$private\n", encoding="utf-8")

    assert adapter.resolve_auth_header(environ={}, env_file=env_file) == "Token public$private"


def test_resolve_auth_header_combines_public_and_secret(tmp_path: Path) -> None:
    adapter = load_adapter()
    env_file = tmp_path / ".env"
    env_file.write_text("NUMERAI_PUBLIC_ID=public\nNUMERAI_SECRET_KEY=private\n", encoding="utf-8")

    assert adapter.resolve_auth_header(environ={}, env_file=env_file) == "Token public$private"


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("create_model", {"name": "example"}),
        ("manage_support_requests", {"operation": "create"}),
        ("run_diagnostics", {"operation": "delete"}),
        ("upload_model", {"operation": "assign"}),
        ("graphql_query", {"query": "mutation { example }"}),
    ],
)
def test_is_write_call_detects_mutations(tool_name: str, arguments: dict[str, object]) -> None:
    adapter = load_adapter()

    assert adapter.is_write_call(tool_name, arguments)


def test_is_write_call_allows_reads() -> None:
    adapter = load_adapter()

    assert not adapter.is_write_call("get_current_round", {"tournament": 8})
    assert not adapter.is_write_call("graphql_query", {"query": "query { tournaments { name } }"})
