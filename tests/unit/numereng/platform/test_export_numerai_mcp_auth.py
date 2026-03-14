from __future__ import annotations

from pathlib import Path

import pytest

from numereng.platform.errors import NumeraiMcpAuthError
from numereng.platform.export_numerai_mcp_auth import (
    build_export_command,
    find_env_file,
    main,
    read_env_value,
    resolve_numerai_mcp_auth,
)


def test_find_env_file_walks_up_to_repo_root(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text('NUMERAI_MCP_AUTH="Token public\\$private"\n', encoding="utf-8")
    nested_dir = tmp_path / "src" / "numereng" / "platform"
    nested_dir.mkdir(parents=True)

    assert find_env_file(start_dir=nested_dir) == env_file


def test_find_env_file_raises_when_missing(tmp_path: Path) -> None:
    with pytest.raises(NumeraiMcpAuthError, match="numerai_mcp_auth_env_file_not_found"):
        find_env_file(start_dir=tmp_path)


def test_read_env_value_unescapes_escaped_dollar(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        '# comment\nexport NUMERAI_MCP_AUTH="Token public\\$private"\n',
        encoding="utf-8",
    )

    assert read_env_value(env_file, key="NUMERAI_MCP_AUTH") == "Token public$private"


def test_resolve_numerai_mcp_auth_prefers_existing_environment(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text('NUMERAI_MCP_AUTH="Token file\\$value"\n', encoding="utf-8")

    resolved = resolve_numerai_mcp_auth(
        env_file=env_file,
        environ={"NUMERAI_MCP_AUTH": "Token env$value"},
    )

    assert resolved == "Token env$value"


def test_resolve_numerai_mcp_auth_raises_when_missing_key(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OTHER_KEY=value\n", encoding="utf-8")

    with pytest.raises(NumeraiMcpAuthError, match="numerai_mcp_auth_not_found"):
        resolve_numerai_mcp_auth(env_file=env_file, environ={})


def test_build_export_command_quotes_value() -> None:
    assert build_export_command("Token public$private") == "export NUMERAI_MCP_AUTH='Token public$private'"


def test_main_prints_shell_export_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text('NUMERAI_MCP_AUTH="Token public\\$private"\n', encoding="utf-8")

    exit_code = main(["--env-file", str(env_file)])

    assert exit_code == 0
    assert capsys.readouterr().out == "export NUMERAI_MCP_AUTH='Token public$private'\n"
