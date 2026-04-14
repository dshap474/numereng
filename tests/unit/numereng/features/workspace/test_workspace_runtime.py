from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from numereng.features.workspace.runtime import sync_workspace_environment
from numereng.platform.errors import PackageError


def _uv_path(name: str) -> str | None:
    return "/opt/uv" if name == "uv" else None


def test_sync_workspace_environment_creates_managed_uv_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "numerai-dev"
    workspace_root.mkdir()

    calls: list[tuple[str, ...]] = []

    def fake_run(
        cmd: list[str],
        *,
        cwd: Path,
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert capture_output is True
        assert text is True
        assert check is False
        assert cwd == workspace_root
        calls.append(tuple(cmd))
        if cmd == ["/opt/uv", "sync"]:
            (workspace_root / ".venv").mkdir(exist_ok=True)
            return subprocess.CompletedProcess(cmd, 0, stdout="synced\n", stderr="")
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout='{"numereng_version":"1.2.3"}\n',
            stderr="",
        )

    monkeypatch.setattr("numereng.features.workspace.runtime.shutil.which", _uv_path)
    monkeypatch.setattr("numereng.features.workspace.runtime.subprocess.run", fake_run)

    result = sync_workspace_environment(workspace_root=workspace_root)

    pyproject_path = workspace_root / "pyproject.toml"
    payload = pyproject_path.read_text(encoding="utf-8")
    assert "[tool.numereng.workspace]" in payload
    assert 'runtime_source = "pypi"' in payload
    assert 'numereng==' in payload
    assert (workspace_root / ".python-version").read_text(encoding="utf-8") == "3.12\n"
    assert result.runtime_source == "pypi"
    assert result.installed_numereng_version == "1.2.3"
    assert result.verified_dependencies == ("numereng", "cloudpickle")
    assert calls[0] == ("/opt/uv", "sync")
    assert calls[1][:4] == ("/opt/uv", "run", "python", "-c")


def test_sync_workspace_environment_supports_local_path_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "numerai-dev"
    runtime_path = tmp_path / "numereng-src"
    workspace_root.mkdir()
    runtime_path.mkdir()

    def fake_run(
        cmd: list[str],
        *,
        cwd: Path,
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        if cmd == ["/opt/uv", "sync"]:
            (workspace_root / ".venv").mkdir(exist_ok=True)
            return subprocess.CompletedProcess(cmd, 0, stdout="synced\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout='{"numereng_version":"1.2.3"}\n', stderr="")

    monkeypatch.setattr("numereng.features.workspace.runtime.shutil.which", _uv_path)
    monkeypatch.setattr("numereng.features.workspace.runtime.subprocess.run", fake_run)

    result = sync_workspace_environment(
        workspace_root=workspace_root,
        runtime_source="path",
        runtime_path=runtime_path,
        with_training=True,
    )

    payload = (workspace_root / "pyproject.toml").read_text(encoding="utf-8")
    assert "[tool.uv.sources]" in payload
    assert f'path = "{str(runtime_path)}"' in payload
    assert 'runtime_source = "path"' in payload
    assert '"training"' in payload
    assert result.runtime_source == "path"
    assert result.runtime_path == runtime_path.resolve()
    assert result.extras == ("training",)


def test_sync_workspace_environment_rejects_unmanaged_pyproject(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "numerai-dev"
    workspace_root.mkdir()
    (workspace_root / "pyproject.toml").write_text(
        '[project]\nname = "custom-workspace"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )

    with pytest.raises(PackageError, match="workspace_pyproject_unmanaged"):
        sync_workspace_environment(workspace_root=workspace_root)


def test_sync_workspace_environment_requires_path_source_for_runtime_path(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "numerai-dev"
    workspace_root.mkdir()

    with pytest.raises(PackageError, match="workspace_runtime_path_requires_path_source"):
        sync_workspace_environment(
            workspace_root=workspace_root,
            runtime_path=tmp_path / "numereng-src",
        )


def test_sync_workspace_environment_reports_uv_sync_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "numerai-dev"
    workspace_root.mkdir()

    def fake_run(
        cmd: list[str],
        *,
        cwd: Path,
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        _ = cwd, capture_output, text, check
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="resolver failed")

    monkeypatch.setattr("numereng.features.workspace.runtime.shutil.which", _uv_path)
    monkeypatch.setattr("numereng.features.workspace.runtime.subprocess.run", fake_run)

    with pytest.raises(PackageError, match="workspace_env_sync_failed:resolver failed"):
        sync_workspace_environment(workspace_root=workspace_root)
