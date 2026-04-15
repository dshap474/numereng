"""Workspace-local uv project management for installed numereng workspaces."""

from __future__ import annotations

import json
import shutil
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from numereng import __version__
from numereng.features.store.layout import resolve_path, resolve_workspace_layout
from numereng.platform.errors import PackageError

WorkspaceRuntimeSource = Literal["pypi", "path"]

_MANAGED_SCHEMA_VERSION = 1
_PYTHON_BASELINE = "3.12"
_REQUIRED_RUNTIME_MODULES: tuple[str, ...] = ("numereng", "cloudpickle")


@dataclass(frozen=True)
class WorkspaceEnvConfig:
    runtime_source: WorkspaceRuntimeSource
    runtime_path: Path | None
    with_training: bool
    with_mlops: bool

    @property
    def extras(self) -> tuple[str, ...]:
        extras: list[str] = []
        if self.with_training:
            extras.append("training")
        if self.with_mlops:
            extras.append("mlops")
        return tuple(extras)

    @property
    def dependency_spec(self) -> str:
        extras = self.extras
        package_name = "numereng"
        if extras:
            package_name = f"{package_name}[{','.join(extras)}]"
        if self.runtime_source == "pypi":
            return f"{package_name}=={__version__}"
        return package_name


@dataclass(frozen=True)
class WorkspaceSyncResult:
    workspace_root: Path
    store_root: Path
    workspace_project_path: Path
    python_version_path: Path
    venv_path: Path
    created_paths: tuple[Path, ...]
    updated_paths: tuple[Path, ...]
    runtime_source: WorkspaceRuntimeSource
    runtime_path: Path | None
    extras: tuple[str, ...]
    dependency_spec: str
    installed_numereng_version: str
    verified_dependencies: tuple[str, ...]


def sync_workspace_environment(
    *,
    workspace_root: str | Path = ".",
    runtime_source: WorkspaceRuntimeSource | None = None,
    runtime_path: str | Path | None = None,
    with_training: bool | None = None,
    with_mlops: bool | None = None,
) -> WorkspaceSyncResult:
    """Ensure the workspace owns a local uv-managed numereng runtime."""

    layout = resolve_workspace_layout(workspace_root)
    created: list[Path] = []
    updated: list[Path] = []

    pyproject_path = layout.workspace_root / "pyproject.toml"
    python_version_path = layout.workspace_root / ".python-version"
    venv_path = layout.workspace_root / ".venv"

    existing_config = _load_managed_workspace_config(pyproject_path)
    resolved_config = _resolve_workspace_env_config(
        existing_config=existing_config,
        runtime_source=runtime_source,
        runtime_path=runtime_path,
        with_training=with_training,
        with_mlops=with_mlops,
    )

    _write_workspace_pyproject(
        pyproject_path,
        config=resolved_config,
        created=created,
        updated=updated,
    )
    _write_python_version_file(python_version_path, created=created)

    uv_path = shutil.which("uv")
    if uv_path is None:
        raise PackageError("workspace_uv_missing")

    _run_uv_sync(uv_path=uv_path, workspace_root=layout.workspace_root)
    installed_version = _verify_workspace_runtime(uv_path=uv_path, workspace_root=layout.workspace_root)
    if not venv_path.is_dir():
        raise PackageError("workspace_venv_missing_after_sync")

    return WorkspaceSyncResult(
        workspace_root=layout.workspace_root,
        store_root=layout.store_root,
        workspace_project_path=pyproject_path,
        python_version_path=python_version_path,
        venv_path=venv_path,
        created_paths=tuple(created),
        updated_paths=tuple(updated),
        runtime_source=resolved_config.runtime_source,
        runtime_path=resolved_config.runtime_path,
        extras=resolved_config.extras,
        dependency_spec=resolved_config.dependency_spec,
        installed_numereng_version=installed_version,
        verified_dependencies=_REQUIRED_RUNTIME_MODULES,
    )


def _load_managed_workspace_config(pyproject_path: Path) -> WorkspaceEnvConfig | None:
    if not pyproject_path.exists():
        return None
    try:
        payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise PackageError(f"workspace_pyproject_invalid:{exc}") from exc

    tool = payload.get("tool")
    if not isinstance(tool, dict):
        raise PackageError("workspace_pyproject_unmanaged")
    numereng_tool = tool.get("numereng")
    if not isinstance(numereng_tool, dict):
        raise PackageError("workspace_pyproject_unmanaged")
    workspace_section = numereng_tool.get("workspace")
    if not isinstance(workspace_section, dict):
        raise PackageError("workspace_pyproject_unmanaged")
    if workspace_section.get("managed") is not True:
        raise PackageError("workspace_pyproject_unmanaged")

    runtime_source = workspace_section.get("runtime_source", "pypi")
    if runtime_source not in {"pypi", "path"}:
        raise PackageError("workspace_pyproject_invalid")

    raw_runtime_path = workspace_section.get("runtime_path")
    resolved_runtime_path: Path | None = None
    if raw_runtime_path:
        resolved_runtime_path = resolve_path(str(raw_runtime_path))

    extras = workspace_section.get("extras", [])
    if extras is None:
        extras = []
    if not isinstance(extras, list) or any(not isinstance(item, str) for item in extras):
        raise PackageError("workspace_pyproject_invalid")
    normalized_extras = {item.strip() for item in extras}
    unknown_extras = normalized_extras - {"training", "mlops"}
    if unknown_extras:
        raise PackageError("workspace_pyproject_invalid")

    return WorkspaceEnvConfig(
        runtime_source=runtime_source,
        runtime_path=resolved_runtime_path,
        with_training="training" in normalized_extras,
        with_mlops="mlops" in normalized_extras,
    )


def _resolve_workspace_env_config(
    *,
    existing_config: WorkspaceEnvConfig | None,
    runtime_source: WorkspaceRuntimeSource | None,
    runtime_path: str | Path | None,
    with_training: bool | None,
    with_mlops: bool | None,
) -> WorkspaceEnvConfig:
    if runtime_source is None:
        resolved_source: WorkspaceRuntimeSource = (
            existing_config.runtime_source if existing_config is not None else "pypi"
        )
    else:
        resolved_source = runtime_source

    if runtime_path is not None and resolved_source != "path":
        raise PackageError("workspace_runtime_path_requires_path_source")

    if resolved_source == "path":
        candidate_runtime_path = (
            resolve_path(runtime_path)
            if runtime_path is not None
            else existing_config.runtime_path
            if existing_config is not None
            else None
        )
        if candidate_runtime_path is None:
            raise PackageError("workspace_runtime_path_required")
        if not candidate_runtime_path.exists():
            raise PackageError(f"workspace_runtime_path_missing:{candidate_runtime_path}")
    else:
        candidate_runtime_path = None

    resolved_with_training = (
        with_training
        if with_training is not None
        else existing_config.with_training
        if existing_config is not None
        else False
    )
    resolved_with_mlops = (
        with_mlops if with_mlops is not None else existing_config.with_mlops if existing_config is not None else False
    )

    return WorkspaceEnvConfig(
        runtime_source=resolved_source,
        runtime_path=candidate_runtime_path,
        with_training=resolved_with_training,
        with_mlops=resolved_with_mlops,
    )


def _write_workspace_pyproject(
    pyproject_path: Path,
    *,
    config: WorkspaceEnvConfig,
    created: list[Path],
    updated: list[Path],
) -> None:
    rendered = _render_workspace_pyproject(config)
    if pyproject_path.exists():
        current = pyproject_path.read_text(encoding="utf-8")
        if current == rendered:
            return
        pyproject_path.write_text(rendered, encoding="utf-8")
        updated.append(pyproject_path)
        return
    pyproject_path.parent.mkdir(parents=True, exist_ok=True)
    pyproject_path.write_text(rendered, encoding="utf-8")
    created.append(pyproject_path)


def _write_python_version_file(path: Path, *, created: list[Path]) -> None:
    if path.exists():
        return
    path.write_text(f"{_PYTHON_BASELINE}\n", encoding="utf-8")
    created.append(path)


def _render_workspace_pyproject(config: WorkspaceEnvConfig) -> str:
    extras = ", ".join(f'"{item}"' for item in config.extras)
    lines = [
        "# Managed by numereng. Update via `numereng init` or `numereng workspace sync`.",
        "",
        "[project]",
        'name = "numereng-workspace"',
        'version = "0.0.0"',
        f'requires-python = ">={_PYTHON_BASELINE}"',
        "dependencies = [",
        f'    "{config.dependency_spec}",',
        "]",
        "",
        "[tool.uv]",
        "package = false",
        "",
        "[tool.numereng.workspace]",
        "managed = true",
        f"schema_version = {_MANAGED_SCHEMA_VERSION}",
        f'runtime_source = "{config.runtime_source}"',
        f"extras = [{extras}]",
    ]
    if config.runtime_path is not None:
        lines.append(f'runtime_path = "{_escape_toml_string(str(config.runtime_path))}"')
    if config.runtime_source == "path" and config.runtime_path is not None:
        lines.extend(
            [
                "",
                "[tool.uv.sources]",
                (f'numereng = {{ path = "{_escape_toml_string(str(config.runtime_path))}", editable = true }}'),
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _run_uv_sync(*, uv_path: str, workspace_root: Path) -> None:
    result = subprocess.run(
        [uv_path, "sync"],
        cwd=workspace_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise PackageError(f"workspace_env_sync_failed:{_best_effort_message(result)}")


def _verify_workspace_runtime(*, uv_path: str, workspace_root: Path) -> str:
    verification_script = (
        "import json; import cloudpickle; import numereng; "
        'print(json.dumps({"numereng_version": numereng.__version__, '
        '"cloudpickle": bool(cloudpickle.__version__ or True)}))'
    )
    result = subprocess.run(
        [uv_path, "run", "python", "-c", verification_script],
        cwd=workspace_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise PackageError(f"workspace_env_verify_failed:{_best_effort_message(result)}")
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise PackageError("workspace_env_verify_failed:invalid_verification_payload") from exc
    version = payload.get("numereng_version")
    if not isinstance(version, str) or not version:
        raise PackageError("workspace_env_verify_failed:missing_numereng_version")
    return version


def _escape_toml_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _best_effort_message(result: subprocess.CompletedProcess[str]) -> str:
    for candidate in (result.stderr, result.stdout):
        if candidate:
            for line in candidate.splitlines():
                stripped = line.strip()
                if stripped:
                    return stripped
    return f"exit_code_{result.returncode}"


__all__ = [
    "WorkspaceEnvConfig",
    "WorkspaceRuntimeSource",
    "WorkspaceSyncResult",
    "sync_workspace_environment",
]
