from __future__ import annotations

from pathlib import Path

from numereng.assets import shipped_skill_ids
from numereng.features.workspace.runtime import WorkspaceSyncResult
from numereng.features.workspace.service import init_workspace


def _fake_sync_workspace_environment(*, workspace_root: str | Path = ".", **_: object) -> WorkspaceSyncResult:
    workspace_path = Path(workspace_root).resolve()
    pyproject_path = workspace_path / "pyproject.toml"
    pyproject_path.write_text(
        '[project]\nname = "numereng-workspace"\nversion = "0.0.0"\n',
        encoding="utf-8",
    )
    python_version_path = workspace_path / ".python-version"
    python_version_path.write_text("3.12\n", encoding="utf-8")
    venv_path = workspace_path / ".venv"
    venv_path.mkdir(exist_ok=True)
    return WorkspaceSyncResult(
        workspace_root=workspace_path,
        store_root=(workspace_path / ".numereng").resolve(),
        workspace_project_path=pyproject_path.resolve(),
        python_version_path=python_version_path.resolve(),
        venv_path=venv_path.resolve(),
        created_paths=(pyproject_path.resolve(), python_version_path.resolve(), venv_path.resolve()),
        updated_paths=(),
        runtime_source="pypi",
        runtime_path=None,
        extras=(),
        dependency_spec="numereng==0.0.0-test",
        installed_numereng_version="0.0.0-test",
        verified_dependencies=("numereng", "cloudpickle"),
    )


def test_init_workspace_creates_canonical_fresh_install_layout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "numereng.features.workspace.service.sync_workspace_environment",
        _fake_sync_workspace_environment,
    )
    workspace_root = tmp_path / "numerai-dev"

    result = init_workspace(workspace_root=workspace_root)

    assert result.workspace_root == workspace_root.resolve()
    assert result.store_root == (workspace_root / ".numereng").resolve()

    for path in (
        workspace_root / "experiments",
        workspace_root / "notes",
        workspace_root / "notes" / "__RESEARCH_MEMORY__",
        workspace_root / "custom_models",
        workspace_root / "research_programs",
        workspace_root / ".agents" / "skills",
        workspace_root / ".numereng" / "runs",
        workspace_root / ".numereng" / "datasets",
        workspace_root / ".numereng" / "cache",
        workspace_root / ".numereng" / "tmp",
        workspace_root / ".numereng" / "remote_ops",
    ):
        assert path.is_dir(), path

    for path in (
        workspace_root / ".gitignore",
        workspace_root / "README.md",
        workspace_root / "AGENTS.md",
        workspace_root / "pyproject.toml",
        workspace_root / ".python-version",
        workspace_root / ".agents" / "README.md",
        workspace_root / ".agents" / "AGENTS.md",
        workspace_root / ".agents" / "skills" / "AGENTS.md",
        workspace_root / "custom_models" / "README.md",
        workspace_root / "custom_models" / "AGENTS.md",
        workspace_root / "custom_models" / "template_model.py",
        workspace_root / "research_programs" / "README.md",
        workspace_root / "research_programs" / "AGENTS.md",
        workspace_root / "research_programs" / "numerai-experiment-loop.md",
        workspace_root / "experiments" / "README.md",
        workspace_root / "experiments" / "AGENTS.md",
        workspace_root / "notes" / "README.md",
        workspace_root / "notes" / "__RESEARCH_MEMORY__" / "AGENTS.md",
        workspace_root / "notes" / "__RESEARCH_MEMORY__" / "README.md",
        workspace_root / "notes" / "__RESEARCH_MEMORY__" / "CURRENT.md",
        workspace_root / ".numereng" / "README.md",
        workspace_root / ".numereng" / "AGENTS.md",
        workspace_root / ".agents" / "skills" / "README.md",
        workspace_root / ".numereng" / "runs" / ".gitkeep",
        workspace_root / ".numereng" / "datasets" / ".gitkeep",
        workspace_root / ".numereng" / "cache" / ".gitkeep",
        workspace_root / ".numereng" / "tmp" / ".gitkeep",
        workspace_root / ".numereng" / "remote_ops" / ".gitkeep",
    ):
        assert path.is_file(), path
    assert (workspace_root / ".venv").is_dir()

    workspace_agents = (workspace_root / "AGENTS.md").read_text(encoding="utf-8")
    assert "numereng viz" in workspace_agents
    assert "Use `uv run numereng ...`" in workspace_agents
    assert "run_id` or `predictions_path`" in workspace_agents
    assert "Do not assume a `numereng` source checkout is required" in workspace_agents
    workspace_readme = (workspace_root / "README.md").read_text(encoding="utf-8")
    assert "local `uv` project" in workspace_readme
    assert "uv run numereng --help" in workspace_readme

    assert result.installed_skill_ids == shipped_skill_ids()
    assert result.installed_skill_ids == (
        "experiment-design",
        "numereng-experiment-ops",
        "numerai-api-ops",
        "implement-custom-model",
        "store-ops",
    )
    assert (workspace_root / ".agents" / "skills" / "numereng-experiment-ops" / "SKILL.md").is_file()
    assert result.sync_result.runtime_source == "pypi"
    assert result.sync_result.installed_numereng_version == "0.0.0-test"
    assert result.sync_result.verified_dependencies == ("numereng", "cloudpickle")


def test_init_workspace_is_idempotent_and_preserves_existing_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "numereng.features.workspace.service.sync_workspace_environment",
        _fake_sync_workspace_environment,
    )
    workspace_root = tmp_path / "numerai-dev"

    first = init_workspace(workspace_root=workspace_root)
    agents_path = workspace_root / "AGENTS.md"
    agents_path.write_text("custom agents\n", encoding="utf-8")

    second = init_workspace(workspace_root=workspace_root)

    assert agents_path.read_text(encoding="utf-8") == "custom agents\n"
    assert agents_path.resolve() in second.skipped_existing_paths
    assert (workspace_root / "experiments").resolve() in first.created_paths
    assert not any(path == agents_path.resolve() for path in second.created_paths)
