from __future__ import annotations

from pathlib import Path

from numereng.assets import shipped_skill_ids
from numereng.features.workspace.service import init_workspace


def test_init_workspace_creates_canonical_fresh_install_layout(tmp_path: Path) -> None:
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

    workspace_agents = (workspace_root / "AGENTS.md").read_text(encoding="utf-8")
    assert "numereng viz" in workspace_agents
    assert "run_id` or `predictions_path`" in workspace_agents
    assert "Do not assume a `numereng` source checkout is required" in workspace_agents

    assert result.installed_skill_ids == shipped_skill_ids()
    assert result.installed_skill_ids == (
        "experiment-design",
        "numereng-experiment-ops",
        "numerai-api-ops",
        "implement-custom-model",
        "store-ops",
    )
    assert (workspace_root / ".agents" / "skills" / "numereng-experiment-ops" / "SKILL.md").is_file()


def test_init_workspace_is_idempotent_and_preserves_existing_files(tmp_path: Path) -> None:
    workspace_root = tmp_path / "numerai-dev"

    first = init_workspace(workspace_root=workspace_root)
    agents_path = workspace_root / "AGENTS.md"
    agents_path.write_text("custom agents\n", encoding="utf-8")

    second = init_workspace(workspace_root=workspace_root)

    assert agents_path.read_text(encoding="utf-8") == "custom agents\n"
    assert agents_path.resolve() in second.skipped_existing_paths
    assert (workspace_root / "experiments").resolve() in first.created_paths
    assert not any(path == agents_path.resolve() for path in second.created_paths)
