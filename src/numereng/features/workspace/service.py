"""Workspace bootstrap helpers for fresh numereng installs."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from numereng.features.store import resolve_workspace_layout

_SHIPPED_SKILLS: tuple[str, ...] = (
    "config-schema",
    "experiment-design",
    "implement-custom-model",
    "kaggle-gm-workflow",
    "numerai-api-ops",
    "numerai-docs",
    "numerai-research",
    "numereng-experiment-ops",
    "research-memory",
    "store-ops",
)


@dataclass(frozen=True)
class WorkspaceInitResult:
    workspace_root: Path
    store_root: Path
    created_paths: tuple[Path, ...]
    skipped_existing_paths: tuple[Path, ...]
    installed_skill_ids: tuple[str, ...]


def init_workspace(*, workspace_root: str | Path = ".") -> WorkspaceInitResult:
    """Create one canonical numereng workspace scaffold without overwriting user files."""

    layout = resolve_workspace_layout(workspace_root)
    created: list[Path] = []
    skipped: list[Path] = []

    for directory in (
        layout.workspace_root,
        layout.experiments_root,
        layout.notes_root,
        layout.notes_root / "__RESEARCH_MEMORY__",
        layout.custom_models_root,
        layout.research_programs_root,
        layout.skills_root.parent,
        layout.skills_root,
        layout.store_root,
        layout.store_root / "runs",
        layout.store_root / "datasets",
        layout.store_root / "cache",
        layout.store_root / "tmp",
        layout.store_root / "remote_ops",
    ):
        if directory.exists():
            skipped.append(directory)
            continue
        directory.mkdir(parents=True, exist_ok=True)
        created.append(directory)

    _write_if_missing(
        layout.workspace_root / ".gitignore",
        ".numereng/\n.venv/\nnode_modules/\n",
        created=created,
        skipped=skipped,
    )
    _write_if_missing(
        layout.workspace_root / "README.md",
        "\n".join(
            [
                "# numereng workspace",
                "",
                "This workspace uses root-level authoring surfaces and a hidden `.numereng/` runtime store.",
                "",
                "Visible workspace roots:",
                "- `experiments/`",
                "- `notes/`",
                "- `custom_models/`",
                "- `research_programs/`",
                "- `.agents/skills/`",
                "",
                "Hidden runtime state:",
                "- `.numereng/runs/`",
                "- `.numereng/datasets/`",
                "- `.numereng/cache/`",
                "- `.numereng/tmp/`",
                "- `.numereng/remote_ops/`",
                "- `.numereng/numereng.db`",
                "",
            ]
        )
        + "\n",
        created=created,
        skipped=skipped,
    )
    _write_if_missing(
        layout.workspace_root / "AGENTS.md",
        "\n".join(
            [
                "# numereng workspace",
                "",
                "- Treat this directory as the canonical workspace root.",
                "- Keep user-authored experiment and notes content at workspace root.",
                "- Keep runtime state under `.numereng/`.",
                "",
            ]
        )
        + "\n",
        created=created,
        skipped=skipped,
    )
    _write_if_missing(
        layout.custom_models_root / "README.md",
        "Place user-authored model plugins here. `numereng` resolves this directory before packaged built-ins.\n",
        created=created,
        skipped=skipped,
    )
    _write_if_missing(
        layout.research_programs_root / "README.md",
        (
            "Place user-authored research program markdown here. "
            "`numereng` resolves this directory before packaged built-ins.\n"
        ),
        created=created,
        skipped=skipped,
    )
    _write_if_missing(
        layout.experiments_root / "README.md",
        "Experiment manifests, configs, notes, and launcher scripts live here.\n",
        created=created,
        skipped=skipped,
    )
    _write_if_missing(
        layout.notes_root / "README.md",
        "Workspace notes and research memory live here.\n",
        created=created,
        skipped=skipped,
    )
    _write_if_missing(
        layout.notes_root / "__RESEARCH_MEMORY__" / "README.md",
        "Canonical rolling research-memory notes live here.\n",
        created=created,
        skipped=skipped,
    )
    _write_if_missing(
        layout.store_root / "README.md",
        "Hidden numereng runtime state. Do not place user-authored experiments or notes here.\n",
        created=created,
        skipped=skipped,
    )
    _write_if_missing(
        layout.skills_root / "README.md",
        "Numereng-owned agent skills installed for this workspace.\n",
        created=created,
        skipped=skipped,
    )

    template_model_path = (
        _repo_root() / "src" / "numereng" / "features" / "models" / "custom_models" / "template_model.py"
    )
    _copy_file_if_missing(
        template_model_path,
        layout.custom_models_root / "template_model.py",
        created=created,
        skipped=skipped,
    )

    installed_skill_ids: list[str] = []
    skills_source_root = _repo_root() / ".codex" / "skills"
    if skills_source_root.is_dir():
        for skill_id in _SHIPPED_SKILLS:
            source = skills_source_root / skill_id
            destination = layout.skills_root / skill_id
            if not source.is_dir():
                continue
            if destination.exists():
                skipped.append(destination)
            else:
                shutil.copytree(source, destination)
                created.append(destination)
            installed_skill_ids.append(skill_id)

    return WorkspaceInitResult(
        workspace_root=layout.workspace_root,
        store_root=layout.store_root,
        created_paths=tuple(created),
        skipped_existing_paths=tuple(skipped),
        installed_skill_ids=tuple(installed_skill_ids),
    )


def _write_if_missing(path: Path, content: str, *, created: list[Path], skipped: list[Path]) -> None:
    if path.exists():
        skipped.append(path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    created.append(path)


def _copy_file_if_missing(source: Path, destination: Path, *, created: list[Path], skipped: list[Path]) -> None:
    if destination.exists():
        skipped.append(destination)
        return
    if not source.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    created.append(destination)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]
