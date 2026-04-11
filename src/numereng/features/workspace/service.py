"""Workspace bootstrap helpers for fresh numereng installs."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from numereng.assets import shipped_skill_ids, shipped_skills_root
from numereng.features.store.layout import resolve_workspace_layout


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
        "\n".join(
            [
                ".numereng/numereng.db",
                ".numereng/numereng.db-shm",
                ".numereng/numereng.db-wal",
                ".numereng/runs/",
                ".numereng/datasets/",
                ".numereng/cache/",
                ".numereng/tmp/",
                ".numereng/remote_ops/",
                ".venv/",
                "node_modules/",
                "",
            ]
        ),
        created=created,
        skipped=skipped,
    )
    _write_if_missing(
        layout.workspace_root / "README.md",
        "\n".join(
            [
                "# numereng workspace",
                "",
                "This workspace uses visible authoring roots plus a hidden `.numereng/` runtime store.",
                "",
                "Visible workspace roots:",
                "- `experiments/`",
                "- `notes/`",
                "- `custom_models/`",
                "- `research_programs/`",
                "- `.agents/skills/`",
                "",
                "Numereng-managed state under `.numereng/`:",
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
                (
                    "- User-authored workspace surfaces live in `experiments/`, `notes/`, "
                    "`custom_models/`, `research_programs/`, and `.agents/skills/`."
                ),
                "- Keep numereng-managed runtime state under `.numereng/`.",
                "- `numereng viz` is monitor-only. Launch runs and mutations via the CLI or API, not the dashboard.",
                "- `research init` requires `--program`.",
                "- Submission and neutralization each accept exactly one source: `run_id` or `predictions_path`.",
                "- Training and HPO configs are JSON-only and reject unknown keys.",
                "- `custom_models/` is the canonical runtime discovery root for custom model plugins.",
                "- `research_programs/` is resolved before packaged built-ins.",
                "- Do not assume a `numereng` source checkout is required to use this workspace.",
                "",
            ]
        )
        + "\n",
        created=created,
        skipped=skipped,
    )
    _write_if_missing(
        layout.workspace_root / ".agents" / "README.md",
        "Workspace-scoped agent assets live here.\n",
        created=created,
        skipped=skipped,
    )
    _write_if_missing(
        layout.workspace_root / ".agents" / "AGENTS.md",
        "- Treat `.agents/skills/` as the canonical harness-agnostic skill root.\n",
        created=created,
        skipped=skipped,
    )
    _write_if_missing(
        layout.skills_root / "AGENTS.md",
        "- These are packaged numereng-owned skills copied into this workspace.\n",
        created=created,
        skipped=skipped,
    )
    _write_if_missing(
        layout.custom_models_root / "README.md",
        (
            "Place user-authored model plugins here. `numereng` resolves custom model plugins only "
            "from this directory by default.\n"
        ),
        created=created,
        skipped=skipped,
    )
    _write_if_missing(
        layout.custom_models_root / "AGENTS.md",
        "- Add user-authored model plugin modules here.\n",
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
        layout.research_programs_root / "AGENTS.md",
        "- Add user-authored research program markdown here.\n",
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
        layout.experiments_root / "AGENTS.md",
        "- Keep experiment manifests, configs, notes, and launchers under `experiments/`.\n",
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
        layout.notes_root / "__RESEARCH_MEMORY__" / "CURRENT.md",
        "# Current Research Frontier\n\nCapture the active synthesis, next action, and open questions here.\n",
        created=created,
        skipped=skipped,
    )
    _write_if_missing(
        layout.notes_root / "__RESEARCH_MEMORY__" / "AGENTS.md",
        "- Keep rolling research-memory notes here.\n",
        created=created,
        skipped=skipped,
    )
    _write_if_missing(
        layout.store_root / "README.md",
        "Numereng-managed runtime state lives here.\n",
        created=created,
        skipped=skipped,
    )
    _write_if_missing(
        layout.store_root / "AGENTS.md",
        "- Treat `.numereng/` as the canonical numereng-managed state root.\n",
        created=created,
        skipped=skipped,
    )
    _write_if_missing(
        layout.skills_root / "README.md",
        "Numereng-owned agent skills installed for this workspace.\n",
        created=created,
        skipped=skipped,
    )

    template_model_path = _package_root() / "features" / "models" / "custom_models" / "template_model.py"
    _copy_file_if_missing(
        template_model_path,
        layout.custom_models_root / "template_model.py",
        created=created,
        skipped=skipped,
    )
    research_program_path = (
        _package_root() / "features" / "agentic_research" / "programs" / "numerai-experiment-loop.md"
    )
    _copy_file_if_missing(
        research_program_path,
        layout.research_programs_root / "numerai-experiment-loop.md",
        created=created,
        skipped=skipped,
    )

    for runtime_dir in (
        layout.store_root / "runs",
        layout.store_root / "datasets",
        layout.store_root / "cache",
        layout.store_root / "tmp",
        layout.store_root / "remote_ops",
    ):
        _write_if_missing(runtime_dir / ".gitkeep", "", created=created, skipped=skipped)

    installed_skill_ids: list[str] = []
    skills_source_root = shipped_skills_root()
    if skills_source_root.is_dir():
        for skill_id in shipped_skill_ids():
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


def _package_root() -> Path:
    return Path(__file__).resolve().parents[2]
