"""Canonical workspace/store layout constants and path-guard helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

CANONICAL_STORE_TOP_LEVEL_DIRS: tuple[str, ...] = (
    "runs",
    "datasets",
    "cloud",
    "experiments",
    "notes",
    "cache",
    "tmp",
    "remote_ops",
)
CANONICAL_WORKSPACE_TOP_LEVEL_DIRS: tuple[str, ...] = (".agents",)
CANONICAL_STORE_TOP_LEVEL_FILES: tuple[str, ...] = (
    "numereng.db",
    "numereng.db-shm",
    "numereng.db-wal",
)
TARGETED_STRAY_DIRS: tuple[str, ...] = (
    "logs",
    "modal_smoke_data",
    "smoke_live_check",
)
CANONICAL_STORE_DIRNAME = ".numereng"
_FEATURES_ROOT = Path(__file__).resolve().parents[1]
_BUILTIN_CUSTOM_MODELS_ROOT = _FEATURES_ROOT / "models" / "custom_models"
_BUILTIN_RESEARCH_PROGRAMS_ROOT = _FEATURES_ROOT / "agentic_research" / "programs"


@dataclass(frozen=True)
class WorkspaceLayout:
    """Canonical resolved paths for one numereng workspace."""

    workspace_root: Path
    store_root: Path
    experiments_root: Path
    notes_root: Path
    custom_models_root: Path
    research_programs_root: Path
    skills_root: Path


def resolve_path(value: str | Path) -> Path:
    """Resolve one filesystem path with user-home expansion."""
    return Path(value).expanduser().resolve()


def resolve_default_workspace_root() -> Path:
    """Resolve the canonical default workspace root (`.`)."""

    return resolve_path(".")


def resolve_workspace_root(workspace_root: str | Path = ".") -> Path:
    """Resolve one workspace root path."""

    return resolve_path(workspace_root)


def resolve_default_store_root() -> Path:
    """Resolve canonical default store root (`.numereng`)."""

    return resolve_default_workspace_root() / CANONICAL_STORE_DIRNAME


def resolve_workspace_layout(workspace_root: str | Path = ".") -> WorkspaceLayout:
    """Resolve the canonical layout for one workspace root."""

    resolved_workspace_root = resolve_workspace_root(workspace_root)
    store_root = resolved_workspace_root / CANONICAL_STORE_DIRNAME
    return WorkspaceLayout(
        workspace_root=resolved_workspace_root,
        store_root=store_root,
        experiments_root=store_root / "experiments",
        notes_root=store_root / "notes",
        custom_models_root=_BUILTIN_CUSTOM_MODELS_ROOT,
        research_programs_root=_BUILTIN_RESEARCH_PROGRAMS_ROOT,
        skills_root=resolved_workspace_root / ".agents" / "skills",
    )


def resolve_workspace_layout_from_store_root(store_root: str | Path = CANONICAL_STORE_DIRNAME) -> WorkspaceLayout:
    """Resolve one workspace layout from the hidden runtime store root."""

    resolved_store_root = resolve_path(store_root)
    workspace_root = resolved_store_root.parent
    layout = resolve_workspace_layout(workspace_root)
    if resolved_store_root == layout.store_root:
        return layout
    return WorkspaceLayout(
        workspace_root=layout.workspace_root,
        store_root=resolved_store_root,
        experiments_root=layout.experiments_root,
        notes_root=layout.notes_root,
        custom_models_root=layout.custom_models_root,
        research_programs_root=layout.research_programs_root,
        skills_root=layout.skills_root,
    )


def is_within(path: Path, root: Path) -> bool:
    """Return whether `path` resolves under `root` (inclusive)."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def ensure_store_root_not_nested(
    *,
    candidate_store_root: str | Path,
    canonical_store_root: str | Path = ".numereng",
    error_code: str = "store_path_noncanonical",
) -> Path:
    """Ensure candidate store root is not nested under canonical `.numereng/*`."""
    candidate = resolve_path(candidate_store_root)
    canonical = resolve_path(canonical_store_root)

    if candidate == canonical:
        return candidate
    if is_within(candidate, canonical):
        raise ValueError(f"{error_code}:{candidate}")
    return candidate


def ensure_allowed_store_target(
    *,
    target_path: str | Path,
    store_root: str | Path,
    allowed_prefixes: tuple[str, ...],
    allow_store_root: bool,
    error_code: str = "store_path_noncanonical",
) -> Path:
    """
    Ensure an in-store target is restricted to allowed canonical prefixes.

    Targets outside `store_root` are permitted (user-managed external outputs).
    """
    target = resolve_path(target_path)
    root = resolve_path(store_root)

    try:
        relative = target.relative_to(root)
    except ValueError:
        return target

    if relative == Path("."):
        if allow_store_root:
            return target
        raise ValueError(f"{error_code}:{target}")

    for prefix in allowed_prefixes:
        prefix_path = Path(prefix)
        if relative == prefix_path or is_within(root / relative, root / prefix_path):
            return target

    raise ValueError(f"{error_code}:{target}")


def targeted_stray_paths(*, store_root: str | Path = ".numereng") -> tuple[Path, ...]:
    """Return the explicit stray directories targeted for cleanup."""
    root = resolve_path(store_root)
    return tuple(root / name for name in TARGETED_STRAY_DIRS)


def resolve_tmp_root(*, store_root: str | Path = ".numereng") -> Path:
    """Return the canonical tmp root under the store."""

    return resolve_path(store_root) / "tmp"


def resolve_tmp_remote_configs_root(*, store_root: str | Path = ".numereng") -> Path:
    """Return the retention-managed remote config staging root."""

    return resolve_tmp_root(store_root=store_root) / "remote-configs"


def resolve_cloud_cache_root(*, store_root: str | Path = ".numereng", provider: str) -> Path:
    """Return the canonical cache-backed cloud root for one provider."""

    return resolve_path(store_root) / "cache" / "cloud" / _safe_cloud_token(provider, field="provider")


def resolve_cloud_run_cache_root(
    *,
    store_root: str | Path = ".numereng",
    provider: str,
    run_id: str,
) -> Path:
    """Return the canonical cache-backed cloud root for one run-bound operation."""

    return (
        resolve_cloud_cache_root(store_root=store_root, provider=provider)
        / "runs"
        / _safe_cloud_token(
            run_id,
            field="run_id",
        )
    )


def resolve_cloud_run_state_path(
    *,
    store_root: str | Path = ".numereng",
    provider: str,
    run_id: str,
) -> Path:
    """Return the canonical cache-backed state file for one run-bound cloud flow."""

    return resolve_cloud_run_cache_root(store_root=store_root, provider=provider, run_id=run_id) / "state.json"


def resolve_cloud_run_pull_dir(
    *,
    store_root: str | Path = ".numereng",
    provider: str,
    run_id: str,
) -> Path:
    """Return the canonical cache-backed pull directory for one run-bound cloud flow."""

    return resolve_cloud_run_cache_root(store_root=store_root, provider=provider, run_id=run_id) / "pull"


def resolve_cloud_op_state_path(
    *,
    store_root: str | Path = ".numereng",
    provider: str,
    op_id: str,
) -> Path:
    """Return the canonical cache-backed state file for one non-run cloud operation."""

    return (
        resolve_cloud_cache_root(store_root=store_root, provider=provider)
        / "ops"
        / _safe_cloud_token(op_id, field="op_id")
        / "state.json"
    )


def resolve_legacy_cloud_state_path(*, store_root: str | Path = ".numereng", name: str) -> Path:
    """Return the legacy cloud state path under `<store_root>/cloud`."""

    token = _safe_cloud_token(name, field="name")
    filename = token if token.endswith(".json") else f"{token}.json"
    return resolve_path(store_root) / "cloud" / filename


def validate_cloud_state_path(
    *,
    target_path: str | Path,
    store_root: str | Path = ".numereng",
    error_code: str = "cloud_state_path_noncanonical",
    allow_legacy_cloud: bool = True,
) -> Path:
    """Validate an explicit cloud state override against canonical cache roots."""

    resolved = resolve_path(target_path)
    root = resolve_path(store_root)
    if resolved.suffix.lower() != ".json":
        raise ValueError(f"{error_code}:{resolved}")
    if is_within(resolved, root / "cache" / "cloud"):
        return resolved
    if allow_legacy_cloud and is_within(resolved, root / "cloud"):
        return resolved
    if _looks_like_canonical_cloud_state_path(resolved, allow_legacy_cloud=allow_legacy_cloud):
        return resolved
    raise ValueError(f"{error_code}:{resolved}")


def is_legacy_cloud_path(*, path: str | Path, store_root: str | Path = ".numereng") -> bool:
    """Return whether one path resolves under the deprecated `<store_root>/cloud` root."""

    return is_within(resolve_path(path), resolve_path(store_root) / "cloud")


def is_cloud_cache_path(*, path: str | Path, store_root: str | Path = ".numereng") -> bool:
    """Return whether one path resolves under the canonical `<store_root>/cache/cloud` root."""

    return is_within(resolve_path(path), resolve_path(store_root) / "cache" / "cloud")


def _safe_cloud_token(value: str, *, field: str) -> str:
    token = value.strip()
    if not token or token in {".", ".."}:
        raise ValueError(f"cloud_layout_{field}_invalid")
    if "/" in token or "\\" in token:
        raise ValueError(f"cloud_layout_{field}_invalid")
    return token


def _looks_like_canonical_cloud_state_path(path: Path, *, allow_legacy_cloud: bool) -> bool:
    parts = path.parts
    for index in range(len(parts) - 2):
        if parts[index] != CANONICAL_STORE_DIRNAME:
            continue
        if allow_legacy_cloud and parts[index + 1] == "cloud":
            return True
        if index + 2 < len(parts) and parts[index + 1] == "cache" and parts[index + 2] == "cloud":
            return True
    return False


__all__ = [
    "CANONICAL_STORE_DIRNAME",
    "CANONICAL_STORE_TOP_LEVEL_DIRS",
    "CANONICAL_STORE_TOP_LEVEL_FILES",
    "CANONICAL_WORKSPACE_TOP_LEVEL_DIRS",
    "TARGETED_STRAY_DIRS",
    "WorkspaceLayout",
    "ensure_allowed_store_target",
    "ensure_store_root_not_nested",
    "is_cloud_cache_path",
    "is_legacy_cloud_path",
    "is_within",
    "resolve_cloud_cache_root",
    "resolve_cloud_op_state_path",
    "resolve_cloud_run_cache_root",
    "resolve_cloud_run_pull_dir",
    "resolve_cloud_run_state_path",
    "resolve_default_workspace_root",
    "resolve_default_store_root",
    "resolve_legacy_cloud_state_path",
    "resolve_tmp_remote_configs_root",
    "resolve_tmp_root",
    "resolve_path",
    "resolve_workspace_layout",
    "resolve_workspace_layout_from_store_root",
    "resolve_workspace_root",
    "targeted_stray_paths",
    "validate_cloud_state_path",
]
