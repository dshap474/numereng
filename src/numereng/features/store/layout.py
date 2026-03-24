"""Canonical store layout constants and path-guard helpers."""

from __future__ import annotations

from pathlib import Path

CANONICAL_STORE_TOP_LEVEL_DIRS: tuple[str, ...] = (
    "runs",
    "datasets",
    "cloud",
    "experiments",
    "notes",
    "cache",
)
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


def resolve_path(value: str | Path) -> Path:
    """Resolve one filesystem path with user-home expansion."""
    return Path(value).expanduser().resolve()


def resolve_default_store_root() -> Path:
    """Resolve canonical default store root (`.numereng`)."""
    return resolve_path(".numereng")


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


__all__ = [
    "CANONICAL_STORE_TOP_LEVEL_DIRS",
    "CANONICAL_STORE_TOP_LEVEL_FILES",
    "TARGETED_STRAY_DIRS",
    "ensure_allowed_store_target",
    "ensure_store_root_not_nested",
    "is_within",
    "resolve_default_store_root",
    "resolve_path",
    "targeted_stray_paths",
]
