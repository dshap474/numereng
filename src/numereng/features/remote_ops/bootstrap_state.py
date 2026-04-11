"""Lightweight persisted viz bootstrap state helpers."""

from __future__ import annotations

import json
from pathlib import Path

from numereng.features.remote_ops.contracts import (
    RemoteBootstrapStatus,
    RemoteTargetRecord,
    RemoteVizBootstrapResult,
    RemoteVizBootstrapTargetResult,
)
from numereng.features.store import resolve_store_root

_REMOTE_BOOTSTRAP_DIR = ("remote_ops", "bootstrap")
_REMOTE_VIZ_BOOTSTRAP_FILE = "viz.json"


def load_viz_bootstrap_state(
    *,
    store_root: str | Path = ".numereng",
) -> RemoteVizBootstrapResult | None:
    """Load the last persisted viz bootstrap result for enabled remote sources."""

    resolved_store_root = resolve_store_root(store_root)
    state_path = remote_viz_bootstrap_state_path(store_root=resolved_store_root)
    if not state_path.is_file():
        return None
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    targets_payload = payload.get("targets")
    if not isinstance(targets_payload, list):
        targets_payload = []
    targets: list[RemoteVizBootstrapTargetResult] = []
    for item in targets_payload:
        if not isinstance(item, dict):
            continue
        target_payload = item.get("target")
        if not isinstance(target_payload, dict):
            continue
        try:
            targets.append(
                RemoteVizBootstrapTargetResult(
                    target=RemoteTargetRecord(
                        id=str(target_payload["id"]),
                        label=str(target_payload["label"]),
                        kind=str(target_payload["kind"]),
                        shell=str(target_payload["shell"]),
                        repo_root=str(target_payload["repo_root"]),
                        store_root=str(target_payload["store_root"]),
                        runner_cmd=str(target_payload["runner_cmd"]),
                        python_cmd=str(target_payload["python_cmd"]),
                        tags=tuple(str(tag) for tag in target_payload.get("tags", [])),
                    ),
                    bootstrap_status=_bootstrap_status_value(item.get("bootstrap_status")),
                    last_bootstrap_at=str(item.get("last_bootstrap_at") or payload.get("bootstrapped_at") or ""),
                    last_bootstrap_error=_json_optional_str(item.get("last_bootstrap_error")),
                    repo_synced=bool(item.get("repo_synced")),
                    repo_sync_skipped=bool(item.get("repo_sync_skipped")),
                    doctor_ok=bool(item.get("doctor_ok")),
                    issues=tuple(str(issue) for issue in item.get("issues", []) if isinstance(issue, str)),
                )
            )
        except KeyError:
            continue
    return RemoteVizBootstrapResult(
        store_root=resolved_store_root,
        state_path=state_path,
        bootstrapped_at=str(payload.get("bootstrapped_at") or ""),
        ready_count=int(payload.get("ready_count") or 0),
        degraded_count=int(payload.get("degraded_count") or 0),
        targets=tuple(targets),
    )


def remote_viz_bootstrap_state_path(*, store_root: str | Path) -> Path:
    """Return the local viz bootstrap state path under the numereng store root."""

    resolved_store_root = resolve_store_root(store_root)
    return resolved_store_root / _REMOTE_BOOTSTRAP_DIR[0] / _REMOTE_BOOTSTRAP_DIR[1] / _REMOTE_VIZ_BOOTSTRAP_FILE


def write_viz_bootstrap_state(result: RemoteVizBootstrapResult) -> None:
    """Persist one viz bootstrap state payload to the numereng store."""

    result.state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "store_root": str(result.store_root),
        "state_path": str(result.state_path),
        "bootstrapped_at": result.bootstrapped_at,
        "ready_count": result.ready_count,
        "degraded_count": result.degraded_count,
        "targets": [
            {
                "target": {
                    "id": item.target.id,
                    "label": item.target.label,
                    "kind": item.target.kind,
                    "shell": item.target.shell,
                    "repo_root": item.target.repo_root,
                    "store_root": item.target.store_root,
                    "runner_cmd": item.target.runner_cmd,
                    "python_cmd": item.target.python_cmd,
                    "tags": list(item.target.tags),
                },
                "bootstrap_status": item.bootstrap_status,
                "last_bootstrap_at": item.last_bootstrap_at,
                "last_bootstrap_error": item.last_bootstrap_error,
                "repo_synced": item.repo_synced,
                "repo_sync_skipped": item.repo_sync_skipped,
                "doctor_ok": item.doctor_ok,
                "issues": list(item.issues),
            }
            for item in result.targets
        ],
    }
    result.state_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _bootstrap_status_value(value: object) -> RemoteBootstrapStatus:
    if value == "ready":
        return "ready"
    return "degraded"


def _json_optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = [
    "load_viz_bootstrap_state",
    "remote_viz_bootstrap_state_path",
    "write_viz_bootstrap_state",
]
