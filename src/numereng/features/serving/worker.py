"""Subprocess worker for one serving component fit + live prediction build."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from numereng.features.serving.contracts import ServingComponentSpec
from numereng.features.serving.runtime import (
    fit_component,
    predict_component_live,
    prepare_component_plan,
    prepare_training_context,
)
from numereng.platform.numerai_client import NumeraiClient
from numereng.platform.parquet import write_parquet


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_phase(
    *,
    status_path: Path | None,
    component_id: str,
    phase: str,
    state: str = "running",
    details: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "component_id": component_id,
        "phase": phase,
        "state": state,
        "updated_at": _utc_now_iso(),
    }
    if details:
        payload["details"] = details
    print(json.dumps({"event": "serving_component_worker_phase", **payload}, sort_keys=True), flush=True)
    if status_path is None:
        return
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    args = parser.parse_args(argv)

    payload_path = Path(args.payload).expanduser().resolve()
    payload = json.loads(payload_path.read_text(encoding="utf-8-sig"))
    workspace_root = str(payload["workspace_root"])
    component_id = str(payload["component"]["component_id"])
    raw_status_path = payload.get("status_path")
    status_path = None if raw_status_path is None else Path(str(raw_status_path)).expanduser().resolve()
    component = ServingComponentSpec(
        component_id=component_id,
        weight=float(payload["component"]["weight"]),
        config_path=Path(str(payload["component"]["config_path"])).resolve(),
        run_id=payload["component"].get("run_id"),
        source_label=payload["component"].get("source_label"),
    )
    config_path = Path(str(payload["config_path"])).expanduser().resolve()
    live_path = Path(str(payload["live_path"])).expanduser().resolve()
    output_path = Path(str(payload["output_path"])).expanduser().resolve()

    try:
        _write_phase(
            status_path=status_path,
            component_id=component_id,
            phase="prepare_component_plan",
            details={"config_path": str(config_path)},
        )
        client = NumeraiClient(tournament="classic")
        plan = prepare_component_plan(
            workspace_root=workspace_root,
            component=component,
            config_path=config_path,
        )
        _write_phase(
            status_path=status_path,
            component_id=component_id,
            phase="prepare_training_context",
            details={
                "data_version": plan.data_key.data_version,
                "feature_set": plan.data_key.feature_set,
                "dataset_scope": plan.data_key.dataset_scope,
                "target_col": plan.data_key.target_col,
            },
        )
        prepared = prepare_training_context(
            workspace_root=workspace_root,
            client=client,
            plan=plan,
        )
        _write_phase(status_path=status_path, component_id=component_id, phase="fit_component")
        fitted = fit_component(
            workspace_root=workspace_root,
            client=client,
            component=component,
            config_path=config_path,
            plan=plan,
            prepared_data=prepared,
        )
        _write_phase(
            status_path=status_path,
            component_id=component_id,
            phase="read_live_features",
            details={"live_path": str(live_path)},
        )
        live_features = pd.read_parquet(live_path)
        _write_phase(
            status_path=status_path,
            component_id=component_id,
            phase="predict_live",
            details={"live_rows": int(len(live_features))},
        )
        predictions = predict_component_live(component=fitted, live_features=live_features)
        _write_phase(
            status_path=status_path,
            component_id=component_id,
            phase="write_predictions",
            details={"output_path": str(output_path), "rows": int(len(predictions))},
        )
        write_parquet(predictions, output_path, index=False)
    except Exception as exc:
        _write_phase(
            status_path=status_path,
            component_id=component_id,
            phase="failed",
            state="failed",
            details={"error": f"{type(exc).__name__}: {exc}"},
        )
        raise
    _write_phase(
        status_path=status_path,
        component_id=component_id,
        phase="complete",
        state="completed",
        details={"output_path": str(output_path)},
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
