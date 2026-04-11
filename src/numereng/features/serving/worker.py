"""Subprocess worker for one serving component fit + live prediction build."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    args = parser.parse_args(argv)

    payload_path = Path(args.payload).expanduser().resolve()
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    workspace_root = str(payload["workspace_root"])
    component = ServingComponentSpec(
        component_id=str(payload["component"]["component_id"]),
        weight=float(payload["component"]["weight"]),
        config_path=Path(str(payload["component"]["config_path"])).resolve(),
        run_id=payload["component"].get("run_id"),
        source_label=payload["component"].get("source_label"),
    )
    config_path = Path(str(payload["config_path"])).expanduser().resolve()
    live_path = Path(str(payload["live_path"])).expanduser().resolve()
    output_path = Path(str(payload["output_path"])).expanduser().resolve()

    client = NumeraiClient(tournament="classic")
    plan = prepare_component_plan(
        workspace_root=workspace_root,
        component=component,
        config_path=config_path,
    )
    prepared = prepare_training_context(
        workspace_root=workspace_root,
        client=client,
        plan=plan,
    )
    fitted = fit_component(
        workspace_root=workspace_root,
        client=client,
        component=component,
        config_path=config_path,
        plan=plan,
        prepared_data=prepared,
    )
    live_features = pd.read_parquet(live_path)
    predictions = predict_component_live(component=fitted, live_features=live_features)
    write_parquet(predictions, output_path, index=False)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
