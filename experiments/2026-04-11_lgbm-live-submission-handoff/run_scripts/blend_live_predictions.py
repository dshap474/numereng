#!/usr/bin/env python3
"""Blend prepared live component prediction files into one submission parquet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    experiment_dir = script_dir.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=str(experiment_dir / "submission_handoff.json"),
        help="Path to the frozen submission handoff manifest.",
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="CANDIDATE_ID=PATH",
        help="One live component predictions parquet per candidate_id.",
    )
    parser.add_argument(
        "--output",
        default=str(experiment_dir / "predictions" / "live_weighted_blend.parquet"),
        help="Destination parquet path for the blended live submission predictions.",
    )
    return parser.parse_args()


def _parse_inputs(items: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"invalid --input entry: {item}")
        candidate_id, raw_path = item.split("=", 1)
        parsed[candidate_id.strip()] = Path(raw_path).expanduser().resolve()
    return parsed


def _rank_predictions(frame: pd.DataFrame, *, use_era: bool) -> pd.Series:
    if use_era:
        return frame.groupby("era")["prediction"].rank(method="average", pct=True)
    return frame["prediction"].rank(method="average", pct=True)


def main() -> int:
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    input_map = _parse_inputs(args.input)
    components = manifest["components"]
    expected_ids = [str(component["candidate_id"]) for component in components]
    missing = [candidate_id for candidate_id in expected_ids if candidate_id not in input_map]
    if missing:
        raise SystemExit(f"missing component inputs: {missing}")

    merged: pd.DataFrame | None = None
    use_era: bool | None = None
    for component in components:
        candidate_id = str(component["candidate_id"])
        frame = pd.read_parquet(input_map[candidate_id]).copy()
        has_era = "era" in frame.columns
        if use_era is None:
            use_era = has_era
        elif use_era != has_era:
            raise SystemExit("all component live prediction files must either all include era or all omit era")

        required = {"id", "prediction"}
        missing_cols = required.difference(frame.columns)
        if missing_cols:
            raise SystemExit(f"{candidate_id} missing required columns: {sorted(missing_cols)}")

        keep_cols = ["id", "prediction"]
        merge_cols = ["id"]
        if use_era:
            keep_cols = ["id", "era", "prediction"]
            merge_cols = ["id", "era"]
            frame["era"] = frame["era"].astype(str)
        frame["id"] = frame["id"].astype(str)
        ranked = frame[keep_cols].copy()
        ranked["prediction"] = _rank_predictions(ranked, use_era=bool(use_era))
        ranked = ranked.rename(columns={"prediction": candidate_id})
        if merged is None:
            merged = ranked
        else:
            merged = merged.merge(ranked, on=merge_cols, how="inner")

    if merged is None or merged.empty:
        raise SystemExit("aligned live component frame is empty")

    blended = pd.Series(0.0, index=merged.index, dtype="float64")
    for component in components:
        candidate_id = str(component["candidate_id"])
        blended = blended + (float(component["weight"]) * merged[candidate_id].astype("float64"))

    output = merged[["id"]].copy()
    output["prediction"] = blended
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)
    print(json.dumps({"output_path": str(output_path), "rows": int(len(output)), "components": expected_ids}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
