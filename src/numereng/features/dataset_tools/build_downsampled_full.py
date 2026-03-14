"""Build downsampled full datasets from canonical split sources."""

from __future__ import annotations

import argparse
from pathlib import Path

from numereng.features.dataset_tools import BuildDownsampledFullRequest, build_downsampled_full
from numereng.features.training.client import create_training_data_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build downsampled_full.parquet and downsampled_full_benchmark_models.parquet."
    )
    parser.add_argument(
        "--data-version",
        type=str,
        default="v5.2",
        help="Numerai data version (default: v5.2).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(".numereng/datasets"),
        help="Dataset root directory (default: .numereng/datasets).",
    )
    parser.add_argument(
        "--downsample-eras-step",
        type=int,
        default=4,
        help="Keep every Nth era when building downsampled_full (default: 4).",
    )
    parser.add_argument(
        "--downsample-eras-offset",
        type=int,
        default=0,
        help="Offset when selecting every Nth era (default: 0).",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild downsampled datasets even if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_version = args.data_version
    data_dir = Path(args.data_dir).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    client = create_training_data_client()
    result = build_downsampled_full(
        BuildDownsampledFullRequest(
            data_dir=data_dir,
            data_version=data_version,
            rebuild=args.rebuild,
            downsample_eras_step=args.downsample_eras_step,
            downsample_eras_offset=args.downsample_eras_offset,
        ),
        client=client,
    )
    print(f"Built {result.downsampled_full_path}")
    print(f"Built {result.downsampled_full_benchmark_path}")


if __name__ == "__main__":
    main()
