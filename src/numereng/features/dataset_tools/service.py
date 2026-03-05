"""Dataset-tools downsampling business logic."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from numereng.features.dataset_tools.contracts import (
    BuildDownsampledFullRequest,
    BuildDownsampledFullResult,
)
from numereng.features.training.client import TrainingDataClient

_DEFAULT_DOWNSAMPLE_STEP = 4
_DEFAULT_DOWNSAMPLE_OFFSET = 0
_ERA_COL = "era"
_ID_COL = "id"
_PARQUET_BATCH_SIZE = 10_000


class DatasetToolsError(Exception):
    """Base error for dataset-tools workflows."""


class DatasetToolsValidationError(DatasetToolsError):
    """Raised when dataset-tools inputs are invalid."""


class DatasetToolsExecutionError(DatasetToolsError):
    """Raised when dataset-tools processing fails."""


def _ensure_dataset_file(
    client: TrainingDataClient,
    *,
    data_version: str,
    version_dir: Path,
    filename: str,
) -> Path:
    path = (version_dir / filename).resolve()
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        client.download_dataset(
            filename=f"{data_version}/{filename}",
            dest_path=str(path),
        )
    except Exception as exc:
        raise DatasetToolsExecutionError(f"dataset_download_failed:{data_version}/{filename}") from exc
    return path


def _safe_read_parquet(path: Path, *, columns: list[str] | None = None) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception as exc:
        raise DatasetToolsExecutionError(f"parquet_read_failed:{path}") from exc


def _safe_write_parquet(frame: pd.DataFrame, path: Path, *, index: bool) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=index)
    except Exception as exc:
        raise DatasetToolsExecutionError(f"parquet_write_failed:{path}") from exc


def _parquet_num_rows(path: Path) -> int:
    try:
        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception as exc:
        raise DatasetToolsExecutionError(f"parquet_read_failed:{path}") from exc


def _iter_parquet_batches(path: Path, *, batch_size: int = _PARQUET_BATCH_SIZE) -> Iterator[pd.DataFrame]:
    try:
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=batch_size, use_pandas_metadata=True):
            yield batch.to_pandas()
    except Exception as exc:
        raise DatasetToolsExecutionError(f"parquet_read_failed:{path}") from exc


def _open_writer(path: Path, table: pa.Table) -> pq.ParquetWriter:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        return pq.ParquetWriter(str(path), table.schema)
    except Exception as exc:
        raise DatasetToolsExecutionError(f"parquet_write_failed:{path}") from exc


def _write_full_dataset_streaming(
    *,
    train_path: Path,
    validation_path: Path,
    full_path: Path,
) -> int:
    writer: pq.ParquetWriter | None = None
    rows_written = 0

    try:
        for source_path, only_validation_rows in ((train_path, False), (validation_path, True)):
            for frame in _iter_parquet_batches(source_path):
                batch = frame
                if only_validation_rows and "data_type" in batch.columns:
                    batch = batch[batch["data_type"] == "validation"].copy()
                batch = batch.drop(columns=["data_type"], errors="ignore")
                if batch.index.name and batch.index.name not in batch.columns:
                    batch = batch.reset_index()
                if batch.empty:
                    continue
                table = pa.Table.from_pandas(batch, preserve_index=False)
                if writer is None:
                    writer = _open_writer(full_path, table)
                writer.write_table(table)
                rows_written += int(len(batch))
    except Exception as exc:
        if writer is not None:
            writer.close()
        if full_path.exists():
            full_path.unlink(missing_ok=True)
        raise DatasetToolsExecutionError(f"parquet_write_failed:{full_path}") from exc

    if writer is None:
        raise DatasetToolsExecutionError(f"parquet_write_failed:{full_path}")
    writer.close()
    return rows_written


def _iter_validation_benchmark_batches(
    *,
    validation_data_path: Path,
    validation_benchmark_path: Path,
) -> Iterator[pd.DataFrame]:
    validation_meta_batches = iter(_iter_parquet_batches(validation_data_path))
    validation_benchmark_batches = iter(_iter_parquet_batches(validation_benchmark_path))

    while True:
        try:
            meta_batch = next(validation_meta_batches)
        except StopIteration:
            break
        try:
            benchmark_batch = next(validation_benchmark_batches)
        except StopIteration as exc:
            raise DatasetToolsExecutionError(
                f"benchmark_validation_alignment_failed:{validation_data_path}:{validation_benchmark_path}"
            ) from exc

        if len(meta_batch) != len(benchmark_batch):
            raise DatasetToolsExecutionError(
                f"benchmark_validation_alignment_failed:{validation_data_path}:{validation_benchmark_path}"
            )

        if "data_type" not in meta_batch.columns:
            yield benchmark_batch
            continue

        mask = meta_batch["data_type"] == "validation"
        if bool(mask.any()):
            yield benchmark_batch.loc[mask.to_numpy()]

    try:
        next(validation_benchmark_batches)
    except StopIteration:
        return
    raise DatasetToolsExecutionError(
        f"benchmark_validation_alignment_failed:{validation_data_path}:{validation_benchmark_path}"
    )


def _write_full_benchmark_streaming(
    *,
    train_benchmark_path: Path,
    validation_benchmark_path: Path,
    validation_data_path: Path,
    full_path: Path,
) -> int:
    writer: pq.ParquetWriter | None = None
    rows_written = 0

    try:
        for frame in _iter_parquet_batches(train_benchmark_path):
            if frame.empty:
                continue
            table = pa.Table.from_pandas(frame, preserve_index=True)
            if writer is None:
                writer = _open_writer(full_path, table)
            writer.write_table(table)
            rows_written += int(len(frame))

        for frame in _iter_validation_benchmark_batches(
            validation_data_path=validation_data_path,
            validation_benchmark_path=validation_benchmark_path,
        ):
            if frame.empty:
                continue
            table = pa.Table.from_pandas(frame, preserve_index=True)
            if writer is None:
                writer = _open_writer(full_path, table)
            writer.write_table(table)
            rows_written += int(len(frame))
    except Exception as exc:
        if writer is not None:
            writer.close()
        if full_path.exists():
            full_path.unlink(missing_ok=True)
        raise DatasetToolsExecutionError(f"parquet_write_failed:{full_path}") from exc

    if writer is None:
        raise DatasetToolsExecutionError(f"parquet_write_failed:{full_path}")
    writer.close()
    return rows_written


def _era_sort_key(value: object) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError) as exc:
        raise DatasetToolsValidationError(f"downsample_era_not_numeric:{value}") from exc


def _build_full_dataset(
    client: TrainingDataClient,
    *,
    data_version: str,
    version_dir: Path,
    reuse_existing: bool,
) -> tuple[Path, int]:
    full_path = (version_dir / "full.parquet").resolve()
    if reuse_existing and full_path.exists():
        return full_path, _parquet_num_rows(full_path)

    train_path = _ensure_dataset_file(
        client,
        data_version=data_version,
        version_dir=version_dir,
        filename="train.parquet",
    )
    validation_path = _ensure_dataset_file(
        client,
        data_version=data_version,
        version_dir=version_dir,
        filename="validation.parquet",
    )

    rows = _write_full_dataset_streaming(
        train_path=train_path,
        validation_path=validation_path,
        full_path=full_path,
    )
    return full_path, rows


def _build_full_benchmark(
    client: TrainingDataClient,
    *,
    data_version: str,
    version_dir: Path,
    reuse_existing: bool,
) -> tuple[Path, int]:
    full_path = (version_dir / "full_benchmark_models.parquet").resolve()
    if reuse_existing and full_path.exists():
        return full_path, _parquet_num_rows(full_path)

    train_path = _ensure_dataset_file(
        client,
        data_version=data_version,
        version_dir=version_dir,
        filename="train_benchmark_models.parquet",
    )
    validation_path = _ensure_dataset_file(
        client,
        data_version=data_version,
        version_dir=version_dir,
        filename="validation_benchmark_models.parquet",
    )
    validation_data_path = _ensure_dataset_file(
        client,
        data_version=data_version,
        version_dir=version_dir,
        filename="validation.parquet",
    )

    rows = _write_full_benchmark_streaming(
        train_benchmark_path=train_path,
        validation_benchmark_path=validation_path,
        validation_data_path=validation_data_path,
        full_path=full_path,
    )
    return full_path, rows


def _build_downsampled_full_dataset(
    *,
    full_path: Path,
    version_dir: Path,
    era_step: int,
    era_offset: int,
) -> tuple[Path, int, int, int]:
    if era_step < 2:
        raise DatasetToolsValidationError(f"downsample_step_invalid:{era_step}")
    if era_offset < 0 or era_offset >= era_step:
        raise DatasetToolsValidationError(
            f"downsample_offset_invalid:{era_offset}:step={era_step}"
        )

    downsampled_path = (version_dir / "downsampled_full.parquet").resolve()
    full = _safe_read_parquet(full_path)
    if _ERA_COL not in full.columns:
        raise DatasetToolsValidationError(f"downsample_missing_era_column:{full_path}:{_ERA_COL}")

    unique_eras = sorted(full[_ERA_COL].unique(), key=_era_sort_key)
    keep_eras = {era for idx, era in enumerate(unique_eras) if idx % era_step == era_offset}
    downsampled = full[full[_ERA_COL].isin(keep_eras)].copy()
    _safe_write_parquet(downsampled, downsampled_path, index=False)
    return downsampled_path, int(len(downsampled)), int(len(unique_eras)), int(len(keep_eras))


def _build_downsampled_full_benchmark(
    *,
    full_benchmark_path: Path,
    downsampled_full_path: Path,
    version_dir: Path,
) -> tuple[Path, int]:
    downsampled_path = (version_dir / "downsampled_full_benchmark_models.parquet").resolve()
    ids = _safe_read_parquet(downsampled_full_path, columns=[_ID_COL])
    if _ID_COL not in ids.columns:
        raise DatasetToolsValidationError(f"downsample_missing_id_column:{downsampled_full_path}:{_ID_COL}")
    id_values = pd.Index(ids[_ID_COL].dropna().unique())
    benchmark = _safe_read_parquet(full_benchmark_path)
    if _ID_COL in benchmark.columns:
        benchmark = benchmark.set_index(_ID_COL)
    benchmark = benchmark.loc[benchmark.index.intersection(id_values)]
    _safe_write_parquet(benchmark, downsampled_path, index=True)
    return downsampled_path, int(len(benchmark))


def build_downsampled_full(
    request: BuildDownsampledFullRequest,
    *,
    client: TrainingDataClient,
) -> BuildDownsampledFullResult:
    """Build full and downsampled full datasets using official era filtering."""

    data_dir = Path(request.data_dir).expanduser().resolve()
    version_dir = (data_dir / request.data_version).resolve()
    version_dir.mkdir(parents=True, exist_ok=True)
    reuse_existing = not request.rebuild

    full_path, full_rows = _build_full_dataset(
        client,
        data_version=request.data_version,
        version_dir=version_dir,
        reuse_existing=reuse_existing,
    )
    full_benchmark_path, full_benchmark_rows = _build_full_benchmark(
        client,
        data_version=request.data_version,
        version_dir=version_dir,
        reuse_existing=reuse_existing,
    )
    downsampled_full_path: Path | None = None
    downsampled_rows: int | None = None
    total_eras: int | None = None
    kept_eras: int | None = None
    downsampled_full_benchmark_path: Path | None = None
    downsampled_full_benchmark_rows: int | None = None
    downsample_built = False

    if not request.skip_downsample:
        downsampled_full_path, downsampled_rows, total_eras, kept_eras = _build_downsampled_full_dataset(
            full_path=full_path,
            version_dir=version_dir,
            era_step=request.downsample_eras_step,
            era_offset=request.downsample_eras_offset,
        )
        downsampled_full_benchmark_path, downsampled_full_benchmark_rows = _build_downsampled_full_benchmark(
            full_benchmark_path=full_benchmark_path,
            downsampled_full_path=downsampled_full_path,
            version_dir=version_dir,
        )
        downsample_built = True

    return BuildDownsampledFullResult(
        data_dir=data_dir,
        data_version=request.data_version,
        full_path=full_path,
        full_benchmark_path=full_benchmark_path,
        downsampled_full_path=downsampled_full_path,
        downsampled_full_benchmark_path=downsampled_full_benchmark_path,
        full_rows=full_rows,
        downsampled_rows=downsampled_rows,
        full_benchmark_rows=full_benchmark_rows,
        downsampled_full_benchmark_rows=downsampled_full_benchmark_rows,
        total_eras=total_eras,
        kept_eras=kept_eras,
        downsample_step=request.downsample_eras_step,
        downsample_offset=request.downsample_eras_offset,
        downsample_built=downsample_built,
    )


def build_downsampled_full_response_payload(result: BuildDownsampledFullResult) -> dict[str, Any]:
    """Return one stable response payload for CLI/API surfaces."""

    return {
        "data_dir": str(result.data_dir),
        "data_version": result.data_version,
        "full_path": str(result.full_path),
        "full_benchmark_path": str(result.full_benchmark_path),
        "downsampled_full_path": str(result.downsampled_full_path) if result.downsampled_full_path else None,
        "downsampled_full_benchmark_path": (
            str(result.downsampled_full_benchmark_path) if result.downsampled_full_benchmark_path else None
        ),
        "full_rows": result.full_rows,
        "downsampled_rows": result.downsampled_rows,
        "full_benchmark_rows": result.full_benchmark_rows,
        "downsampled_full_benchmark_rows": result.downsampled_full_benchmark_rows,
        "total_eras": result.total_eras,
        "kept_eras": result.kept_eras,
        "downsample_step": result.downsample_step,
        "downsample_offset": result.downsample_offset,
        "downsample_built": result.downsample_built,
    }


__all__ = [
    "DatasetToolsError",
    "DatasetToolsExecutionError",
    "DatasetToolsValidationError",
    "build_downsampled_full",
    "build_downsampled_full_response_payload",
]
