"""Live benchmark-model alignment shared by the local and Numerai-hosted predict paths.

USAGE:
    from numereng.features.serving.live_benchmark import attach_live_benchmark

    live = attach_live_benchmark(
        live,
        benchmark=live_benchmark_models,
        id_col="id",
        baseline_col="bench_raw",
        benchmark_model_col="v53_lgbm_ender20",
    )

This module must stay free of `numereng` imports. `features/serving/hosted.py` carries
it into the uploaded pickle with `cloudpickle.register_pickle_by_value`, so Numerai's
hosted runtime (where `numereng` is not installed) executes this exact code and the
local live path cannot drift away from it.
"""

from __future__ import annotations

import pandas as pd

# --------------------------------------------------------------------------- #
# Live benchmark attach
# --------------------------------------------------------------------------- #


def attach_live_benchmark(
    frame: pd.DataFrame,
    *,
    benchmark: pd.DataFrame | None,
    id_col: str,
    baseline_col: str,
    benchmark_model_col: str,
) -> pd.DataFrame:
    """Attach one live benchmark-model column to a live frame under `baseline_col`.

    Raises on every ambiguity instead of substituting another benchmark: a silent
    wrong-benchmark join would be an invisible correctness bug in a live path.
    """
    if benchmark is None:
        raise ValueError(f"serving_live_benchmark_models_missing:{benchmark_model_col}")
    if benchmark_model_col not in benchmark.columns:
        available = ",".join(str(col) for col in list(benchmark.columns)[:8])
        raise ValueError(f"serving_live_benchmark_column_missing:{benchmark_model_col}:available={available}")
    if id_col not in frame.columns:
        raise ValueError(f"serving_live_missing_id_col:{id_col}")

    if id_col in benchmark.columns:
        keys = benchmark[id_col].astype(str).to_numpy()
    elif benchmark.index.name == id_col:
        keys = benchmark.index.astype(str).to_numpy()
    else:
        raise ValueError(f"serving_live_benchmark_missing_id_col:{id_col}")

    values = pd.to_numeric(benchmark[benchmark_model_col], errors="coerce").to_numpy(dtype="float64")
    lookup = pd.Series(values, index=pd.Index(keys, dtype=object))
    if lookup.index.has_duplicates:
        raise ValueError(f"serving_live_benchmark_duplicate_ids:{benchmark_model_col}")

    aligned = lookup.reindex(pd.Index(frame[id_col].astype(str).to_numpy(), dtype=object))
    missing = int(aligned.isna().sum())
    if missing:
        raise ValueError(f"serving_live_benchmark_values_missing:{benchmark_model_col}:{missing}")

    attached = frame.copy()
    attached[baseline_col] = aligned.to_numpy(dtype="float64")
    return attached


__all__ = ["attach_live_benchmark"]
