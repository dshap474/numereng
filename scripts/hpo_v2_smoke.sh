#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
WORKSPACE_DIR="${1:-$(mktemp -d "${TMPDIR:-/tmp}/numereng-hpo-v2-smoke.XXXXXX")}"
SOURCE_DATASETS_DIR="${NUMERENG_HPO_SMOKE_DATASETS_DIR:-}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "missing project python: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ -e "${WORKSPACE_DIR}" ]] && [[ -n "$(find "${WORKSPACE_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null || true)" ]]; then
  echo "workspace must be empty or absent: ${WORKSPACE_DIR}" >&2
  exit 1
fi

mkdir -p "${WORKSPACE_DIR}/.numereng"

if [[ -n "${SOURCE_DATASETS_DIR}" ]]; then
  if [[ ! -d "${SOURCE_DATASETS_DIR}/v5.2" ]]; then
    echo "missing source datasets under ${SOURCE_DATASETS_DIR}/v5.2" >&2
    exit 1
  fi
  cp -R "${SOURCE_DATASETS_DIR}" "${WORKSPACE_DIR}/.numereng/"
fi

export ROOT_DIR WORKSPACE_DIR

"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

workdir = Path(os.environ["WORKSPACE_DIR"])
datasets_root = workdir / ".numereng" / "datasets" / "v5.2"

if not datasets_root.exists():
    datasets_root.mkdir(parents=True, exist_ok=True)
    feature_names = [f"feature_{index}" for index in range(6)]
    rng = np.random.default_rng(13)

    def build_frame(*, era_start: int, era_count: int, rows_per_era: int, include_data_type: bool) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for era_offset in range(era_count):
            era_value = era_start + era_offset
            era_label = f"{era_value:04d}"
            for row_index in range(rows_per_era):
                features = rng.random(len(feature_names))
                target = float((features[0] * 0.35) + (features[1] * 0.25) + (features[2] * 0.2))
                payload: dict[str, object] = {
                    "id": f"{era_label}_{row_index:05d}",
                    "era": era_label,
                    "target": min(max(target, 0.0), 1.0),
                }
                if include_data_type:
                    payload["data_type"] = "validation"
                for feature_name, feature_value in zip(feature_names, features, strict=True):
                    payload[feature_name] = float(feature_value)
                rows.append(payload)
        return pd.DataFrame(rows)

    train = build_frame(era_start=1, era_count=5, rows_per_era=60, include_data_type=False)
    validation = build_frame(era_start=6, era_count=3, rows_per_era=60, include_data_type=True)
    combined = pd.concat(
        [
            train[["id", "era", "target"]].copy(),
            validation[["id", "era", "target"]].copy(),
        ],
        ignore_index=True,
    )
    benchmark = combined[["id", "era"]].copy()
    benchmark["v52_lgbm_ender20"] = rng.random(len(benchmark))
    meta_model = combined[["id", "era"]].copy()
    meta_model["numerai_meta_model"] = rng.random(len(meta_model))

    train.to_parquet(datasets_root / "train.parquet", index=False)
    validation.to_parquet(datasets_root / "validation.parquet", index=False)
    benchmark.to_parquet(datasets_root / "benchmark.parquet", index=False)
    meta_model.to_parquet(datasets_root / "meta_model.parquet", index=False)
    (datasets_root / "features.json").write_text(
        json.dumps(
            {
                "feature_sets": {
                    "small": feature_names,
                    "all": feature_names,
                    "fncv3_features": feature_names,
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )

base_config = {
    "data": {
        "data_version": "v5.2",
        "dataset_variant": "non_downsampled",
        "feature_set": "small",
        "target_col": "target",
        "scoring_targets": ["target"],
        "era_col": "era",
        "id_col": "id",
        "benchmark_source": {
            "source": "path",
            "predictions_path": "v5.2/benchmark.parquet",
            "pred_col": "v52_lgbm_ender20",
        },
        "meta_model_data_path": "v5.2/meta_model.parquet",
    },
    "model": {
        "type": "LGBMRegressor",
        "params": {
            "n_estimators": 25,
            "learning_rate": 0.05,
            "num_leaves": 15,
            "random_state": 13,
            "feature_fraction_seed": 13,
            "bagging_seed": 13,
            "data_random_seed": 13,
            "deterministic": True,
            "force_col_wise": True,
            "num_threads": 1,
            "verbose": -1,
        },
        "x_groups": ["features"],
    },
    "training": {
        "engine": {"profile": "simple"},
        "post_training_scoring": "core",
        "resources": {
            "parallel_folds": 1,
            "parallel_backend": "joblib",
            "memmap_enabled": False,
            "max_threads_per_worker": 1,
        },
        "cache": {
            "mode": "deterministic",
            "cache_fold_specs": True,
            "cache_features": True,
            "cache_labels": True,
            "cache_fold_matrices": False,
        },
    },
    "output": {"output_dir": str(workdir / ".numereng")},
}

common_random = {
    "study_id": "smoke-random-resume",
    "study_name": "smoke-random-resume",
    "config_path": str(workdir / "base.json"),
    "experiment_id": "exp-smoke",
    "objective": {
        "metric": "metrics.corr.mean",
        "direction": "maximize",
        "neutralization": {
            "enabled": False,
            "neutralizer_path": None,
            "proportion": 0.5,
            "mode": "era",
            "neutralizer_cols": None,
            "rank_output": True,
        },
    },
    "search_space": {
        "model.params.learning_rate": {
            "type": "float",
            "low": 0.02,
            "high": 0.08,
            "log": False,
        },
        "model.params.num_leaves": {
            "type": "int",
            "low": 8,
            "high": 24,
            "step": 4,
            "log": False,
        },
    },
    "sampler": {"kind": "random", "seed": 17},
}

duplicate_common = {
    "study_name": "smoke-duplicate",
    "config_path": str(workdir / "base.json"),
    "experiment_id": "exp-smoke",
    "objective": {
        "metric": "metrics.corr.mean",
        "direction": "maximize",
        "neutralization": {
            "enabled": False,
            "neutralizer_path": None,
            "proportion": 0.5,
            "mode": "era",
            "neutralizer_cols": None,
            "rank_output": True,
        },
    },
    "search_space": {
        "model.params.learning_rate": {
            "type": "categorical",
            "choices": [0.05],
        },
        "model.params.num_leaves": {
            "type": "categorical",
            "choices": [15],
        },
    },
    "sampler": {"kind": "random", "seed": 5},
}

(workdir / "base.json").write_text(json.dumps(base_config, indent=2), encoding="utf-8")

resume1 = dict(common_random)
resume1["stopping"] = {
    "max_trials": 1,
    "max_completed_trials": None,
    "timeout_seconds": None,
    "plateau": {
        "enabled": False,
        "min_completed_trials": 15,
        "patience_completed_trials": 10,
        "min_improvement_abs": 0.00025,
    },
}

resume2 = dict(common_random)
resume2["stopping"] = {
    "max_trials": 2,
    "max_completed_trials": None,
    "timeout_seconds": None,
    "plateau": {
        "enabled": False,
        "min_completed_trials": 15,
        "patience_completed_trials": 10,
        "min_improvement_abs": 0.00025,
    },
}

duplicate_same = dict(duplicate_common)
duplicate_same["study_id"] = "smoke-duplicate-same-study"
duplicate_same["stopping"] = {
    "max_trials": 2,
    "max_completed_trials": None,
    "timeout_seconds": None,
    "plateau": {
        "enabled": False,
        "min_completed_trials": 15,
        "patience_completed_trials": 10,
        "min_improvement_abs": 0.00025,
    },
}

duplicate_external = dict(duplicate_common)
duplicate_external["study_id"] = "smoke-duplicate-existing-run"
duplicate_external["stopping"] = {
    "max_trials": 1,
    "max_completed_trials": None,
    "timeout_seconds": None,
    "plateau": {
        "enabled": False,
        "min_completed_trials": 15,
        "patience_completed_trials": 10,
        "min_improvement_abs": 0.00025,
    },
}

for name, payload in (
    ("resume1.json", resume1),
    ("resume2.json", resume2),
    ("duplicate_same.json", duplicate_same),
    ("duplicate_external.json", duplicate_external),
):
    (workdir / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY

run_cli_json() {
  local output_path="$1"
  shift
  (
    cd "${WORKSPACE_DIR}"
    PYTHONPATH="${ROOT_DIR}/src" "${PYTHON_BIN}" -m numereng.cli "$@" --workspace "${WORKSPACE_DIR}" \
      2>"${output_path}.stderr" | tail -n 1 >"${output_path}"
  )
}

run_cli_json "${WORKSPACE_DIR}/resume1.out.json" \
  hpo create --study-config "${WORKSPACE_DIR}/resume1.json"

run_cli_json "${WORKSPACE_DIR}/resume2.out.json" \
  hpo create --study-config "${WORKSPACE_DIR}/resume2.json"

run_cli_json "${WORKSPACE_DIR}/details.out.json" \
  hpo details --study-id smoke-random-resume --format json

run_cli_json "${WORKSPACE_DIR}/trials.out.json" \
  hpo trials --study-id smoke-random-resume --format json

run_cli_json "${WORKSPACE_DIR}/duplicate_same.out.json" \
  hpo create --study-config "${WORKSPACE_DIR}/duplicate_same.json"

run_cli_json "${WORKSPACE_DIR}/duplicate_same_trials.out.json" \
  hpo trials --study-id smoke-duplicate-same-study --format json

run_cli_json "${WORKSPACE_DIR}/duplicate_external.out.json" \
  hpo create --study-config "${WORKSPACE_DIR}/duplicate_external.json"

run_cli_json "${WORKSPACE_DIR}/duplicate_external_trials.out.json" \
  hpo trials --study-id smoke-duplicate-existing-run --format json

"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

workdir = Path(os.environ["WORKSPACE_DIR"])

def read_json(name: str) -> dict:
    return json.loads((workdir / name).read_text(encoding="utf-8"))

resume1 = read_json("resume1.out.json")
resume2 = read_json("resume2.out.json")
details = read_json("details.out.json")
trials = read_json("trials.out.json")
duplicate_same = read_json("duplicate_same.out.json")
duplicate_same_trials = read_json("duplicate_same_trials.out.json")
duplicate_external = read_json("duplicate_external.out.json")
duplicate_external_trials = read_json("duplicate_external_trials.out.json")
summary = json.loads(
    (
        workdir
        / ".numereng"
        / "experiments"
        / "exp-smoke"
        / "hpo"
        / "smoke-random-resume"
        / "study_summary.json"
    ).read_text(encoding="utf-8")
)

assert resume1["spec"]["sampler"] == {"kind": "random", "seed": 17}
assert resume1["attempted_trials"] == 1
assert resume1["completed_trials"] == 1
assert resume1["failed_trials"] == 0

assert resume2["spec"]["sampler"] == {"kind": "random", "seed": 17}
assert resume2["attempted_trials"] == 2
assert resume2["completed_trials"] == 2
assert resume2["failed_trials"] == 0
assert resume2["stop_reason"] == "max_trials_reached"

assert details["attempted_trials"] == 2
assert details["completed_trials"] == 2
assert len(trials["trials"]) == 2
assert summary["attempted_trials"] == 2
assert summary["completed_trials"] == 2
assert summary["spec"]["sampler"] == {"kind": "random", "seed": 17}

study_root = workdir / ".numereng" / "experiments" / "exp-smoke" / "hpo" / "smoke-random-resume"
assert (study_root / "study_spec.json").is_file()
assert (study_root / "study_summary.json").is_file()
assert (study_root / "optuna_journal.log").is_file()
assert (study_root / "trials_live.parquet").is_file()

same_trials = duplicate_same_trials["trials"]
assert duplicate_same["completed_trials"] == 2
assert len(same_trials) == 2
same_run_ids = {trial["run_id"] for trial in same_trials}
assert len(same_run_ids) == 1
same_run_id = next(iter(same_run_ids))

external_trials = duplicate_external_trials["trials"]
assert duplicate_external["completed_trials"] == 1
assert len(external_trials) == 1
assert external_trials[0]["run_id"] == same_run_id

print(
    json.dumps(
        {
            "status": "ok",
            "workspace": str(workdir),
            "resume_study_id": resume2["study_id"],
            "resume_attempted_trials": resume2["attempted_trials"],
            "resume_completed_trials": resume2["completed_trials"],
            "duplicate_reused_run_id": same_run_id,
        }
    )
)
PY
