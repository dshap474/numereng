from __future__ import annotations

import json
from pathlib import Path

import pytest

from numereng.config.hpo import (
    HpoConfigLoaderError,
    ensure_json_config_path,
    load_hpo_study_config_json,
)


def _write_valid_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "study_name": "lgbm-sweep",
                "config_path": "configs/base.json",
                "experiment_id": "2026-02-22_exp",
                "metric": "bmc_last_200_eras.mean",
                "direction": "maximize",
                "n_trials": 20,
                "sampler": "tpe",
                "seed": 1337,
                "search_space": {
                    "model.params.learning_rate": {
                        "type": "float",
                        "low": 0.001,
                        "high": 0.1,
                        "log": True,
                    },
                    "model.params.num_leaves": {
                        "type": "int",
                        "low": 16,
                        "high": 128,
                        "step": 8,
                    },
                },
                "neutralization": {
                    "enabled": True,
                    "neutralizer_path": "neutralizers.parquet",
                    "proportion": 0.5,
                    "mode": "era",
                    "neutralizer_cols": ["feature_a", "feature_b"],
                    "rank_output": True,
                },
            }
        ),
        encoding="utf-8",
    )


def test_load_hpo_study_config_json_accepts_canonical_payload(tmp_path: Path) -> None:
    config_path = tmp_path / "study.json"
    _write_valid_config(config_path)

    payload = load_hpo_study_config_json(config_path)
    assert payload["study_name"] == "lgbm-sweep"
    assert payload["config_path"] == "configs/base.json"
    assert payload["n_trials"] == 20
    assert isinstance(payload["search_space"], dict)

    neutralization = payload["neutralization"]
    assert isinstance(neutralization, dict)
    assert neutralization["enabled"] is True
    assert neutralization["neutralizer_path"] == "neutralizers.parquet"


def test_load_hpo_study_config_json_rejects_unknown_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "study.json"
    _write_valid_config(config_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["unknown"] = "x"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(HpoConfigLoaderError, match="hpo_study_config_schema_invalid"):
        load_hpo_study_config_json(config_path)


def test_ensure_json_config_path_validation() -> None:
    assert ensure_json_config_path("configs/study.json") == "configs/study.json"

    with pytest.raises(ValueError, match="study_config_path must reference a .json file"):
        ensure_json_config_path("configs/study.yaml")


def test_load_hpo_study_config_json_rejects_invalid_search_space_spec(tmp_path: Path) -> None:
    config_path = tmp_path / "study.json"
    _write_valid_config(config_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["search_space"]["model.params.learning_rate"]["high"] = None
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(HpoConfigLoaderError, match="hpo_study_config_schema_invalid"):
        load_hpo_study_config_json(config_path)


def test_load_hpo_study_config_json_preserves_null_seed(tmp_path: Path) -> None:
    config_path = tmp_path / "study.json"
    _write_valid_config(config_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["seed"] = None
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_hpo_study_config_json(config_path)
    assert "seed" in loaded
    assert loaded["seed"] is None
