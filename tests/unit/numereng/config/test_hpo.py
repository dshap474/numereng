from __future__ import annotations

import json
from pathlib import Path

import pytest

from numereng.config.hpo import HpoConfigLoaderError, ensure_json_config_path, load_hpo_study_config_json


def _write_valid_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "study_id": "ender20_lgbm_gpu_v1",
                "study_name": "Ender20 LGBM GPU v1",
                "config_path": "configs/base.json",
                "experiment_id": "2026-04-11_ender20_hpo",
                "objective": {
                    "metric": "post_fold_champion_objective",
                    "direction": "maximize",
                    "neutralization": {
                        "enabled": True,
                        "neutralizer_path": "neutralizers.parquet",
                        "proportion": 0.5,
                        "mode": "era",
                        "neutralizer_cols": ["feature_a", "feature_b"],
                        "rank_output": True,
                    },
                },
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
                "sampler": {
                    "kind": "tpe",
                    "seed": 1337,
                    "n_startup_trials": 10,
                    "multivariate": True,
                    "group": False,
                },
                "stopping": {
                    "max_trials": 20,
                    "max_completed_trials": 15,
                    "timeout_seconds": 3600,
                    "plateau": {
                        "enabled": True,
                        "min_completed_trials": 10,
                        "patience_completed_trials": 5,
                        "min_improvement_abs": 0.0001,
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def test_load_hpo_study_config_json_accepts_canonical_payload(tmp_path: Path) -> None:
    config_path = tmp_path / "study.json"
    _write_valid_config(config_path)

    payload = load_hpo_study_config_json(config_path)
    assert payload["study_id"] == "ender20_lgbm_gpu_v1"
    assert payload["config_path"] == "configs/base.json"
    assert payload["sampler"]["kind"] == "tpe"
    assert payload["stopping"]["max_trials"] == 20
    assert isinstance(payload["search_space"], dict)
    assert payload["objective"]["neutralization"]["neutralizer_path"] == "neutralizers.parquet"


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
    payload["sampler"]["seed"] = None
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_hpo_study_config_json(config_path)
    assert loaded["sampler"]["seed"] is None


def test_load_hpo_study_config_json_canonicalizes_random_sampler_shape(tmp_path: Path) -> None:
    config_path = tmp_path / "study.json"
    _write_valid_config(config_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["sampler"] = {
        "kind": "random",
        "seed": 17,
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_hpo_study_config_json(config_path)

    assert loaded["sampler"] == {
        "kind": "random",
        "seed": 17,
    }


def test_load_hpo_study_config_json_rejects_group_without_multivariate(tmp_path: Path) -> None:
    config_path = tmp_path / "study.json"
    _write_valid_config(config_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["sampler"]["group"] = True
    payload["sampler"]["multivariate"] = False
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(HpoConfigLoaderError, match="hpo_study_config_schema_invalid"):
        load_hpo_study_config_json(config_path)


def test_shipped_hpo_study_templates_validate_and_match() -> None:
    source_template = Path(".codex/skills/numereng-experiment-ops/assets/hpo-study-template.json")
    packaged_template = Path(
        "src/numereng/assets/shipped_skills/numereng-experiment-ops/assets/hpo-study-template.json"
    )

    source_payload = load_hpo_study_config_json(source_template)
    packaged_payload = load_hpo_study_config_json(packaged_template)

    assert source_payload == packaged_payload
    assert source_payload["study_id"] == "example-lgbm-hpo-v1"
    assert source_payload["sampler"]["kind"] == "tpe"
    assert source_payload["stopping"]["max_trials"] == 25
