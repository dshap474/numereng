from __future__ import annotations

import json
from pathlib import Path

from numereng.features.submission.registry import record_submission_upload


def test_record_submission_upload_closes_previous_open_upload(tmp_path: Path) -> None:
    first_path = record_submission_upload(
        workspace_root=tmp_path,
        model_name="model_a",
        model_id="model-1",
        upload_id="upload-1",
        pickle_path=tmp_path / "model-1.pkl",
        experiment_id="exp-1",
        package_id="pkg-1",
        package_path=tmp_path / ".numereng" / "experiments" / "exp-1" / "submission_packages" / "pkg-1",
        recipe="moderate_lgbm",
        data_version="v5.2",
        docker_image="Python 3.12",
        uploaded_at="2026-05-08T00:00:00+00:00",
    )

    record_submission_upload(
        workspace_root=tmp_path,
        model_name="model_a",
        model_id="model-1",
        upload_id="upload-2",
        pickle_path=tmp_path / "model-2.pkl",
        experiment_id="exp-1",
        package_id="pkg-2",
        package_path=tmp_path / ".numereng" / "experiments" / "exp-1" / "submission_packages" / "pkg-2",
        recipe="moderate_lgbm",
        data_version="v5.2",
        docker_image="Python 3.12",
        uploaded_at="2026-05-09T00:00:00+00:00",
    )

    metadata = json.loads(first_path.read_text(encoding="utf-8"))
    assert metadata["hosted_pickle"]["upload_id"] == "upload-2"
    assert [item["upload_id"] for item in metadata["uploads"]] == ["upload-1", "upload-2"]
    assert metadata["uploads"][0]["live_ended_at"] == "2026-05-09T00:00:00+00:00"
    assert metadata["uploads"][1]["live_ended_at"] is None
