"""CLI integration tests for dataset-tools commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from numereng import api as api_module
from numereng.cli.main import main
from numereng.platform.errors import PackageError


class TestDatasetToolsRouting:
    def test_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["dataset-tools", "--help"])
        assert rc == 0
        assert "dataset-tools" in capsys.readouterr().out

    def test_no_args_prints_usage(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["dataset-tools"])
        assert rc == 0
        assert "usage:" in capsys.readouterr().out

    def test_unknown_subcommand(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["dataset-tools", "banana"])
        assert rc == 2
        assert "unknown arguments" in capsys.readouterr().err

    def test_removed_quantize_subcommand(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["dataset-tools", "quantize"])
        assert rc == 2
        assert "quantization commands are removed" in capsys.readouterr().err


def test_build_downsampled_full_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected_data_dir = str(tmp_path / "datasets")

    def fake_build(
        request: api_module.DatasetToolsBuildDownsampleRequest,
    ) -> api_module.DatasetToolsBuildDownsampleResponse:
        assert request.data_version == "v5.2"
        assert request.data_dir == expected_data_dir
        assert request.downsample_eras_step == 8
        assert request.downsample_eras_offset == 2
        assert request.rebuild is True
        return api_module.DatasetToolsBuildDownsampleResponse(
            data_version="v5.2",
            data_dir=expected_data_dir,
            downsampled_full_path=f"{expected_data_dir}/v5.2/downsampled_full.parquet",
            downsampled_full_benchmark_path=f"{expected_data_dir}/v5.2/downsampled_full_benchmark_models.parquet",
            downsampled_rows=25,
            downsampled_full_benchmark_rows=25,
            total_eras=200,
            kept_eras=50,
            downsample_step=8,
            downsample_offset=2,
        )

    monkeypatch.setattr(api_module, "dataset_tools_build_downsampled_full", fake_build)
    rc = main(
        [
            "dataset-tools",
            "build-downsampled-full",
            "--data-version",
            "v5.2",
            "--data-dir",
            expected_data_dir,
            "--downsample-eras-step",
            "8",
            "--downsample-eras-offset",
            "2",
            "--rebuild",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["downsampled_rows"] == 25
    assert payload["downsample_step"] == 8
    assert payload["downsample_offset"] == 2


def test_build_downsampled_full_boundary_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_build(
        request: api_module.DatasetToolsBuildDownsampleRequest,
    ) -> api_module.DatasetToolsBuildDownsampleResponse:
        _ = request
        raise PackageError("dataset_build_failed")

    monkeypatch.setattr(api_module, "dataset_tools_build_downsampled_full", fake_build)

    rc = main(["dataset-tools", "build-downsampled-full"])
    assert rc == 1
    assert "dataset_build_failed" in capsys.readouterr().err


def test_build_downsampled_full_is_canonical_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def fake_build(
        request: api_module.DatasetToolsBuildDownsampleRequest,
    ) -> api_module.DatasetToolsBuildDownsampleResponse:
        nonlocal called
        _ = request
        called = True
        return api_module.DatasetToolsBuildDownsampleResponse(
            data_version="v5.2",
            data_dir=".numereng/datasets",
            downsampled_full_path=".numereng/datasets/v5.2/downsampled_full.parquet",
            downsampled_full_benchmark_path=".numereng/datasets/v5.2/downsampled_full_benchmark_models.parquet",
            downsampled_rows=1,
            downsampled_full_benchmark_rows=1,
            total_eras=1,
            kept_eras=1,
            downsample_step=4,
            downsample_offset=0,
        )

    monkeypatch.setattr(api_module, "dataset_tools_build_downsampled_full", fake_build)
    rc = main(["dataset-tools", "build-downsampled-full"])
    assert rc == 0
    assert called is True
