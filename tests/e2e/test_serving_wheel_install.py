from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.e2e
@pytest.mark.slow
def test_serving_wheel_install_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    dist_dir = tmp_path / "dist"
    venv_dir = tmp_path / "venv"
    wheel_env = dict(os.environ)
    wheel_env["UV_CACHE_DIR"] = str(tmp_path / ".uv-cache")

    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(dist_dir)],
        cwd=repo_root,
        env=wheel_env,
        check=True,
    )
    wheel_path = max(dist_dir.glob("numereng-*.whl"), key=lambda path: path.stat().st_mtime)

    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], cwd=repo_root, env=wheel_env, check=True)
    python_bin = venv_dir / "bin" / "python"
    subprocess.run([str(python_bin), "-m", "pip", "install", str(wheel_path)], cwd=repo_root, env=wheel_env, check=True)
    smoke = """
import cloudpickle
import numereng
import numereng.api as api
import numereng.features.serving as serving
from numereng.cli import main

assert callable(api.serve_package_create)
assert hasattr(serving, "inspect_package")
assert main(["serve", "--help"]) == 0
"""
    subprocess.run([str(python_bin), "-c", smoke], cwd=repo_root, env=wheel_env, check=True)
