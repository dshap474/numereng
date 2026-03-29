from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "run_lifecycle_smoke.py"
_SPEC = importlib.util.spec_from_file_location("numereng_test_run_lifecycle_smoke", _SCRIPT_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
run_lifecycle_smoke = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = run_lifecycle_smoke
_SPEC.loader.exec_module(run_lifecycle_smoke)


def test_finalize_generated_session_dir_removes_successful_session(tmp_path: Path, capsys) -> None:
    generated_dir = tmp_path / "session-success"
    generated_dir.mkdir(parents=True)
    (generated_dir / "generated.json").write_text("{}", encoding="utf-8")

    run_lifecycle_smoke.finalize_generated_session_dir(generated_dir=generated_dir, success=True)

    assert not generated_dir.exists()
    assert "[smoke] cleaned generated_dir=" in capsys.readouterr().out


def test_finalize_generated_session_dir_preserves_failed_session(tmp_path: Path, capsys) -> None:
    generated_dir = tmp_path / "session-failed"
    generated_dir.mkdir(parents=True)
    (generated_dir / "generated.json").write_text("{}", encoding="utf-8")

    run_lifecycle_smoke.finalize_generated_session_dir(generated_dir=generated_dir, success=False)

    assert generated_dir.exists()
    assert "[smoke] retained generated_dir=" in capsys.readouterr().out
