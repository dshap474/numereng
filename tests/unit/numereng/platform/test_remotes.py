from __future__ import annotations

from pathlib import Path

import pytest

from numereng.platform import load_remote_targets, resolve_remote_profiles_dir
from numereng.platform.remotes.contracts import RemoteTargetError, SshRemoteTargetProfile
from numereng.platform.remotes.ssh import build_monitor_snapshot_command, build_remote_python_command


def test_resolve_remote_profiles_dir_prefers_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    profiles_dir = tmp_path / "profiles"
    monkeypatch.setenv("NUMERENG_REMOTE_PROFILES_DIR", str(profiles_dir))

    assert resolve_remote_profiles_dir() == profiles_dir.resolve()


def test_load_remote_targets_reads_enabled_yaml_profiles(tmp_path: Path) -> None:
    profile_path = tmp_path / "remote.yaml"
    profile_path.write_text(
        "\n".join(
            [
                "id: workstation",
                "label: Workstation",
                "kind: ssh",
                "ssh_config_host: workstation",
                "repo_root: /srv/numereng",
                "store_root: /srv/numereng/.numereng",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    targets = load_remote_targets(tmp_path)

    assert [target.id for target in targets] == ["workstation"]
    assert targets[0].runner_cmd == "uv run numereng"
    assert targets[0].python_cmd == "uv run python"
    assert targets[0].shell == "posix"


def test_load_remote_targets_accepts_powershell_shell_for_windows_targets(tmp_path: Path) -> None:
    profile_path = tmp_path / "pc.yaml"
    profile_path.write_text(
        "\n".join(
            [
                "id: pc",
                "label: Windows remote example",
                "kind: ssh",
                "ssh_config_host: pc",
                "shell: powershell",
                r"repo_root: 'C:\Users\<you>\remote-access\numereng'",
                r"store_root: 'C:\Users\<you>\remote-access\numereng\.numereng'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    targets = load_remote_targets(tmp_path)

    assert [target.id for target in targets] == ["pc"]
    assert targets[0].python_cmd == "uv run python"
    assert targets[0].shell == "powershell"


def test_load_remote_targets_skips_duplicate_ids_unless_strict(tmp_path: Path) -> None:
    payload = "\n".join(
        [
            "id: workstation",
            "label: Workstation",
            "kind: ssh",
            "ssh_config_host: workstation",
            "repo_root: /srv/numereng",
            "store_root: /srv/numereng/.numereng",
        ]
    )
    (tmp_path / "a.yaml").write_text(payload + "\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text(payload.replace("Workstation", "Workstation B") + "\n", encoding="utf-8")

    targets = load_remote_targets(tmp_path)
    assert [target.label for target in targets] == ["Workstation"]

    with pytest.raises(RemoteTargetError, match="duplicate remote target id"):
        load_remote_targets(tmp_path, strict=True)


def test_load_remote_targets_rejects_invalid_payload_in_strict_mode(tmp_path: Path) -> None:
    (tmp_path / "bad.yaml").write_text(
        "\n".join(
            [
                "id: broken",
                "label: Broken",
                "kind: ssh",
                "repo_root: /srv/numereng",
                "store_root: /srv/numereng/.numereng",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert load_remote_targets(tmp_path) == []
    with pytest.raises(RemoteTargetError):
        load_remote_targets(tmp_path, strict=True)


def test_build_monitor_snapshot_command_uses_posix_shell() -> None:
    target = SshRemoteTargetProfile(
        id="gpu",
        label="GPU",
        ssh_config_host="gpu",
        repo_root="/srv/numereng",
        store_root="/srv/numereng/.numereng",
    )

    command = build_monitor_snapshot_command(target)

    assert command == "cd /srv/numereng && uv run numereng monitor snapshot --store-root /srv/numereng/.numereng --json"


def test_build_remote_python_command_uses_powershell_shell() -> None:
    target = SshRemoteTargetProfile(
        id="pc",
        label="PC",
        ssh_config_host="pc",
        shell="powershell",
        repo_root=r"C:\Users\<you>\remote-access\numereng",
        store_root=r"C:\Users\<you>\remote-access\numereng\.numereng",
    )

    command = build_remote_python_command(target, "print('ok')", args=["first arg"], cwd=target.repo_root)

    assert command.startswith('powershell -NoProfile -Command "Set-Location ')
    assert "uv run python -c 'import base64,sys; exec(base64.b64decode(sys.argv.pop(1)).decode())'" in command
    assert "'first arg'" in command
