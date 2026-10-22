"""Autostart registration: Linux systemd --user unit, Windows Task Scheduler.

Linux writes a real unit file (enabling is one printed systemctl command, or
`brainbox autostart install --enable`). Windows scheduling is generated as a
schtasks command — on Windows `brainbox autostart install` runs it; elsewhere
it is printed for the user. Nothing here pretends a Windows machine was
available for testing (see docs/windows.md).
"""

from __future__ import annotations

import os
import platform
import sys


def default_exec() -> str:
    return f"{sys.executable} -m brainbox"


def systemd_unit(exec_cmd: str, watch_dir: str, home: str | None = None) -> str:
    env = f"\nEnvironment=BRAINBOX_HOME={home}" if home else ""
    return f"""[Unit]
Description=brainbox watch runner (neurobrain runtime)
After=default.target

[Service]
Type=simple
ExecStart={exec_cmd} run --watch {watch_dir}{env}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
"""


def systemd_path() -> str:
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return os.path.join(base, "systemd", "user", "brainbox.service")


def install_systemd(exec_cmd: str, watch_dir: str, home: str | None = None) -> str:
    path = systemd_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(systemd_unit(exec_cmd, watch_dir, home))
    return path


def uninstall_systemd() -> bool:
    path = systemd_path()
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def schtasks_create_command(exe: str, watch_dir: str) -> str:
    return (f'schtasks /Create /TN "brainbox" '
            f'/TR "{exe} run --watch {watch_dir}" /SC ONLOGON /RL LIMITED /F')


def schtasks_delete_command() -> str:
    return 'schtasks /Delete /TN "brainbox" /F'


def status() -> dict:
    unit = systemd_path()
    return {"platform": platform.system(),
            "systemd_unit": os.path.exists(unit), "systemd_path": unit,
            "schtasks_command": schtasks_create_command(
                "brainbox", r"%USERPROFILE%\.brainbox\inbox")}
