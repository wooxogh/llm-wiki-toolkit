"""Small localhost protocol shared by the daemon and its CLI clients."""
from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path
from typing import Any

from .io import read_json, write_json


DAEMON_STATE_FILE = "daemon.json"
AUTOEMBED_STATE_FILE = "autoembed.json"


def daemon_state_path(artifact_dir: Path) -> Path:
    return artifact_dir / DAEMON_STATE_FILE


def autoembed_state_path(artifact_dir: Path) -> Path:
    return artifact_dir / AUTOEMBED_STATE_FILE


def append_log(artifact_dir: Path, name: str, message: str) -> None:
    """Append one human-readable local service event without touching sources."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with (artifact_dir / name).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(f"[{timestamp}] {message}\n")


def read_daemon_state(artifact_dir: Path) -> dict[str, Any] | None:
    state = read_json(daemon_state_path(artifact_dir), None)
    return state if isinstance(state, dict) else None


def write_daemon_state(artifact_dir: Path, state: dict[str, Any]) -> None:
    write_json(daemon_state_path(artifact_dir), state)


def clear_daemon_state(artifact_dir: Path, *, pid: int | None = None) -> None:
    path = daemon_state_path(artifact_dir)
    state = read_daemon_state(artifact_dir)
    if pid is not None and state and int(state.get("pid", -1)) != pid:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def process_running(pid: Any) -> bool:
    try:
        resolved_pid = int(pid)
    except (TypeError, ValueError):
        return False
    if resolved_pid <= 0:
        return False
    if os.name == "nt":
        # ``os.kill(pid, 0)`` is not a reliable existence probe on every
        # supported Windows Python build. OpenProcess works for the same-user
        # daemon without delivering a signal or changing its state.
        try:
            import ctypes

            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, resolved_pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if not handle:
                return False
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        except (AttributeError, OSError):
            return False
    try:
        os.kill(resolved_pid, 0)
    except OSError:
        return False
    return True


def request(artifact_dir: Path, payload: dict[str, Any], *, timeout: float = 30.0) -> dict[str, Any] | None:
    """Send one newline-delimited JSON request to this vault's daemon.

    ``None`` means no usable daemon is running. Daemon-reported failures are
    returned unchanged so callers can show the actual error.
    """
    state = read_daemon_state(artifact_dir)
    if not state or not process_running(state.get("pid")):
        return None
    host = str(state.get("host") or "127.0.0.1")
    try:
        port = int(state["port"])
        with socket.create_connection((host, port), timeout=timeout) as connection:
            connection.sendall(json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n")
            connection.shutdown(socket.SHUT_WR)
            response = bytearray()
            while True:
                block = connection.recv(65536)
                if not block:
                    break
                response.extend(block)
    except (KeyError, OSError, ValueError):
        return None
    try:
        value = json.loads(response.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None
