"""Polling watcher that asks a running wiki-daemon to rebuild incrementally."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .config import Config, load
from .indexing import collect_markdown
from .io import configure_stdio, read_json, write_json
from .pathing import relative_to_root
from .service import append_log, autoembed_state_path, process_running, request


def _snapshot(config: Config) -> dict[str, tuple[int, int]]:
    return {
        relative_to_root(path, config.root).as_posix(): (path.stat().st_mtime_ns, path.stat().st_size)
        for path in collect_markdown(config)
    }


def _changed(before: dict[str, tuple[int, int]], after: dict[str, tuple[int, int]]) -> list[str]:
    return sorted({*before, *after} - {path for path in before.keys() & after.keys() if before[path] == after[path]})


def _state(config: Config) -> dict[str, Any] | None:
    value = read_json(autoembed_state_path(config.artifact_dir), None)
    return value if isinstance(value, dict) else None


def _write_state(config: Config, state: dict[str, Any]) -> None:
    write_json(autoembed_state_path(config.artifact_dir), state)


def _clear_state(config: Config, pid: int) -> None:
    state = _state(config)
    path = autoembed_state_path(config.artifact_dir)
    if state and int(state.get("pid", -1)) != pid:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _stop_path(config: Config) -> Path:
    return config.artifact_dir / "autoembed.stop"


def _embed(config: Config, paths: list[str]) -> dict[str, Any]:
    response = request(config.artifact_dir, {"action": "embed", "paths": paths}, timeout=3600.0)
    if response is None:
        raise RuntimeError("wiki-daemon is not running; start it before wiki-autoembed")
    if not response.get("ok"):
        raise RuntimeError(str(response.get("error") or "daemon embedding failed"))
    return dict(response["result"])


def _record(config: Config, **values: Any) -> None:
    current = _state(config) or {}
    current.update(values)
    _write_state(config, current)


def _event(config: Config, message: str) -> None:
    append_log(config.artifact_dir, "autoembed.log", message)
    print(f"wiki-autoembed: {message}", flush=True)


def watch(config: Config, *, interval: float, debounce: float, initial: bool) -> int:
    if interval <= 0 or debounce < 0:
        raise ValueError("--interval must be positive and --debounce must not be negative")
    existing = _state(config)
    if existing and process_running(existing.get("pid")):
        raise RuntimeError("wiki-autoembed is already running for this vault")
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    _stop_path(config).unlink(missing_ok=True)
    pid = os.getpid()
    _write_state(
        config,
        {
            "pid": pid,
            "vault": str(config.root),
            "interval_seconds": interval,
            "debounce_seconds": debounce,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "last_result": None,
        },
    )
    _event(config, f"watching {config.root}; interval={interval:g}s debounce={debounce:g}s")
    snapshot = _snapshot(config)
    pending: set[str] = set()
    changed_at: float | None = None
    if initial:
        pending.update(snapshot)
        changed_at = time.monotonic() - debounce
        _event(config, f"initial incremental embed queued for {len(snapshot)} Markdown file(s)")
    try:
        while not _stop_path(config).is_file():
            time.sleep(interval)
            next_snapshot = _snapshot(config)
            paths = _changed(snapshot, next_snapshot)
            snapshot = next_snapshot
            if paths:
                pending.update(paths)
                changed_at = time.monotonic()
                _event(config, f"detected {len(paths)} Markdown change(s); waiting {debounce:g}s before embedding")
            if pending and changed_at is not None and time.monotonic() - changed_at >= debounce:
                try:
                    _event(config, f"embedding {len(pending)} changed path(s): {', '.join(sorted(pending)[:5])}")
                    result = _embed(config, sorted(pending))
                except RuntimeError as exc:
                    # Retain paths so a restarted daemon needs no extra edit.
                    _record(config, last_error=str(exc))
                    _event(config, f"embed failed; will retry: {exc}")
                    changed_at = time.monotonic()
                else:
                    _record(config, last_result=result, last_error=None)
                    manifest = result["manifest"]
                    _event(
                        config,
                        f"embed complete; embedded={manifest['embedded_chunk_count']} reused={manifest['reused_chunk_count']} generation={result['generation']}",
                    )
                    pending.clear()
                    changed_at = None
    finally:
        _event(config, "watcher stopped")
        _clear_state(config, pid)
        _stop_path(config).unlink(missing_ok=True)
    return 0


def _background(config: Config, interval: float, debounce: float, initial: bool) -> int:
    command = [sys.executable, "-m", "llm_wiki_v3.autoembed", "start", "--vault", str(config.config_dir), "--interval", str(interval), "--debounce", str(debounce)]
    if initial:
        command.append("--initial")
    kwargs: dict[str, Any] = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen(command, **kwargs)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Watch Markdown changes and request daemon-owned incremental embedding")
    subparsers = parser.add_subparsers(dest="command", required=True)
    start = subparsers.add_parser("start", help="watch in this terminal")
    start.add_argument("--vault", type=Path, default=None)
    start.add_argument("--interval", type=float, default=1.0)
    start.add_argument("--debounce", type=float, default=2.0)
    start.add_argument("--initial", action="store_true", help="request an incremental build immediately")
    start.add_argument("--background", action="store_true")
    for name in ("status", "stop"):
        child = subparsers.add_parser(name)
        child.add_argument("--vault", type=Path, default=None)
        child.add_argument("--json", action="store_true")
    logs = subparsers.add_parser("logs", help="print watcher events written while running in the background")
    logs.add_argument("--vault", type=Path, default=None)
    logs.add_argument("--tail", type=int, default=50)
    return parser


def main() -> int:
    configure_stdio()
    args = _parser().parse_args()
    try:
        config = load(args.vault)
        if args.command == "start":
            if args.background:
                _background(config, args.interval, args.debounce, args.initial)
                print("wiki-autoembed: starting in background")
                return 0
            return watch(config, interval=args.interval, debounce=args.debounce, initial=args.initial)
        if args.command == "logs":
            if args.tail < 0:
                raise ValueError("--tail must not be negative")
            path = config.artifact_dir / "autoembed.log"
            lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
            print("\n".join(lines[-args.tail:]))
            return 0
        state = _state(config)
        if args.command == "stop":
            if state and process_running(state.get("pid")):
                _stop_path(config).touch()
                payload = {"ok": True, "stopping": True, "pid": state["pid"]}
            else:
                payload = {"ok": False, "error": "wiki-autoembed is not running"}
        else:
            payload = {"ok": bool(state and process_running(state.get("pid"))), "state": state}
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        payload = {"ok": False, "error": str(exc)}
    if getattr(args, "json", False):
        print(json.dumps(payload, ensure_ascii=False))
    elif payload.get("ok"):
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"wiki-autoembed: {payload.get('error', 'not running')}", file=sys.stderr)
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
