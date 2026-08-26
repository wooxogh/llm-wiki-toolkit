"""Long-lived local runtime for LLM-Wiki embeddings, search, and indexing."""
from __future__ import annotations

import argparse
import json
import os
import socketserver
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .config import Config, load
from .indexing import _default_runtime, build
from .io import configure_stdio
from .search import SearchEngine, _auto_decision
from .service import append_log, clear_daemon_state, process_running, read_daemon_state, request, write_daemon_state


HOST = "127.0.0.1"
# Let each vault use an OS-selected port by default. The artifact-local state
# file tells its clients which port to use, so multiple vault daemons coexist.
DEFAULT_PORT = 0


def _startup_event(config: Config, message: str) -> None:
    append_log(config.artifact_dir, "daemon.log", message)
    print(f"wiki-daemon: {message}", flush=True)


class WikiRuntime:
    """Own one embedding model, one chunker, and one loaded index."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.lock = threading.RLock()
        self.embedder, self.chunker = _default_runtime(config)
        self.engine: SearchEngine | None = None
        self.generation = 0
        self.last_build: dict[str, Any] | None = None
        self.runtime_id = uuid.uuid4().hex[:12]

    def start(self) -> None:
        # Load the two always-used models once rather than on the first event.
        _startup_event(self.config, f"runtime={self.runtime_id} loading Qwen embedding model on {self.embedder._delegate.device}")
        self.embedder._delegate._load_model()
        _startup_event(self.config, f"runtime={self.runtime_id} Qwen embedding model ready")
        _startup_event(self.config, f"runtime={self.runtime_id} loading Small-V3 boundary model")
        self.chunker._boundary_verifier()
        _startup_event(self.config, f"runtime={self.runtime_id} Small-V3 boundary model ready")
        self.reload_index(required=False)
        _startup_event(
            self.config,
            f"runtime={self.runtime_id} startup models ready; existing_index_loaded={self.engine is not None}",
        )

    def reload_index(self, *, required: bool) -> bool:
        try:
            self.engine = SearchEngine(self.config, embedder=self.embedder)
        except RuntimeError:
            if required:
                raise
            self.engine = None
            return False
        self.generation += 1
        return True

    def status(self) -> dict[str, Any]:
        return {
            "index_loaded": self.engine is not None,
            "generation": self.generation,
            "last_build": self.last_build,
            "model_id": self.config.model_id,
            "device": self.embedder._delegate.device,
            "runtime_id": self.runtime_id,
            "qwen_loaded": self.embedder._delegate._model is not None,
            "small_v3_loaded": self.chunker._verifier is not None,
        }

    def embed(self, changed_paths: list[str] | None = None, *, full: bool = False) -> dict[str, Any]:
        with self.lock:
            paths = changed_paths or []
            append_log(
                self.config.artifact_dir,
                "daemon.log",
                f"runtime={self.runtime_id} embedding requested; changed_paths={len(paths)} full={full}",
            )
            manifest = build(self.config, full=full, embedder=self.embedder, chunker=self.chunker)
            self.last_build = manifest
            self.reload_index(required=True)
            append_log(
                self.config.artifact_dir,
                "daemon.log",
                f"runtime={self.runtime_id} embedding complete; embedded={manifest['embedded_chunk_count']} reused={manifest['reused_chunk_count']} generation={self.generation}",
            )
            return {"manifest": manifest, "changed_paths": paths, "generation": self.generation, "runtime_id": self.runtime_id}

    def search(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if self.engine is None:
                self.reload_index(required=True)
            assert self.engine is not None
            query = str(payload.get("query") or "")
            if not query:
                raise ValueError("query must not be empty")
            response = self.engine.search(
                query,
                k=int(payload.get("k", 8)),
                years=payload.get("range_years"),
            )
            result: dict[str, Any] = {
                "query": query,
                "range_years": payload.get("range_years"),
                "results": [hit.to_dict() for hit in response.hits],
                "generation": self.generation,
            }
            if payload.get("auto"):
                result["decision"], result["reason"] = _auto_decision(response, self.config)
            return result


class _RequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        try:
            raw = self.rfile.readline()
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("request must be a JSON object")
            action = payload.get("action")
            runtime: WikiRuntime = self.server.runtime  # type: ignore[attr-defined]
            if action == "status":
                result = runtime.status()
            elif action == "embed":
                paths = payload.get("paths")
                result = runtime.embed(
                    [str(path) for path in paths] if isinstance(paths, list) else None,
                    full=bool(payload.get("full")),
                )
            elif action == "search":
                result = runtime.search(payload)
            elif action == "stop":
                result = {"stopping": True}
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            else:
                raise ValueError(f"unknown daemon action: {action}")
            response = {"ok": True, "result": result}
        except (ImportError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            response = {"ok": False, "error": str(exc)}
        self.wfile.write(json.dumps(response, ensure_ascii=False).encode("utf-8") + b"\n")


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def _serve(config: Config, *, port: int) -> int:
    existing = request(config.artifact_dir, {"action": "status"}, timeout=1.0)
    if existing and existing.get("ok"):
        raise RuntimeError(f"wiki-daemon is already running for {config.root}")
    stale = read_daemon_state(config.artifact_dir)
    if stale and not process_running(stale.get("pid")):
        clear_daemon_state(config.artifact_dir)

    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    runtime = WikiRuntime(config)
    runtime.start()
    with _Server((HOST, port), _RequestHandler) as server:
        server.runtime = runtime  # type: ignore[attr-defined]
        actual_port = int(server.server_address[1])
        write_daemon_state(
            config.artifact_dir,
            {
                "pid": os.getpid(),
                "host": HOST,
                "port": actual_port,
                "vault": str(config.root),
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
        _startup_event(config, f"runtime={runtime.runtime_id} READY; listening on {HOST}:{actual_port}")
        try:
            server.serve_forever(poll_interval=0.25)
        finally:
            append_log(config.artifact_dir, "daemon.log", f"runtime={runtime.runtime_id} stopped")
            clear_daemon_state(config.artifact_dir, pid=os.getpid())
    return 0


def _background(config: Config, port: int) -> int:
    existing = request(config.artifact_dir, {"action": "status"}, timeout=1.0)
    if existing and existing.get("ok"):
        raise RuntimeError(f"wiki-daemon is already running for {config.root}")
    stale = read_daemon_state(config.artifact_dir)
    if stale and not process_running(stale.get("pid")):
        clear_daemon_state(config.artifact_dir)
    command = [sys.executable, "-m", "llm_wiki_v3.daemon", "serve", "--vault", str(config.config_dir), "--port", str(port)]
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    log = (config.artifact_dir / "daemon.log").open("a", encoding="utf-8", newline="\n")
    # Structured lifecycle events use append_log themselves. Keep stderr in
    # the same file for unexpected startup failures without duplicating output.
    kwargs: dict[str, Any] = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": log}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    try:
        process = subprocess.Popen(command, **kwargs)
    finally:
        log.close()
    # The daemon writes its state only after the resident models have loaded.
    # Waiting here prevents autoembed/search from accidentally taking their
    # standalone paths during daemon startup.
    deadline = time.monotonic() + 300.0
    started = time.monotonic()
    next_update = started
    print("wiki-daemon: starting in background; waiting for Qwen and Small-V3 to become ready...", flush=True)
    while time.monotonic() < deadline:
        response = request(config.artifact_dir, {"action": "status"}, timeout=1.0)
        if response and response.get("ok"):
            runtime_id = response["result"].get("runtime_id", "unknown")
            print(f"wiki-daemon: READY in {time.monotonic() - started:.1f}s (runtime={runtime_id})", flush=True)
            return 0
        if process.poll() is not None:
            raise RuntimeError(f"wiki-daemon exited during startup; inspect {config.artifact_dir / 'daemon.log'}")
        if time.monotonic() >= next_update:
            elapsed = time.monotonic() - started
            print(f"wiki-daemon: still preparing resident models ({elapsed:.0f}s)", flush=True)
            next_update = time.monotonic() + 5.0
        time.sleep(0.25)
    raise RuntimeError("wiki-daemon did not become ready within 300 seconds; inspect daemon.log")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Keep LLM-Wiki V3 models and index resident in one local process")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("serve", "start"):
        child = subparsers.add_parser(name, help="start the daemon (serve stays in this terminal)")
        child.add_argument("--vault", type=Path, default=None)
        child.add_argument("--port", type=int, default=DEFAULT_PORT)
        child.add_argument("--background", action="store_true", help="launch serve in the background")
    for name in ("status", "stop"):
        child = subparsers.add_parser(name)
        child.add_argument("--vault", type=Path, default=None)
        child.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    configure_stdio()
    args = _parser().parse_args()
    try:
        config = load(args.vault)
        if args.command in {"serve", "start"}:
            if args.port < 0 or args.port > 65535:
                raise ValueError("--port must be between 0 and 65535")
            if args.background:
                _background(config, args.port)
                return 0
            return _serve(config, port=args.port)
        response = request(config.artifact_dir, {"action": args.command}, timeout=10.0)
        payload = response if response is not None else {"ok": False, "error": "wiki-daemon is not running"}
    except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as exc:
        payload = {"ok": False, "error": str(exc)}
    if getattr(args, "json", False):
        print(json.dumps(payload, ensure_ascii=False))
    elif payload.get("ok"):
        print(json.dumps(payload["result"], ensure_ascii=False, indent=2))
    else:
        print(f"wiki-daemon: {payload['error']}", file=sys.stderr)
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
