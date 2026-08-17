#!/usr/bin/env python3
"""Resident local embedding server for the vault.

Loads the embedding model ONCE at startup and keeps it warm in memory, so
`wiki-recall` queries answer in ~0.1s instead of paying the multi-second cold
model-load on every call. Local, offline, NO API key. Listens only on 127.0.0.1.

Idle cost: the model sits in RAM using ~0% CPU/GPU; it only computes when a
query arrives. Start via launchd (RunAtLoad+KeepAlive) — see install.sh.

  POST /embed  {"text": "..."}  -> {"vector": [...]}   (query-side, instruction-wrapped)
  GET  /health                  -> {"ok": true, "model": ...}
"""
from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from llm_wiki.retrieval._embedder import MODEL, embed_query_local

HOST = "127.0.0.1"
PORT = int(os.environ.get("WIKI_EMBED_PORT", "8477"))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence access logging
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/health":
            self._json({"ok": True, "model": MODEL})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/embed":
            return self._json({"error": "not found"}, 404)
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        vec = embed_query_local(body.get("text", "")).tolist()
        self._json({"vector": vec})


if __name__ == "__main__":
    embed_query_local("warmup")  # preload the model before serving
    print(f"embed server listening on {HOST}:{PORT} ({MODEL})", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
