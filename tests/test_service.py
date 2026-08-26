from __future__ import annotations

import json
import os
import socketserver
import tempfile
import threading
import unittest
from pathlib import Path

from llm_wiki_v3.service import clear_daemon_state, request, write_daemon_state


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        payload = json.loads(self.rfile.readline().decode("utf-8"))
        self.wfile.write(json.dumps({"ok": True, "echo": payload}).encode("utf-8") + b"\n")


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


class ServiceProtocolTests(unittest.TestCase):
    def test_request_uses_artifact_local_daemon_state(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory)
            with _Server(("127.0.0.1", 0), _Handler) as server:
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                write_daemon_state(
                    artifacts,
                    {"pid": os.getpid(), "host": "127.0.0.1", "port": server.server_address[1]},
                )
                self.assertEqual(request(artifacts, {"action": "status"}), {"ok": True, "echo": {"action": "status"}})
                clear_daemon_state(artifacts, pid=os.getpid())
                server.shutdown()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
