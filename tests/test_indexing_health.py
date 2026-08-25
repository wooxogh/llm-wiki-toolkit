from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from llm_wiki_v3.config import Config
from llm_wiki_v3.health import inspect
from llm_wiki_v3.indexing import build


class FakeChunk:
    def __init__(self, path: Path, document_id: str, text: str) -> None:
        self.payload = {
            "id": "chunk:" + hashlib.sha1(document_id.encode()).hexdigest()[:12],
            "document_id": document_id,
            "source_path": str(path),
            "ordinal": 0,
            "kind": "paragraph",
            "text": text,
            "source_text": text,
            "source_start": 0,
            "source_end": len(text),
            "heading_path": [],
            "chunk_heading": None,
            "previous_chunk_id": None,
            "next_chunk_id": None,
            "document_created_at": "2026-08-26T00:00:00+00:00",
            "document_modified_at": "2026-08-26T00:00:00+00:00",
            "sentence_count": 1,
            "semantic_refined": False,
            "content_hash": hashlib.sha256(text.encode()).hexdigest(),
        }

    def to_dict(self):
        return dict(self.payload)


class FakeChunker:
    def __init__(self) -> None:
        self.calls = 0

    def chunk_file(self, path: Path, *, document_id: str):
        self.calls += 1
        text = path.read_text(encoding="utf-8")
        return SimpleNamespace(chunks=(FakeChunk(path, document_id, text),))


class FakeEmbedder:
    model_id = "fake-model"

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


class IndexingHealthTests(unittest.TestCase):
    def test_incremental_reuses_vectors_and_health_is_clean(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "doc.md").write_text("A complete sentence.", encoding="utf-8")
            cfg = Config(root, root, root / ".llm_wiki_v3", root / "_wiki_corrections", model_id="fake-model")
            chunker, embedder = FakeChunker(), FakeEmbedder()
            first = build(cfg, embedder=embedder, chunker=chunker)
            second = build(cfg, embedder=embedder, chunker=chunker)
            self.assertEqual(first["embedded_chunk_count"], 1)
            self.assertEqual(second["embedded_chunk_count"], 0)
            self.assertEqual(second["reused_chunk_count"], 1)
            self.assertEqual(embedder.calls, 1)
            self.assertEqual(chunker.calls, 1)
            self.assertEqual(inspect(cfg), [])


if __name__ == "__main__":
    unittest.main()
