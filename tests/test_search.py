from __future__ import annotations

import hashlib
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from llm_wiki_v3.config import Config
from llm_wiki_v3.io import write_json, write_jsonl
from llm_wiki_v3.search import SearchEngine


class FakeEmbedder:
    def encode_query(self, query: str) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=np.float32)


class FakeReranker:
    def score(self, query: str, texts: list[str]) -> list[float]:
        return [0.1 if "two" in text else 0.9 for text in texts]


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.config = Config(root, root, root / ".llm_wiki_v3", root / "_wiki_corrections", candidate_pool=10)
        self.config.artifact_dir.mkdir()
        now = "2026-08-26T00:00:00+00:00"
        rows = []
        for chunk_id, text, path in (("old", "retry count two", "old.md"), ("new", "retry count three", "new.md"), ("bad", "unrelated", "bad.md")):
            rows.append({
                "id": chunk_id,
                "document_id": path[:-3],
                "source_path": path,
                "ordinal": 0,
                "text": text,
                "source_text": text,
                "source_start": 0,
                "source_end": len(text),
                "content_hash": hashlib.sha256(text.encode()).hexdigest(),
                "heading_path": ["Retry"],
                "document_created_at": now,
                "document_modified_at": now,
            })
        write_jsonl(self.config.artifact_dir / "chunks.jsonl", rows)
        np.save(self.config.artifact_dir / "vectors.npy", np.asarray([[1, 0], [.9, .1], [0, 1]], dtype=np.float32))
        np.savez(
            self.config.artifact_dir / "knn_graph.npz",
            indices=np.asarray([[1], [0], [1]], dtype=np.int64),
            scores=np.asarray([[.9], [.9], [.1]], dtype=np.float32),
        )
        write_jsonl(self.config.artifact_dir / "hygiene" / "events.jsonl", [{
            "type": "partial_supersede",
            "event_id": "event:1",
            "old_chunk_id": "old",
            "superseding_chunk_id": "new",
            "claim_id": "retry",
            "quote": "retry count two",
        }, {
            "type": "error_correction",
            "event_id": "event:2",
            "old_chunk_id": "bad",
            "correction_source": "missing-correction.md",
        }])

    def tearDown(self):
        self.temp.cleanup()

    def test_partial_supersede_is_searchable_and_successor_is_attached(self):
        response = SearchEngine(
            self.config,
            embedder=FakeEmbedder(),
            now=datetime(2026, 8, 26, tzinfo=timezone.utc),
        ).search("retry", k=2)
        ids = [hit.chunk["id"] for hit in response.hits]
        self.assertIn("old", ids)
        old = next(hit for hit in response.hits if hit.chunk["id"] == "old")
        self.assertEqual(old.related_evidence[0]["chunk"]["id"], "new")
        self.assertNotIn("bad", ids)

    def test_range_filters_chunks_before_ranking(self):
        engine = SearchEngine(self.config, embedder=FakeEmbedder(), now=datetime(2030, 8, 26, tzinfo=timezone.utc))
        self.assertEqual(engine.search("retry", years=1).hits, ())

    def test_reranker_reorders_the_fused_candidate_pool(self):
        engine = SearchEngine(self.config, embedder=FakeEmbedder(), reranker=FakeReranker())
        response = engine.search("retry", k=2, rerank_pool=2)
        self.assertTrue(response.reranked)
        self.assertEqual(response.hits[0].chunk["id"], "new")
        self.assertEqual(response.hits[0].rerank_score, 0.9)


if __name__ == "__main__":
    unittest.main()
