from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llm_wiki_v3.config import Config
from llm_wiki_v3.hygiene import apply_decision, apply_events


def config(root: Path) -> Config:
    return Config(root, root, root / ".llm_wiki_v3", root / "_wiki_corrections")


def chunk(chunk_id: str, text: str, source: str = "doc.md") -> dict:
    import hashlib

    return {
        "id": chunk_id,
        "text": text,
        "source_text": text,
        "source_path": source,
        "content_hash": hashlib.sha256(text.encode()).hexdigest(),
        "heading_path": [],
    }


class HygieneTests(unittest.TestCase):
    def test_partial_supersede_keeps_old_searchable_and_links_successor(self):
        rows = [chunk("old", "Retry count is 2."), chunk("new", "Retry count is 3.", "new.md")]
        event = {
            "type": "partial_supersede",
            "event_id": "event:1",
            "old_chunk_id": "old",
            "superseding_chunk_id": "new",
            "claim_id": "retry",
            "quote": "Retry count is 2.",
        }
        updated = apply_events(rows, [event])
        self.assertTrue(updated[0]["searchable"])
        self.assertEqual(updated[0]["status"], "active")
        self.assertEqual(updated[0]["superseded_claims"][0]["chunk_text_start"], None)
        self.assertEqual(updated[0]["superseded_claims"][0]["superseded_by_chunk_id"], "new")
        self.assertEqual(updated[1]["supersedes"][0]["chunk_id"], "old")

    def test_error_correction_creates_source_and_retracts_old(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = config(root)
            old = chunk("old", "Retry count is 2.")
            decision = {
                "type": "error_correction",
                "user_approved": True,
                "old_chunk_id": "old",
                "expected_content_hash": old["content_hash"],
                "quote": "Retry count is 2.",
                "corrected_text": "Retry count is 3. This replaces the incorrect value while preserving context.",
                "reason": "user correction",
            }
            event = apply_decision(cfg, decision, [old])
            source = root / event["correction_source"]
            self.assertTrue(source.is_file())
            replacement = chunk("replacement", "Retry count is 3.", event["correction_source"])
            updated = apply_events([old, replacement], [event])
            self.assertFalse(updated[0]["searchable"])
            self.assertEqual(updated[0]["status"], "retracted")
            self.assertEqual(updated[0]["replaced_by"], ["replacement"])
            self.assertEqual(updated[1]["corrects"], ["old"])
            self.assertEqual(updated[1]["corrected_at"], event["decided_at"])

    def test_decision_without_user_approval_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            old = chunk("old", "wrong")
            with self.assertRaisesRegex(ValueError, "user_approved"):
                apply_decision(config(Path(directory)), {"type": "error_correction"}, [old])

    def test_partial_supersede_can_create_a_new_resolution_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = config(root)
            old = chunk("old", "Retry count is 2.")
            event = apply_decision(cfg, {
                "type": "partial_supersede",
                "user_approved": True,
                "old_chunk_id": "old",
                "expected_content_hash": old["content_hash"],
                "claim_id": "retry",
                "quote": "Retry count is 2.",
                "replacement_quote": "Retry count is 3.",
                "replacement_text": "The current retry count is 3. This setting applies to the default runtime.",
                "reason": "user confirmed the current value",
            }, [old])
            self.assertIn("resolution_source", event)
            self.assertEqual(event["chunk_text_start"], 0)
            self.assertEqual(event["chunk_text_end"], len("Retry count is 2."))
            self.assertTrue((root / event["resolution_source"]).is_file())
            replacement = chunk("resolution", "Retry count is 3.", event["resolution_source"])
            updated = apply_events([old, replacement], [event])
            self.assertTrue(updated[0]["searchable"])
            self.assertEqual(updated[0]["superseded_claims"][0]["superseded_by_chunk_id"], "resolution")
            self.assertEqual(updated[1]["supersedes"][0]["chunk_id"], "old")


if __name__ == "__main__":
    unittest.main()
