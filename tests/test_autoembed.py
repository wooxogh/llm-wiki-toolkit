from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from llm_wiki_v3.autoembed import _changed, _snapshot
from llm_wiki_v3.config import Config


class AutoEmbedTests(unittest.TestCase):
    def test_changed_reports_add_modify_and_delete(self):
        before = {"same.md": (1, 10), "modified.md": (1, 10), "deleted.md": (1, 10)}
        after = {"same.md": (1, 10), "modified.md": (2, 10), "added.md": (3, 10)}
        self.assertEqual(_changed(before, after), ["added.md", "deleted.md", "modified.md"])

    def test_snapshot_includes_markdown_but_excludes_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            (root / "docs" / "note.md").write_text("note", encoding="utf-8")
            (root / ".llm_wiki_v3").mkdir()
            (root / ".llm_wiki_v3" / "derived.md").write_text("derived", encoding="utf-8")
            config = Config(root, root, root / ".llm_wiki_v3", root / "_wiki_corrections")
            self.assertEqual(set(_snapshot(config)), {"docs/note.md"})


if __name__ == "__main__":
    unittest.main()
