from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from importlib.resources import files
from pathlib import Path
from unittest.mock import patch

import tomllib

from llm_wiki_v3.chunking.config import DEFAULT_CHECKPOINT
from llm_wiki_v3.pathing import relative_to_root
from llm_wiki_v3.skill_install import install


EXPECTED_CHECKPOINT_SHA256 = (
    "2887a57e41ba9c832f4562099b4a824ec68af9dd43c1c782d17acfb5d205b4e8"
)


class StandalonePackageTests(unittest.TestCase):
    def test_checkpoint_is_packaged_with_matching_metadata(self) -> None:
        self.assertTrue(DEFAULT_CHECKPOINT.is_file())
        digest = hashlib.sha256(DEFAULT_CHECKPOINT.read_bytes()).hexdigest()
        metadata_path = Path(files("llm_wiki_v3").joinpath("assets", "checkpoint.json"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(digest, EXPECTED_CHECKPOINT_SHA256)
        self.assertEqual(metadata["sha256"], digest)
        self.assertEqual(metadata["bytes"], DEFAULT_CHECKPOINT.stat().st_size)

    def test_packaged_skill_installs_for_both_hosts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp)
            destinations = install("both", home=home, codex_home=home / "codex")
            self.assertEqual(len(destinations), 2)
            for destination in destinations:
                self.assertTrue((destination / "SKILL.md").is_file())
                self.assertTrue((destination / "references" / "decisions.md").is_file())

    def test_runtime_source_has_no_research_workspace_imports(self) -> None:
        package_root = Path(files("llm_wiki_v3"))
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in package_root.rglob("*.py")
        )
        self.assertNotIn("from chunk_model", source)
        self.assertNotIn("embedding_test", source)

    def test_vault_relative_path_prefers_lexical_windows_spelling(self) -> None:
        root = Path("C:/Users/RUNNER~1/AppData/Local/Temp/example")
        source = root / "nested" / "doc.md"
        with patch.object(Path, "resolve", side_effect=AssertionError("resolve is fallback only")):
            self.assertEqual(relative_to_root(source, root), Path("nested/doc.md"))

    def test_public_cli_commands_are_packaged(self) -> None:
        project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
        scripts = project["project"]["scripts"]
        self.assertEqual(scripts["wiki-embed"], "llm_wiki_v3.indexing:main")
        self.assertEqual(scripts["wiki-search"], "llm_wiki_v3.search:main")
        self.assertEqual(scripts["wiki-daemon"], "llm_wiki_v3.daemon:main")
        self.assertEqual(scripts["wiki-autoembed"], "llm_wiki_v3.autoembed:main")


if __name__ == "__main__":
    unittest.main()
