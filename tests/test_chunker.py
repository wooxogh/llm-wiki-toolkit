from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from llm_wiki_v3.chunking.chunker import HybridMarkdownChunker
from llm_wiki_v3.chunking.config import ChunkerConfig
from llm_wiki_v3.chunking.semantic import BoundaryContext
from llm_wiki_v3.chunking.structural import parse_markdown_blocks, sentence_spans


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts):
        self.calls.append(list(texts))
        vectors = np.zeros((len(texts), 4), dtype=np.float32)
        for index in range(len(texts)):
            vectors[index, index % 4] = 1.0
        return vectors


class FakeVerifier:
    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = probabilities
        self.contexts: list[BoundaryContext] = []

    def predict(self, contexts):
        self.contexts.extend(contexts)
        return self.probabilities[: len(contexts)]


class ChunkerV3Tests(unittest.TestCase):
    def config(self, cache: Path, **changes) -> ChunkerConfig:
        return replace(ChunkerConfig(cache_directory=cache), **changes)

    def test_structure_preserves_frontmatter_headings_and_special_blocks(self) -> None:
        markdown = """---
id: example
---

# Root

Short paragraph.

## Child

| A | B |
|---|---|
| 1 | 2 |

```python
print('x')
```
"""
        blocks = parse_markdown_blocks(markdown)

        self.assertEqual(
            [block.kind for block in blocks],
            ["frontmatter", "heading", "paragraph", "heading", "table+code_fence"],
        )
        self.assertEqual(blocks[2].heading_path, ("Root",))
        self.assertEqual(blocks[-1].heading_path, ("Root", "Child"))
        for block in blocks:
            self.assertEqual(markdown[block.source_start : block.source_end], block.text)

    def test_all_plain_paragraphs_enter_semantic_pipeline(self) -> None:
        markdown = "# H\n\nOne sentence.\n\nA first sentence. A second sentence. A third sentence.\n\n- item one\n- item two"
        with tempfile.TemporaryDirectory() as temp:
            embedder = FakeEmbedder()
            verifier = FakeVerifier([0.9, 0.9])
            chunker = HybridMarkdownChunker(
                self.config(Path(temp)),
                embedder=embedder,
                verifier=verifier,
            )
            result = chunker.chunk_markdown(markdown, document_id="doc")

        self.assertEqual(len(embedder.calls), 1)
        self.assertEqual(len(embedder.calls[0]), 4)
        self.assertEqual(result.summary()["semantic_refined_block_count"], 2)
        self.assertEqual(result.summary()["semantic_prose_block_count"], 2)
        self.assertTrue(any(chunk.kind == "list" for chunk in result.chunks))
        self.assertTrue(any(not chunk.semantic_refined for chunk in result.chunks))

    def test_soft_wrapped_line_is_not_a_sentence_boundary(self) -> None:
        markdown = "A sentence continues\nacross a Markdown soft wrap. A second sentence follows."
        block = parse_markdown_blocks(markdown)[0]

        spans = sentence_spans(block)

        self.assertEqual(len(spans), 2)
        self.assertEqual(spans[0].text, "A sentence continues\nacross a Markdown soft wrap.")
        self.assertEqual(spans[1].text, "A second sentence follows.")

    def test_soft_wrap_cannot_create_a_chunk_boundary(self) -> None:
        markdown = "A sentence continues\nacross a Markdown soft wrap. A second sentence follows."
        with tempfile.TemporaryDirectory() as temp:
            result = HybridMarkdownChunker(
                self.config(Path(temp), boundary_keep_threshold=0.50),
                embedder=FakeEmbedder(),
                verifier=FakeVerifier([0.9]),
            ).chunk_markdown(markdown, document_id="doc")

        self.assertEqual(len(result.boundary_decisions), 1)
        self.assertEqual(len(result.chunks), 2)
        self.assertEqual(
            result.chunks[0].text,
            "A sentence continues across a Markdown soft wrap.",
        )
        self.assertEqual(
            result.chunks[0].source_text,
            "A sentence continues\nacross a Markdown soft wrap.",
        )
        self.assertEqual(result.chunks[1].text, "A second sentence follows.")

    def test_blank_line_prose_boundary_enters_attention_and_can_merge(self) -> None:
        markdown = "# Topic\n\nFirst paragraph sentence. Second sentence.\n\nNext paragraph sentence. Final sentence."
        with tempfile.TemporaryDirectory() as temp:
            result = HybridMarkdownChunker(
                self.config(Path(temp), candidate_budget=1.0),
                embedder=FakeEmbedder(),
                verifier=FakeVerifier([0.0, 0.0, 0.0]),
            ).chunk_markdown(markdown, document_id="doc")

        cross_block = [
            decision
            for decision in result.boundary_decisions
            if decision.right_block_index is not None
        ]
        self.assertEqual(len(cross_block), 1)
        self.assertTrue(cross_block[0].is_candidate)
        self.assertFalse(cross_block[0].keep_boundary)
        self.assertEqual(len(result.chunks), 1)
        self.assertIn("Second sentence.\n\nNext paragraph sentence.", result.chunks[0].text)

    def test_heading_still_blocks_cross_paragraph_merge(self) -> None:
        markdown = "# Root\n\nRoot paragraph.\n\n## Child\n\nChild paragraph."
        with tempfile.TemporaryDirectory() as temp:
            result = HybridMarkdownChunker(
                self.config(Path(temp), candidate_budget=1.0),
                embedder=FakeEmbedder(),
                verifier=FakeVerifier([]),
            ).chunk_markdown(markdown, document_id="doc")

        self.assertFalse(
            any(decision.right_block_index is not None for decision in result.boundary_decisions)
        )
        self.assertEqual(len(result.chunks), 2)

    def test_chunk_text_joins_prose_soft_wraps_only(self) -> None:
        markdown = "A sentence continues\nacross a soft wrap.\n\nA new paragraph starts."
        with tempfile.TemporaryDirectory() as temp:
            result = HybridMarkdownChunker(
                self.config(Path(temp)),
                embedder=FakeEmbedder(),
                verifier=FakeVerifier([0.0]),
            ).chunk_markdown(markdown, document_id="doc")

        self.assertEqual(result.chunks[0].text, "A sentence continues across a soft wrap.\n\nA new paragraph starts.")
        self.assertEqual(
            result.chunks[0].source_text,
            "A sentence continues\nacross a soft wrap.\n\nA new paragraph starts.",
        )

    def test_composite_prose_and_code_normalizes_only_the_prose_prefix(self) -> None:
        markdown = "Sentence continues\nonto the next line.\n\n```yaml\nkey: value\n```"
        with tempfile.TemporaryDirectory() as temp:
            result = HybridMarkdownChunker(
                self.config(Path(temp)),
                embedder=FakeEmbedder(),
                verifier=FakeVerifier([]),
            ).chunk_markdown(markdown, document_id="doc")

        chunk = result.chunks[0]
        self.assertEqual(chunk.kind, "paragraph+code_fence")
        self.assertEqual(
            chunk.text,
            "Sentence continues onto the next line.\n\n```yaml\nkey: value\n```",
        )
        self.assertEqual(chunk.source_text, markdown)

    def test_lower_threshold_never_produces_fewer_chunks(self) -> None:
        markdown = "A one. A two. A three. A four. A five."
        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp)
            fine = HybridMarkdownChunker(
                self.config(cache, boundary_keep_threshold=0.50),
                embedder=FakeEmbedder(),
                verifier=FakeVerifier([0.60, 0.70]),
            ).chunk_markdown(markdown, document_id="fine")
            coarse = HybridMarkdownChunker(
                self.config(cache, boundary_keep_threshold=0.80),
                embedder=FakeEmbedder(),
                verifier=FakeVerifier([0.60, 0.70]),
            ).chunk_markdown(markdown, document_id="coarse")

        self.assertGreaterEqual(len(fine.chunks), len(coarse.chunks))
        self.assertEqual(len(fine.chunks), 3)
        self.assertEqual(len(coarse.chunks), 1)

    def test_candidate_budget_limits_attention_calls(self) -> None:
        markdown = "One. Two. Three. Four. Five."
        with tempfile.TemporaryDirectory() as temp:
            verifier = FakeVerifier([0.0, 0.0])
            result = HybridMarkdownChunker(
                self.config(Path(temp), candidate_budget=0.50),
                embedder=FakeEmbedder(),
                verifier=verifier,
            ).chunk_markdown(markdown, document_id="doc")

        self.assertEqual(len(result.boundary_decisions), 4)
        self.assertEqual(sum(item.is_candidate for item in result.boundary_decisions), 2)
        self.assertEqual(len(verifier.contexts), 2)

    def test_chunks_retain_exact_source_spans_and_heading_path(self) -> None:
        markdown = "# Topic\n\nFirst sentence. Second sentence. Third sentence."
        with tempfile.TemporaryDirectory() as temp:
            result = HybridMarkdownChunker(
                self.config(Path(temp), boundary_keep_threshold=0.50),
                embedder=FakeEmbedder(),
                verifier=FakeVerifier([0.9]),
            ).chunk_markdown(markdown, document_id="doc", source_path="doc.md")

        for chunk in result.chunks:
            self.assertEqual(markdown[chunk.source_start : chunk.source_end], chunk.source_text)
            self.assertEqual(chunk.heading_path, ("Topic",))
            self.assertEqual(chunk.chunk_heading, "Topic")
            self.assertEqual(chunk.document_id, "doc")
            self.assertEqual(chunk.source_path, "doc.md")

    def test_chunk_heading_is_immediate_parent_only(self) -> None:
        markdown = "# Root\n\n## Child\n\nBody sentence."
        with tempfile.TemporaryDirectory() as temp:
            result = HybridMarkdownChunker(
                self.config(Path(temp)),
                embedder=FakeEmbedder(),
                verifier=FakeVerifier([]),
            ).chunk_markdown(markdown, document_id="doc")

        self.assertEqual(result.chunks[0].heading_path, ("Root", "Child"))
        self.assertEqual(result.chunks[0].chunk_heading, "Child")

    def test_chunks_link_to_adjacent_chunks_and_store_document_timestamps(self) -> None:
        markdown = "# Topic\n\nOne. Two. Three. Four."
        with tempfile.TemporaryDirectory() as temp:
            result = HybridMarkdownChunker(
                self.config(Path(temp), boundary_keep_threshold=0.50),
                embedder=FakeEmbedder(),
                verifier=FakeVerifier([0.9, 0.9]),
            ).chunk_markdown(
                markdown,
                document_id="doc",
                source_path="doc.md",
                document_created_at="2026-01-01",
                document_modified_at="2026-08-25",
            )

        self.assertEqual(len(result.chunks), 3)
        self.assertIsNone(result.chunks[0].previous_chunk_id)
        self.assertEqual(result.chunks[0].next_chunk_id, result.chunks[1].id)
        self.assertEqual(result.chunks[1].previous_chunk_id, result.chunks[0].id)
        self.assertEqual(result.chunks[1].next_chunk_id, result.chunks[2].id)
        self.assertEqual(result.chunks[2].previous_chunk_id, result.chunks[1].id)
        self.assertIsNone(result.chunks[2].next_chunk_id)
        self.assertEqual(result.chunks[0].document_created_at, "2026-01-01")
        self.assertEqual(result.chunks[0].document_modified_at, "2026-08-25")


if __name__ == "__main__":
    unittest.main()
