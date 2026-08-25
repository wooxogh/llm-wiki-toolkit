from __future__ import annotations

import unittest

import numpy as np

from llm_wiki_v3.chunking.runtime import QwenSentenceEmbedder


class _FakeSentenceTransformer:
    def __init__(self) -> None:
        self.max_seq_length = 0
        self.received_texts: list[str] = []

    def encode(self, texts, **_kwargs):
        self.received_texts = list(texts)
        return np.asarray([[len(text), index] for index, text in enumerate(texts)], dtype=np.float32)


class QwenSentenceEmbedderTests(unittest.TestCase):
    def test_length_sorting_reduces_padding_order_and_restores_original_vector_order(self):
        embedder = QwenSentenceEmbedder("fake", batch_size=2, sort_by_length=True)
        fake_model = _FakeSentenceTransformer()
        embedder._model = fake_model
        texts = ["long text", "x", "medium"]

        vectors = embedder.encode(texts)

        self.assertEqual(fake_model.received_texts, ["x", "medium", "long text"])
        self.assertEqual(vectors[:, 0].tolist(), [9.0, 1.0, 6.0])
        # The second column originates from sorted inference order. Its restored
        # placement proves downstream callers still receive source-order rows.
        self.assertEqual(vectors[:, 1].tolist(), [2.0, 0.0, 1.0])

    def test_sorting_can_be_disabled_for_diagnostics(self):
        embedder = QwenSentenceEmbedder("fake", batch_size=2, sort_by_length=False)
        fake_model = _FakeSentenceTransformer()
        embedder._model = fake_model

        vectors = embedder.encode(["long text", "x"])

        self.assertEqual(fake_model.received_texts, ["long text", "x"])
        self.assertEqual(vectors[:, 0].tolist(), [9.0, 1.0])


if __name__ == "__main__":
    unittest.main()
