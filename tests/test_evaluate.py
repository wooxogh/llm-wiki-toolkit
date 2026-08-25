from __future__ import annotations

import unittest

from llm_wiki_v3.evaluate import _metrics, validate_gold


class EvaluateTests(unittest.TestCase):
    def test_gold_contract_and_metrics(self):
        gold = validate_gold([{"query": "q", "relevant_chunk_ids": ["a"]}])
        metrics = _metrics([(gold[0], [{"id": "a", "document_id": "doc"}])], 5)
        self.assertEqual(metrics["hit_at_5"], 1.0)
        self.assertEqual(metrics["mrr"], 1.0)

    def test_empty_gold_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_gold([])

    def test_multiple_chunks_from_one_relevant_document_count_once(self):
        gold = {"query": "q", "relevant_document_ids": ["doc"]}
        metrics = _metrics([(gold, [
            {"id": "a", "document_id": "doc"},
            {"id": "b", "document_id": "doc"},
        ])], 5)
        self.assertEqual(metrics["recall_at_5"], 1.0)
        self.assertEqual(metrics["ndcg_at_5"], 1.0)


if __name__ == "__main__":
    unittest.main()
