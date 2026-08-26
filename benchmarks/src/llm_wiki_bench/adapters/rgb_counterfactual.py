"""RGB counterfactual-robustness adapter for en_fact.json.

Source: https://github.com/chen700564/RGB  (no license declaration)

This variant carries an `answer` (string or list), a `fakeanswer`, and
`positive_wrong` documents that support the fake answer. Reproducing the fake
answer is a distinct failure from being merely wrong, so the fake answer is
scored as a distractor rather than folded into the accepted answers.
`positive_wrong` documents are emitted as candidates but never counted as
expected evidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BenchmarkAdapter
from .rgb_base import flatten_answers, require_documents

_CONSUMED = {"id", "query", "answer", "fakeanswer", "positive", "positive_wrong", "negative"}


class RGBCounterfactualAdapter(BenchmarkAdapter):
    name = "rgb_counterfactual"
    profile = "counterfactual_qa"
    container = "jsonl"
    evidence_id_origin = "synthesized"

    def normalize(
        self, record: dict[str, Any], path: Path, record_number: int, split: str
    ) -> dict[str, Any]:
        case_id = str(self._required(record, "id", path, record_number))
        positive = require_documents(record, "positive", path, record_number, required=True)
        negative = require_documents(record, "negative", path, record_number, required=False)
        positive_wrong = require_documents(
            record, "positive_wrong", path, record_number, required=True
        )

        positive_ids = tuple(f"{case_id}:positive:{index}" for index in range(len(positive)))
        negative_ids = tuple(f"{case_id}:negative:{index}" for index in range(len(negative)))
        wrong_ids = tuple(
            f"{case_id}:positive_wrong:{index}" for index in range(len(positive_wrong))
        )

        metadata = self._metadata(record, _CONSUMED)
        metadata["candidate_ids"] = positive_ids + negative_ids + wrong_ids
        return {
            "id": case_id,
            "prompt": self._required(record, "query", path, record_number),
            "context": tuple(positive) + tuple(negative) + tuple(positive_wrong),
            "evidence_ids": positive_ids,
            "labels": {
                "answers": flatten_answers(
                    self._required(record, "answer", path, record_number), path, record_number
                ),
                "distractor_answers": flatten_answers(
                    self._required(record, "fakeanswer", path, record_number), path, record_number
                ),
            },
            "metadata": metadata,
        }
