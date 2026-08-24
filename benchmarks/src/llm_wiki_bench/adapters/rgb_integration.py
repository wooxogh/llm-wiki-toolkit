"""RGB information-integration adapter for en_int.json.

Source: https://github.com/chen700564/RGB  (no license declaration)

This variant does not share the base schema: answer carries one slot per
sub-question and positive is grouped per sub-question. The record also carries
an upstream field-name typo, `asnwer1`, which is preserved verbatim rather than
corrected, so the normalized record stays traceable to the release.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BenchmarkAdapter
from .rgb_base import require_documents, synthesize_pool_ids

_CONSUMED = {"id", "query", "answer", "positive", "negative"}


class RGBIntegrationAdapter(BenchmarkAdapter):
    name = "rgb_integration"
    profile = "multi_slot_retrieval_qa"
    container = "jsonl"
    evidence_id_origin = "synthesized"

    def normalize(
        self, record: dict[str, Any], path: Path, record_number: int, split: str
    ) -> dict[str, Any]:
        case_id = str(self._required(record, "id", path, record_number))
        slots = self._answer_slots(record, path, record_number)
        positive = self._flatten_positive(record, path, record_number)
        negative = require_documents(record, "negative", path, record_number, required=False)
        context, positive_ids, candidate_ids = synthesize_pool_ids(case_id, positive, negative)
        metadata = self._metadata(record, _CONSUMED)
        metadata["candidate_ids"] = candidate_ids
        return {
            "id": case_id,
            "prompt": self._required(record, "query", path, record_number),
            "context": context,
            "evidence_ids": positive_ids,
            "labels": {"answer_slots": slots},
            "metadata": metadata,
        }

    def _answer_slots(
        self, record: dict[str, Any], path: Path, record_number: int
    ) -> tuple[tuple[str, ...], ...]:
        value = self._required(record, "answer", path, record_number)
        if not isinstance(value, list) or len(value) < 2:
            raise ValueError(f"{path}: record {record_number}: answer must declare at least two slots")
        slots: list[tuple[str, ...]] = []
        for slot in value:
            if isinstance(slot, str):
                slots.append((slot,))
            elif isinstance(slot, list) and slot and all(isinstance(alias, str) for alias in slot):
                slots.append(tuple(slot))
            else:
                raise ValueError(
                    f"{path}: record {record_number}: each answer slot must be a string or a non-empty list of strings"
                )
        return tuple(slots)

    def _flatten_positive(self, record: dict[str, Any], path: Path, record_number: int) -> list[str]:
        value = record.get("positive")
        if not isinstance(value, list) or not value:
            raise ValueError(f"{path}: record {record_number}: positive must be a non-empty list")
        if all(isinstance(item, str) for item in value):
            return list(value)
        documents: list[str] = []
        for group in value:
            if not isinstance(group, list) or any(not isinstance(item, str) for item in group):
                raise ValueError(
                    f"{path}: record {record_number}: positive must be strings or lists of strings"
                )
            documents.extend(group)
        if not documents:
            raise ValueError(f"{path}: record {record_number}: positive must be a non-empty list")
        return documents
