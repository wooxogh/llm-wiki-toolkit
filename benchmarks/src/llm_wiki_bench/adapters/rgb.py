"""RGB JSONL adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BenchmarkAdapter


class RGBAdapter(BenchmarkAdapter):
    name = "rgb"

    def normalize(self, record: dict[str, Any], path: Path, record_number: int) -> dict[str, Any]:
        documents = self._required(record, "documents", path, record_number)
        if not isinstance(documents, list) or any(not isinstance(document, dict) for document in documents):
            raise ValueError(f"{path}: record {record_number}: documents must be a list of objects")
        consumed = {"id", "split", "source_version", "query", "documents", "answers", "supporting_ids"}
        return {
            "id": self._required(record, "id", path, record_number),
            "split": self._required(record, "split", path, record_number),
            "task": "rag_qa",
            "prompt": self._required(record, "query", path, record_number),
            "context": [self._required(document, "text", path, record_number) for document in documents],
            "evidence_ids": self._required(record, "supporting_ids", path, record_number),
            "labels": {"answers": self._required(record, "answers", path, record_number)},
            "metadata": self._metadata(record, consumed),
        }
