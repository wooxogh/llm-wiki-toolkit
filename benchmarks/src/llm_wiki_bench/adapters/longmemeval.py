"""LongMemEval JSONL adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BenchmarkAdapter


class LongMemEvalAdapter(BenchmarkAdapter):
    name = "longmemeval"

    def normalize(self, record: dict[str, Any], path: Path, record_number: int) -> dict[str, Any]:
        consumed = {"id", "split", "source_version", "question", "sessions", "answers", "evidence_ids"}
        return {
            "id": self._required(record, "id", path, record_number),
            "split": self._required(record, "split", path, record_number),
            "task": "memory_qa",
            "prompt": self._required(record, "question", path, record_number),
            "context": self._required(record, "sessions", path, record_number),
            "evidence_ids": self._required(record, "evidence_ids", path, record_number),
            "labels": {"answers": self._required(record, "answers", path, record_number)},
            "metadata": self._metadata(record, consumed),
        }
