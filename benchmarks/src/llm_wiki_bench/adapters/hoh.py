"""HoH JSONL adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BenchmarkAdapter


class HoHAdapter(BenchmarkAdapter):
    name = "hoh"

    def normalize(self, record: dict[str, Any], path: Path, record_number: int) -> dict[str, Any]:
        passages = self._required(record, "passages", path, record_number)
        if not isinstance(passages, list) or any(not isinstance(passage, dict) for passage in passages):
            raise ValueError(f"{path}: record {record_number}: passages must be a list of objects")
        consumed = {"id", "split", "source_version", "question", "passages", "answers", "evidence_ids"}
        return {
            "id": self._required(record, "id", path, record_number),
            "split": self._required(record, "split", path, record_number),
            "task": "multi_hop_qa",
            "prompt": self._required(record, "question", path, record_number),
            "context": [self._required(passage, "text", path, record_number) for passage in passages],
            "evidence_ids": self._required(record, "evidence_ids", path, record_number),
            "labels": {"answers": self._required(record, "answers", path, record_number)},
            "metadata": self._metadata(record, consumed),
        }
