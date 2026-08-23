"""VitaminC JSONL adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BenchmarkAdapter


_LABELS = {"SUPPORTS": "entailment", "REFUTES": "contradiction", "NOT_ENOUGH_INFO": "neutral"}


class VitaminCAdapter(BenchmarkAdapter):
    name = "vitaminc"

    def normalize(self, record: dict[str, Any], path: Path, record_number: int) -> dict[str, Any]:
        label = self._required(record, "label", path, record_number)
        if label not in _LABELS:
            raise ValueError(f"{path}: record {record_number}: unsupported VitaminC label {label!r}")
        consumed = {"id", "split", "source_version", "claim", "evidence", "evidence_id", "label"}
        return {
            "id": self._required(record, "id", path, record_number),
            "split": self._required(record, "split", path, record_number),
            "task": "verification",
            "prompt": self._required(record, "claim", path, record_number),
            "context": (self._required(record, "evidence", path, record_number),),
            "evidence_ids": (self._required(record, "evidence_id", path, record_number),),
            "labels": {"label": _LABELS[label]},
            "metadata": self._metadata(record, consumed),
        }
