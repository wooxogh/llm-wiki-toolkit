"""Optional FactLens JSONL adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BenchmarkAdapter


_VERDICTS = {"SUPPORTED": "supported", "CONTRADICTED": "contradicted", "UNSUPPORTED": "unsupported"}


class FactLensAdapter(BenchmarkAdapter):
    name = "factlens"
    required = False

    def normalize(self, record: dict[str, Any], path: Path, record_number: int) -> dict[str, Any]:
        verdict = self._required(record, "verdict", path, record_number)
        if verdict not in _VERDICTS:
            raise ValueError(f"{path}: record {record_number}: unsupported FactLens verdict {verdict!r}")
        consumed = {"id", "split", "source_version", "claim", "source", "source_id", "verdict"}
        return {
            "id": self._required(record, "id", path, record_number),
            "split": self._required(record, "split", path, record_number),
            "task": "factual_consistency",
            "prompt": self._required(record, "claim", path, record_number),
            "context": (self._required(record, "source", path, record_number),),
            "evidence_ids": (self._required(record, "source_id", path, record_number),),
            "labels": {"label": _VERDICTS[verdict]},
            "metadata": self._metadata(record, consumed),
        }
