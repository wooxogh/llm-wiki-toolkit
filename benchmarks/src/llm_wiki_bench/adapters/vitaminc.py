"""VitaminC adapter for the released JSONL.

Source: https://huggingface.co/datasets/tals/vitaminc  (CC-BY-SA-3.0)
Files: train.jsonl, dev.jsonl, test.jsonl

Evidence is supplied inline and there is no retrievable corpus, so this suite
scores a label and nothing else. The released label spelling uses spaces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BenchmarkAdapter

_LABELS = {
    "SUPPORTS": "entailment",
    "REFUTES": "contradiction",
    "NOT ENOUGH INFO": "neutral",
}

_CONSUMED = {
    "unique_id",
    "claim",
    "evidence",
    "label",
    "case_id",
    "page",
    "revision_type",
    "wiki_revision_id",
}


class VitaminCAdapter(BenchmarkAdapter):
    name = "vitaminc"
    profile = "grounded_verification"
    container = "jsonl"
    evidence_id_origin = "upstream"

    def normalize(
        self, record: dict[str, Any], path: Path, record_number: int, split: str
    ) -> dict[str, Any]:
        label = self._required(record, "label", path, record_number)
        if label not in _LABELS:
            raise ValueError(
                f"{path}: record {record_number}: unsupported VitaminC label {label!r}; "
                f"expected one of {sorted(_LABELS)}"
            )
        metadata = self._metadata(record, _CONSUMED)
        metadata.update(
            {
                "case_id": record.get("case_id"),
                "page": record.get("page"),
                "revision_type": record.get("revision_type"),
                "wiki_revision_id": record.get("wiki_revision_id"),
            }
        )
        return {
            "id": str(self._required(record, "unique_id", path, record_number)),
            "prompt": self._required(record, "claim", path, record_number),
            "context": (self._required(record, "evidence", path, record_number),),
            "labels": {"label": _LABELS[label]},
            "metadata": metadata,
        }
