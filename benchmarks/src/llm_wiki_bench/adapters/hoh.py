"""HoH adapter for the released Parquet table.

Source: https://huggingface.co/datasets/russwest404/HoH-QAs  (Apache-2.0)
File: hoh_qas_240601_241201.parquet

HoH measures whether a system distinguishes current evidence from the outdated
variants in `outdated_infos`; it is not multi-hop QA. Reproducing an outdated
answer is scored as a distractor failure rather than as being merely wrong.

The release carries no record identifier, so the case id combines
`document.id` with the record number. That makes identifiers stable only for a
fixed file, which is why the manifest records the source digest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BenchmarkAdapter

_CONSUMED = {"question", "answer", "evidence", "outdated_infos", "document"}


class HoHAdapter(BenchmarkAdapter):
    name = "hoh"
    profile = "temporal_discrimination"
    container = "parquet"
    evidence_id_origin = "synthesized"

    def normalize(
        self, record: dict[str, Any], path: Path, record_number: int, split: str
    ) -> dict[str, Any]:
        document = self._required(record, "document", path, record_number)
        if not isinstance(document, dict) or not str(document.get("id") or "").strip():
            raise ValueError(f"{path}: record {record_number}: document.id is required")
        outdated = self._required(record, "outdated_infos", path, record_number)
        if not isinstance(outdated, list) or not outdated:
            raise ValueError(f"{path}: record {record_number}: outdated_infos must be a non-empty list")

        distractors: list[str] = []
        outdated_evidence: list[str] = []
        for index, entry in enumerate(outdated):
            if not isinstance(entry, dict):
                raise ValueError(f"{path}: record {record_number}: outdated_infos[{index}] is not an object")
            distractors.append(str(self._required(entry, "answer", path, record_number)))
            outdated_evidence.append(str(self._required(entry, "evidence", path, record_number)))

        metadata = self._metadata(record, _CONSUMED)
        metadata.update(
            {
                "document_id": str(document["id"]),
                "document_title": document.get("title"),
                "last_modified_time": str(record.get("last_modified_time")),
                "outdated_count": len(outdated),
            }
        )
        metadata["source_fields"] = {
            key: str(value) for key, value in metadata.get("source_fields", {}).items()
        }
        return {
            "id": f"{document['id']}:{record_number}",
            "prompt": self._required(record, "question", path, record_number),
            "context": (self._required(record, "evidence", path, record_number), *outdated_evidence),
            "labels": {
                "answers": (self._required(record, "answer", path, record_number),),
                "distractor_answers": tuple(distractors),
            },
            "metadata": metadata,
        }
