"""Shared JSONL loading and provenance handling for source adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any

from llm_wiki_bench.schema import BenchmarkCase


class BenchmarkAdapter(ABC):
    """Translate one source suite's JSONL contract into normalized cases."""

    name: str
    required: bool = True

    def load(self, path: Path, split: str | None = None) -> list[BenchmarkCase]:
        """Load JSONL records, retaining their origin in every normalized case."""
        cases: list[BenchmarkCase] = []
        source_path = Path(path)
        with source_path.open(encoding="utf-8") as source:
            for record_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                record = self._read_record(line, source_path, record_number)
                if split is not None and record.get("split") != split:
                    continue
                values = self.normalize(record, source_path, record_number)
                metadata = dict(values.pop("metadata", {}))
                metadata.update(
                    {
                        "source_path": str(source_path),
                        "source_record": record_number,
                        "source_version": self._source_version(record, source_path, record_number),
                    }
                )
                cases.append(BenchmarkCase(dataset=self.name, metadata=metadata, **values))
        return cases

    @abstractmethod
    def normalize(self, record: dict[str, Any], source_path: Path, record_number: int) -> dict[str, Any]:
        """Return ``BenchmarkCase`` fields excluding dataset and provenance metadata."""

    def _read_record(self, line: str, path: Path, record_number: int) -> dict[str, Any]:
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}: record {record_number}: invalid JSON") from error
        if not isinstance(record, dict):
            raise ValueError(f"{path}: record {record_number}: expected a JSON object")
        return record

    def _source_version(self, record: dict[str, Any], path: Path, record_number: int) -> str:
        version = record.get("source_version")
        if not isinstance(version, str) or not version.strip():
            raise ValueError(f"{path}: record {record_number}: source_version must be a non-blank string")
        return version

    @staticmethod
    def _required(record: dict[str, Any], key: str, path: Path, record_number: int) -> Any:
        if key not in record:
            raise ValueError(f"{path}: record {record_number}: missing {key}")
        return record[key]

    @staticmethod
    def _metadata(record: dict[str, Any], consumed: set[str]) -> dict[str, Any]:
        return {"source_fields": {key: value for key, value in record.items() if key not in consumed}}
