"""Adapter base: declare a container, map fields, return provenance.

Provenance is derived, never asserted by the record. No released dataset
carries a per-record version or split, so `load` takes the split from
configuration and computes the digest from the bytes it actually read.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_wiki_bench.readers import READERS, file_digest
from llm_wiki_bench.schema import BenchmarkCase


@dataclass(frozen=True)
class LoadResult:
    cases: tuple[BenchmarkCase, ...]
    content_digest: str
    record_count: int


class BenchmarkAdapter(ABC):
    """Translate one released source format into normalized cases."""

    name: str
    profile: str
    container: str
    evidence_id_origin: str
    required: bool = True

    def load(self, path: Path, *, split: str) -> LoadResult:
        if not isinstance(split, str) or not split.strip():
            raise ValueError("split must be a non-blank string")
        try:
            reader = READERS[self.container]
        except KeyError as error:
            raise ValueError(f"unknown container: {self.container}") from error
        source = Path(path)
        cases: list[BenchmarkCase] = []
        record_count = 0
        for record_number, record in reader(source):
            record_count += 1
            values = self.normalize(record, source, record_number, split)
            metadata = dict(values.pop("metadata", {}))
            metadata.update({"source_path": str(source), "source_record": record_number})
            cases.append(
                BenchmarkCase(
                    dataset=self.name,
                    split=split,
                    profile=self.profile,
                    metadata=metadata,
                    **values,
                )
            )
        return LoadResult(tuple(cases), file_digest(source), record_count)

    @abstractmethod
    def normalize(
        self, record: dict[str, Any], path: Path, record_number: int, split: str
    ) -> dict[str, Any]:
        """Return BenchmarkCase fields other than dataset, split, and profile."""

    @staticmethod
    def _required(record: dict[str, Any], key: str, path: Path, record_number: int) -> Any:
        if key not in record:
            raise ValueError(f"{path}: record {record_number}: missing {key}")
        return record[key]

    @staticmethod
    def _metadata(record: dict[str, Any], consumed: set[str]) -> dict[str, Any]:
        return {"source_fields": {key: value for key, value in record.items() if key not in consumed}}
