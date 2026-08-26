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

from llm_wiki_bench.profiles import get_profile
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
            cases.append(build_case(self, record, source, record_number, split))
        return LoadResult(tuple(cases), file_digest(source), record_count)

    @abstractmethod
    def normalize(
        self, record: dict[str, Any], path: Path, record_number: int, split: str
    ) -> dict[str, Any]:
        """Return BenchmarkCase fields other than dataset and split.

        May include a `profile` key to select a profile other than the
        class attribute `self.profile` for this record (must be a
        registered profile); when omitted, `load` uses `self.profile`.
        """

    @staticmethod
    def _required(record: dict[str, Any], key: str, path: Path, record_number: int) -> Any:
        if key not in record:
            raise ValueError(f"{path}: record {record_number}: missing {key}")
        return record[key]

    @staticmethod
    def _metadata(record: dict[str, Any], consumed: set[str]) -> dict[str, Any]:
        return {"source_fields": {key: value for key, value in record.items() if key not in consumed}}


def build_case(
    adapter: BenchmarkAdapter,
    record: dict[str, Any],
    source: Path,
    record_number: int,
    split: str,
) -> BenchmarkCase:
    """Normalize one record into a `BenchmarkCase`, injecting derived provenance.

    The single construction path shared by `BenchmarkAdapter.load` and
    `runner.check_conformance`, so a step added to one (the blank-string
    check schema-level validation performs, for instance) can never be
    missing from the other -- the two must never diverge on how a case gets
    built from a record.
    """
    values = adapter.normalize(record, source, record_number, split)
    metadata = dict(values.pop("metadata", {}))
    metadata.update({"source_path": str(source), "source_record": record_number})
    if "profile" in values:
        profile = values.pop("profile")
        try:
            get_profile(profile)
        except ValueError as error:
            raise ValueError(f"{source}: record {record_number}: {error}") from error
    else:
        profile = adapter.profile
    return BenchmarkCase(
        dataset=adapter.name,
        split=split,
        profile=profile,
        metadata=metadata,
        **values,
    )
