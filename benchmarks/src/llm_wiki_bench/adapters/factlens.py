"""Optional FactLens adapter for the released CSV.

Source: https://github.com/megagonlabs/factlens  (BSD-3-Clause)
File: benchmark/fact_lens_benchmark.csv

The benchmark carries no evidence: it measures decomposition of a complex claim
into sub-claims and the per-sub-claim verdicts. `sub_claims` and `labels` are
Python-repr lists rather than JSON, so they are parsed with ast.literal_eval.

`labels` mixes string ('true'/'false') and bool (True/False) items in the
released file (502 of 733 rows carry at least one bool item) even though
`sub_claims` is strings-only; both forms are canonicalized to the lowercase
strings 'true'/'false'.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from .base import BenchmarkAdapter

_CONSUMED = {"ind", "claim", "sub_claims", "labels", "aggregated_label"}


class FactLensAdapter(BenchmarkAdapter):
    name = "factlens"
    profile = "claim_decomposition"
    container = "csv"
    evidence_id_origin = "upstream"
    required = False

    def normalize(
        self, record: dict[str, Any], path: Path, record_number: int, split: str
    ) -> dict[str, Any]:
        sub_claims = self._repr_list(record, "sub_claims", path, record_number)
        labels = self._label_list(record, path, record_number)
        if len(sub_claims) != len(labels):
            raise ValueError(f"{path}: record {record_number}: sub_claims and labels differ in length")
        if not sub_claims:
            raise ValueError(f"{path}: record {record_number}: sub_claims must not be empty")
        return {
            "id": str(self._required(record, "ind", path, record_number)),
            "prompt": self._required(record, "claim", path, record_number),
            "labels": {
                "sub_claims": sub_claims,
                "sub_claim_labels": labels,
                "aggregated_label": self._boolean(record, path, record_number),
            },
            "metadata": self._metadata(record, _CONSUMED),
        }

    def _parse_repr_list(self, raw: Any, path: Path, record_number: int, key: str) -> list[Any]:
        try:
            value = ast.literal_eval(raw) if isinstance(raw, str) else raw
        except (SyntaxError, ValueError) as error:
            raise ValueError(f"{path}: record {record_number}: {key} is not a list of strings") from error
        if not isinstance(value, list):
            raise ValueError(f"{path}: record {record_number}: {key} is not a list of strings")
        return value

    def _repr_list(
        self, record: dict[str, Any], key: str, path: Path, record_number: int
    ) -> tuple[str, ...]:
        raw = self._required(record, key, path, record_number)
        value = self._parse_repr_list(raw, path, record_number, key)
        if any(not isinstance(item, str) for item in value):
            raise ValueError(f"{path}: record {record_number}: {key} is not a list of strings")
        return tuple(value)

    def _label_list(
        self, record: dict[str, Any], path: Path, record_number: int
    ) -> tuple[str, ...]:
        raw = self._required(record, "labels", path, record_number)
        try:
            value = self._parse_repr_list(raw, path, record_number, "labels")
        except ValueError as error:
            raise ValueError(
                f"{path}: record {record_number}: labels is not a list of true/false values"
            ) from error

        canonical: list[str] = []
        for item in value:
            if isinstance(item, bool):
                canonical.append("true" if item else "false")
                continue
            if isinstance(item, str) and item.strip().casefold() in {"true", "false"}:
                canonical.append(item.strip().casefold())
                continue
            raise ValueError(f"{path}: record {record_number}: labels is not a list of true/false values")
        return tuple(canonical)

    def _boolean(self, record: dict[str, Any], path: Path, record_number: int) -> bool:
        raw = self._required(record, "aggregated_label", path, record_number)
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str) and raw.strip().lower() in {"true", "false"}:
            return raw.strip().lower() == "true"
        raise ValueError(f"{path}: record {record_number}: aggregated_label must be True or False")
