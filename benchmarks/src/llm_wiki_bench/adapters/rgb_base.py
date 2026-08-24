"""RGB base adapter for en.json and en_refine.json.

Source: https://github.com/chen700564/RGB  (no license declaration)
en_refine is the corrected edition of en and shares its schema. Files are JSON
Lines despite the .json extension.

RGB supplies document pools, not an assembled context. The upstream harness
builds a context from them at a chosen noise_rate/passage_num; those are
parameters of the prediction step, so this adapter emits both pools with
synthesized identifiers and leaves assembly to whatever produced the
predictions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BenchmarkAdapter

_CONSUMED = {"id", "query", "answer", "positive", "negative"}


def flatten_answers(value: Any, path: Path, record_number: int) -> tuple[str, ...]:
    """Accept a string, a list of strings, or slots of aliases.

    en and en_int use list[list[str]]; en_fact uses a bare string.
    """
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path}: record {record_number}: answer must be a string or a non-empty list")
    answers: list[str] = []
    for item in value:
        if isinstance(item, str):
            answers.append(item)
        elif isinstance(item, list) and all(isinstance(alias, str) for alias in item):
            answers.extend(item)
        else:
            raise ValueError(
                f"{path}: record {record_number}: answer entries must be strings or lists of strings"
            )
    if not answers:
        raise ValueError(f"{path}: record {record_number}: answer must contain at least one string")
    return tuple(answers)


def synthesize_pool_ids(
    case_id: str, positive: list[str], negative: list[str]
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Return (context, positive_ids, all_candidate_ids) in a single stable order."""
    positive_ids = tuple(f"{case_id}:positive:{index}" for index in range(len(positive)))
    negative_ids = tuple(f"{case_id}:negative:{index}" for index in range(len(negative)))
    return tuple(positive) + tuple(negative), positive_ids, positive_ids + negative_ids


def require_documents(record: dict[str, Any], key: str, path: Path, record_number: int) -> list[str]:
    value = record.get(key)
    if key == "positive" and (not isinstance(value, list) or not value):
        raise ValueError(f"{path}: record {record_number}: positive must be a non-empty list")
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{path}: record {record_number}: {key} must be a list of strings")
    return value


class RGBBaseAdapter(BenchmarkAdapter):
    name = "rgb_base"
    profile = "retrieval_qa"
    container = "jsonl"
    evidence_id_origin = "synthesized"

    def normalize(
        self, record: dict[str, Any], path: Path, record_number: int, split: str
    ) -> dict[str, Any]:
        case_id = str(self._required(record, "id", path, record_number))
        positive = require_documents(record, "positive", path, record_number)
        negative = require_documents(record, "negative", path, record_number)
        context, positive_ids, candidate_ids = synthesize_pool_ids(case_id, positive, negative)
        metadata = self._metadata(record, _CONSUMED)
        metadata["candidate_ids"] = candidate_ids
        return {
            "id": case_id,
            "prompt": self._required(record, "query", path, record_number),
            "context": context,
            "evidence_ids": positive_ids,
            "labels": {
                "answers": flatten_answers(
                    self._required(record, "answer", path, record_number), path, record_number
                )
            },
            "metadata": metadata,
        }
