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


def parse_answer_slots(value: Any, path: Path, record_number: int) -> tuple[tuple[str, ...], ...]:
    """Parse a `list[list[str]]` answer field into one tuple of aliases per slot.

    Each outer element is a genuinely distinct required slot; a bare string
    element is that slot's single alias. Shared by `rgb_base` (which uses
    this only when the record is actually multi-slot; a single slot still
    flattens through `flatten_answers`) and `rgb_integration` (always
    multi-slot).
    """
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path}: record {record_number}: answer must be a non-empty list")
    slots: list[tuple[str, ...]] = []
    for item in value:
        if isinstance(item, str):
            slots.append((item,))
        elif isinstance(item, list) and item and all(isinstance(alias, str) for alias in item):
            slots.append(tuple(item))
        else:
            raise ValueError(
                f"{path}: record {record_number}: each answer slot must be a string or a non-empty list of strings"
            )
    return tuple(slots)


def is_multi_slot_answer(value: Any) -> bool:
    """True when `value` is genuinely more than one required answer slot.

    `en`/`en_refine` encode a single slot's aliases as one inner list, e.g.
    `[["Lisbon", "Lisbon, Portugal"]]` -- a bare list of strings, one slot.
    Only a list of two-or-more nested lists (`[["A"], ["B"]]`) is a case where
    a prediction must satisfy every slot, not just alias-match one answer, so
    only that shape routes to the `multi_slot_retrieval_qa` profile. Anything
    else (a bare string, a flat list of string aliases, or a single-element
    outer list) is one slot and keeps flattening through `flatten_answers`.
    """
    return isinstance(value, list) and len(value) > 1 and all(isinstance(item, list) for item in value)


def synthesize_pool_ids(
    case_id: str, positive: list[str], negative: list[str]
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Return (context, positive_ids, all_candidate_ids) in a single stable order."""
    positive_ids = tuple(f"{case_id}:positive:{index}" for index in range(len(positive)))
    negative_ids = tuple(f"{case_id}:negative:{index}" for index in range(len(negative)))
    return tuple(positive) + tuple(negative), positive_ids, positive_ids + negative_ids


def require_documents(
    record: dict[str, Any], key: str, path: Path, record_number: int, *, required: bool
) -> list[str]:
    """Return record[key] as a list of strings; caller declares whether it must be non-empty.

    `required` has no default: every call site must state its own requirement,
    so a caller who needs non-empty validation for a new key cannot forget the
    flag and silently get pass-through tolerance instead.
    """
    value = record.get(key)
    if required and (not isinstance(value, list) or not value):
        raise ValueError(f"{path}: record {record_number}: {key} must be a non-empty list")
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
        positive = require_documents(record, "positive", path, record_number, required=True)
        negative = require_documents(record, "negative", path, record_number, required=False)
        context, positive_ids, candidate_ids = synthesize_pool_ids(case_id, positive, negative)
        metadata = self._metadata(record, _CONSUMED)
        metadata["candidate_ids"] = candidate_ids
        answer = self._required(record, "answer", path, record_number)
        case = {
            "id": case_id,
            "prompt": self._required(record, "query", path, record_number),
            "context": context,
            "evidence_ids": positive_ids,
            "metadata": metadata,
        }
        if is_multi_slot_answer(answer):
            case["labels"] = {"answer_slots": parse_answer_slots(answer, path, record_number)}
            case["profile"] = "multi_slot_retrieval_qa"
        else:
            case["labels"] = {"answers": flatten_answers(answer, path, record_number)}
        return case
