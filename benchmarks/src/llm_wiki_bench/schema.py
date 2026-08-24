"""Immutable, normalized records shared by benchmark adapters and runners."""

from copy import deepcopy
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any, Iterable, Mapping


DATASETS = frozenset({"longmemeval", "hoh", "vitaminc", "rgb", "factlens"})


def _freeze(value: Any) -> Any:
    if isinstance(value, MappingABC):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    if isinstance(value, bytearray):
        return bytes(value)
    return deepcopy(value)


def _normalize_strings(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError(f"{field_name} must be a sequence of strings")
    try:
        normalized = tuple(values)
    except TypeError as error:
        raise ValueError(f"{field_name} must be a sequence of strings") from error
    if any(not isinstance(value, str) or not value.strip() for value in normalized):
        raise ValueError(f"{field_name} must contain non-blank strings")
    return normalized


def _require_nonblank(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank string")


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    dataset: str
    split: str
    task: str
    prompt: str
    labels: Mapping[str, Any]
    context: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.labels, dict):
            raise ValueError("labels must be a dictionary")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary")
        object.__setattr__(self, "context", _normalize_strings(self.context, "context"))
        object.__setattr__(self, "evidence_ids", _normalize_strings(self.evidence_ids, "evidence_ids"))
        object.__setattr__(self, "labels", _freeze(self.labels))
        object.__setattr__(self, "metadata", _freeze(self.metadata))
        validate_case(self)


@dataclass(frozen=True)
class Prediction:
    case_id: str
    answer: str | None = None
    label: str | None = None
    ranked_evidence_ids: tuple[str, ...] = ()
    cited_evidence_ids: tuple[str, ...] = ()
    abstained: bool = False
    latency_ms: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "ranked_evidence_ids",
            _normalize_strings(self.ranked_evidence_ids, "ranked_evidence_ids"),
        )
        object.__setattr__(
            self,
            "cited_evidence_ids",
            _normalize_strings(self.cited_evidence_ids, "cited_evidence_ids"),
        )
        validate_prediction(self)


def validate_case(case: BenchmarkCase) -> None:
    _require_nonblank(case.id, "id")
    _require_nonblank(case.dataset, "dataset")
    if case.dataset not in DATASETS:
        raise ValueError(f"dataset must be one of: {', '.join(sorted(DATASETS))}")
    _require_nonblank(case.split, "split")
    _require_nonblank(case.task, "task")
    _require_nonblank(case.prompt, "prompt")
    if len(case.evidence_ids) != len(set(case.evidence_ids)):
        raise ValueError("evidence_ids must not contain duplicates")


def validate_prediction(prediction: Prediction) -> None:
    _require_nonblank(prediction.case_id, "case_id")
    if prediction.answer is not None and not isinstance(prediction.answer, str):
        raise ValueError("answer must be a string or None")
    if prediction.label is not None and not isinstance(prediction.label, str):
        raise ValueError("label must be a string or None")
    if not isinstance(prediction.abstained, bool):
        raise ValueError("abstained must be a boolean")
    if prediction.abstained and prediction.answer is not None:
        raise ValueError("an abstained prediction cannot include an answer")
    if prediction.latency_ms is not None:
        if (
            isinstance(prediction.latency_ms, bool)
            or not isinstance(prediction.latency_ms, (int, float))
            or not isfinite(prediction.latency_ms)
            or prediction.latency_ms < 0
        ):
            raise ValueError("latency_ms must be non-negative")
    if len(prediction.ranked_evidence_ids) != len(set(prediction.ranked_evidence_ids)):
        raise ValueError("ranked_evidence_ids must not contain duplicates")
    if len(prediction.cited_evidence_ids) != len(set(prediction.cited_evidence_ids)):
        raise ValueError("cited_evidence_ids must not contain duplicates")
