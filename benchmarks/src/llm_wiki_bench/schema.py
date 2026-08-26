"""Immutable, normalized records shared by benchmark adapters and runners."""

from copy import deepcopy
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .profiles import capability_requirements, get_profile


DATASETS = frozenset(
    {
        "longmemeval",
        "hoh",
        "vitaminc",
        "factlens",
        "rgb_base",
        "rgb_integration",
        "rgb_counterfactual",
    }
)


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
    profile: str
    prompt: str
    labels: Mapping[str, Any]
    context: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    fine_evidence_ids: tuple[str, ...] = ()
    expects_abstention: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.labels, dict):
            raise ValueError("labels must be a dictionary")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary")
        object.__setattr__(self, "context", _normalize_strings(self.context, "context"))
        object.__setattr__(self, "evidence_ids", _normalize_strings(self.evidence_ids, "evidence_ids"))
        object.__setattr__(
            self, "fine_evidence_ids", _normalize_strings(self.fine_evidence_ids, "fine_evidence_ids")
        )
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
    sub_claim_labels: tuple[str, ...] = ()
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
        object.__setattr__(
            self,
            "sub_claim_labels",
            _normalize_strings(self.sub_claim_labels, "sub_claim_labels"),
        )
        validate_prediction(self)


def validate_case(case: BenchmarkCase) -> None:
    _require_nonblank(case.id, "id")
    _require_nonblank(case.dataset, "dataset")
    if case.dataset not in DATASETS:
        raise ValueError(f"dataset must be one of: {', '.join(sorted(DATASETS))}")
    _require_nonblank(case.split, "split")
    _require_nonblank(case.prompt, "prompt")
    if not isinstance(case.expects_abstention, bool):
        raise ValueError("expects_abstention must be a boolean")
    if len(case.evidence_ids) != len(set(case.evidence_ids)):
        raise ValueError("evidence_ids must not contain duplicates")
    if len(case.fine_evidence_ids) != len(set(case.fine_evidence_ids)):
        raise ValueError("fine_evidence_ids must not contain duplicates")
    _require_declared_capabilities(case)
    _require_nonblank_label_strings(case)


_LABEL_STRING_LIST_KEYS = ("answers", "distractor_answers")


def _require_nonblank_label_strings(case: BenchmarkCase) -> None:
    """Reject a blank gold string so a data hole cannot masquerade as a model failure.

    A blank answer, distractor answer, or slot alias would normalize cleanly
    and then score as if the model answered wrongly, when the defect is in the
    source data. Checked at the schema level so every adapter inherits it.
    """
    for key in _LABEL_STRING_LIST_KEYS:
        value = case.labels.get(key)
        if value is None:
            continue
        if any(not isinstance(item, str) or not item.strip() for item in value):
            raise ValueError(f"case {case.id}: labels.{key} must contain non-blank strings")
    slots = case.labels.get("answer_slots")
    if slots is not None:
        for slot in slots:
            if any(not isinstance(item, str) or not item.strip() for item in slot):
                raise ValueError(f"case {case.id}: labels.answer_slots must contain non-blank strings")


def _require_declared_capabilities(case: BenchmarkCase) -> None:
    """Fail when a profile declares a capability the case has no data for.

    Scoring must never quietly drop a metric because a field was empty; that is
    how a partial run comes to look like a complete one.
    """
    profile = get_profile(case.profile)
    for capability in sorted(profile.capabilities):
        for requirement in capability_requirements(capability):
            if not _has_requirement(case, requirement):
                raise ValueError(f"{profile.name} requires {requirement} on case {case.id}")


def _has_requirement(case: BenchmarkCase, requirement: str) -> bool:
    if requirement.startswith("labels."):
        value = case.labels.get(requirement.removeprefix("labels."))
        return value is not None and (not isinstance(value, (str, tuple, list)) or len(value) > 0)
    value = getattr(case, requirement)
    if isinstance(value, bool):
        return True
    return bool(value)


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
