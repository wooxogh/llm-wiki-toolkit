"""Serializable contracts shared by indexing, search, and hygiene."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SearchHit:
    chunk: dict[str, Any]
    score: float
    channel_ranks: dict[str, int] = field(default_factory=dict)
    channel_scores: dict[str, float] = field(default_factory=dict)
    rerank_score: float | None = None
    related_evidence: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["score"] = round(float(self.score), 8)
        if self.rerank_score is not None:
            payload["rerank_score"] = round(float(self.rerank_score), 8)
        return payload


@dataclass(frozen=True)
class HealthIssue:
    severity: str
    code: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

