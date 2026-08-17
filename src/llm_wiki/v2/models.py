"""Typed v2 artifact models.

The model layer is deliberately plain dataclasses so tests and offline tools can
mock the LLM seam and inspect artifacts without provider SDKs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

from llm_wiki.v2.schemas import (
    ARTIFACT_SCHEMA_VERSION,
    CHUNK_SCHEMA_VERSION,
    CONCEPT_PROMPT_VERSION,
    CONCEPT_SCHEMA_VERSION,
    ConceptState,
    EdgeType,
    NodeType,
    RelationType,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


T = TypeVar("T")


def _from_dict(cls: type[T], data: dict[str, Any]) -> T:
    names = {f.name for f in fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"{cls.__name__} contains unknown field(s): {', '.join(sorted(unknown))}")
    return cls(**data)


@dataclass(frozen=True)
class Document:
    id: str
    path: str
    content_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_ids: list[str] = field(default_factory=list)
    updated_at: str | None = None
    schema_version: str = ARTIFACT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return _plain(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        return _from_dict(cls, data)


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    path: str
    heading_path: list[str]
    ordinal: int
    text: str
    source_start: int
    source_end: int
    content_hash: str
    target_chars: int
    oversized: bool = False
    schema_version: str = CHUNK_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return _plain(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        return _from_dict(cls, data)


@dataclass(frozen=True)
class Concept:
    id: str
    document_id: str
    chunk_id: str
    text: str
    summary: str
    source_quote: str
    confidence: float
    chunk_hash: str
    source_start: int
    source_end: int
    heading_path: list[str] = field(default_factory=list)
    state: str = ConceptState.ACTIVE.value
    primary_topic_id: str | None = None
    secondary_topic_ids: list[str] = field(default_factory=list)
    schema_version: str = CONCEPT_SCHEMA_VERSION
    prompt_version: str = CONCEPT_PROMPT_VERSION
    created_at: str = field(default_factory=utc_now)
    updated_at: str | None = None
    embedding_ref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _plain(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Concept":
        return _from_dict(cls, data)


@dataclass(frozen=True)
class ConceptProposal:
    text: str
    summary: str
    source_quote: str
    confidence: float


@dataclass(frozen=True)
class PlacementProposal:
    concept_id: str
    primary_topic_id: str | None
    secondary_topic_ids: list[str] = field(default_factory=list)
    create_topic_label: str | None = None
    collection_id: str | None = None
    create_collection_label: str | None = None
    collection_type: str | None = None
    confidence: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class Topic:
    id: str
    name: str
    parent_topic_id: str | None = None
    created_by: str = "ai"
    created_at: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class Collection:
    id: str
    name: str
    parent_topic_id: str | None = None
    collection_type: str | None = None
    document_ids: list[str] = field(default_factory=list)
    created_by: str = "ai"
    created_at: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class RelationProposal:
    id: str
    source_concept_id: str
    target_concept_id: str
    relation: str
    confidence: float
    evidence: str
    status: str = "PROPOSED"
    requires_approval: bool = False
    prompt_version: str = ""
    created_at: str = field(default_factory=utc_now)
    approved_by: str | None = None
    approved_at: str | None = None
    same_subject: bool | None = None
    same_scope: bool | None = None
    temporal_change_possible: bool | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _plain(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RelationProposal":
        return _from_dict(cls, data)


@dataclass(frozen=True)
class NetNode:
    id: str
    type: str
    label: str
    state: str = "ACTIVE"
    created_by: str = "system"
    created_at: str = field(default_factory=utc_now)
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _plain(asdict(self))

    @classmethod
    def topic(cls, label: str, node_id: str | None = None, created_by: str = "ai") -> "NetNode":
        slug = "-".join(label.lower().split()) or "topic"
        return cls(id=node_id or f"topic:{slug}", type=NodeType.TOPIC.value, label=label, created_by=created_by)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NetNode":
        return _from_dict(cls, data)


@dataclass(frozen=True)
class NetEdge:
    id: str
    type: str
    source: str
    target: str
    relation: str | None = None
    confidence: float | None = None
    approved_by: str | None = None
    approved_at: str | None = None
    created_at: str = field(default_factory=utc_now)
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _plain(asdict(self))

    @classmethod
    def relation_edge(cls, proposal: RelationProposal) -> "NetEdge":
        return cls(
            id=f"edge:{proposal.id}",
            type=EdgeType.RELATES_TO.value,
            source=proposal.source_concept_id,
            target=proposal.target_concept_id,
            relation=RelationType(proposal.relation).value,
            confidence=proposal.confidence,
            approved_by=proposal.approved_by,
            approved_at=proposal.approved_at,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NetEdge":
        return _from_dict(cls, data)


@dataclass(frozen=True)
class ReviewItem:
    id: str
    kind: str
    proposal_id: str
    reason: str
    state: str = "OPEN"
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return _plain(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReviewItem":
        return _from_dict(cls, data)


@dataclass(frozen=True)
class Operation:
    id: str
    op: str
    actor: str
    before: dict[str, Any]
    after: dict[str, Any]
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return _plain(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Operation":
        return _from_dict(cls, data)
