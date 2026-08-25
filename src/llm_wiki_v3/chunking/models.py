"""Plain data models used by the hybrid chunker and its diagnostics."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class StructuralBlock:
    index: int
    kind: str
    text: str
    source_start: int
    source_end: int
    heading_level: int | None
    heading_path: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["heading_path"] = list(self.heading_path)
        return payload


@dataclass(frozen=True)
class SentenceSpan:
    text: str
    embedding_text: str
    source_start: int
    source_end: int


@dataclass(frozen=True)
class BoundaryDecision:
    block_index: int
    right_block_index: int | None
    gap_index: int
    left_sentence: str
    right_sentence: str
    contextual_cosine: float
    local_valley: bool
    candidate_rank: int | None
    is_candidate: bool
    boundary_probability: float | None
    keep_boundary: bool
    effective_threshold: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    source_path: str
    ordinal: int
    kind: str
    text: str
    source_text: str
    source_start: int
    source_end: int
    heading_path: tuple[str, ...]
    chunk_heading: str | None
    previous_chunk_id: str | None
    next_chunk_id: str | None
    document_created_at: str | None
    document_modified_at: str | None
    sentence_count: int
    semantic_refined: bool
    content_hash: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["heading_path"] = list(self.heading_path)
        return payload


@dataclass(frozen=True)
class ChunkingResult:
    document_id: str
    source_path: str
    chunks: tuple[Chunk, ...]
    structural_blocks: tuple[StructuralBlock, ...]
    boundary_decisions: tuple[BoundaryDecision, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        candidates = [item for item in self.boundary_decisions if item.is_candidate]
        kept = [item for item in candidates if item.keep_boundary]
        refined_blocks = {item.block_index for item in self.boundary_decisions}
        refined_blocks.update(
            item.right_block_index
            for item in self.boundary_decisions
            if item.right_block_index is not None
        )
        return {
            "document_id": self.document_id,
            "source_path": self.source_path,
            "structural_block_count": len(self.structural_blocks),
            "semantic_refined_block_count": len(refined_blocks),
            "semantic_gap_count": len(self.boundary_decisions),
            "candidate_count": len(candidates),
            "kept_semantic_boundary_count": len(kept),
            "final_chunk_count": len(self.chunks),
            **self.metadata,
        }
