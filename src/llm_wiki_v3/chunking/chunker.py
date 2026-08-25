"""Hybrid Markdown chunking: structural blocks first, semantic refinement second."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import ChunkerConfig
from .models import BoundaryDecision, Chunk, ChunkingResult, SentenceSpan, StructuralBlock
from .runtime import QwenSentenceEmbedder, SmallV3BoundaryVerifier
from .semantic import (
    BoundaryVerifier,
    EmbeddingProvider,
    boundary_context,
    contextual_cosine_curve,
    local_valleys,
)
from .structural import normalize_chunk_text, parse_markdown_blocks, sentence_spans


@dataclass(frozen=True)
class _RefinableBlock:
    block: StructuralBlock
    sentences: tuple[SentenceSpan, ...]
    embeddings: np.ndarray


@dataclass(frozen=True)
class _Gap:
    block_index: int
    right_block_index: int | None
    gap_index: int
    cosine: float
    local_valley: bool


class HybridMarkdownChunker:
    """Combine Markdown hard boundaries with the best validated Small-V3 model."""

    def __init__(
        self,
        config: ChunkerConfig | None = None,
        *,
        embedder: EmbeddingProvider | None = None,
        verifier: BoundaryVerifier | None = None,
    ) -> None:
        self.config = config or ChunkerConfig()
        self._embedder = embedder
        self._verifier = verifier

    def chunk_file(self, path: Path, *, document_id: str | None = None) -> ChunkingResult:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Markdown document not found: {resolved}")
        markdown = resolved.read_text(encoding="utf-8")
        created_at, modified_at = _document_timestamps(markdown, resolved)
        return self.chunk_markdown(
            markdown,
            document_id=document_id or resolved.stem,
            source_path=str(resolved),
            document_created_at=created_at,
            document_modified_at=modified_at,
        )

    def chunk_markdown(
        self,
        markdown: str,
        *,
        document_id: str,
        source_path: str = "",
        document_created_at: str | None = None,
        document_modified_at: str | None = None,
    ) -> ChunkingResult:
        blocks = parse_markdown_blocks(markdown)
        refinable_specs: list[tuple[StructuralBlock, tuple[SentenceSpan, ...]]] = []
        for block in blocks:
            spans = tuple(sentence_spans(block)) if block.kind == "paragraph" else ()
            if spans:
                refinable_specs.append((block, spans))

        refinable = self._embed_refinable_blocks(document_id, refinable_specs)
        decisions = self._decide_boundaries(refinable)
        decisions_by_block: dict[int, list[BoundaryDecision]] = {}
        for decision in decisions:
            decisions_by_block.setdefault(decision.block_index, []).append(decision)
        refinable_by_block = {item.block.index: item for item in refinable}

        chunks: list[Chunk] = []
        chunk_memberships: list[set[int]] = []
        for block in blocks:
            if block.kind in {"frontmatter", "heading"}:
                continue
            refined = refinable_by_block.get(block.index)
            if refined is None:
                chunks.append(
                    self._make_chunk(
                        markdown,
                        document_id,
                        source_path,
                        block,
                        block.source_start,
                        block.source_end,
                        sentence_count=len(sentence_spans(block)) if block.kind == "paragraph" else 0,
                        semantic_refined=False,
                        ordinal=len(chunks),
                        document_created_at=document_created_at,
                        document_modified_at=document_modified_at,
                    )
                )
                chunk_memberships.append({block.index})
                continue
            kept_gaps = {
                item.gap_index
                for item in decisions_by_block.get(block.index, [])
                if item.right_block_index is None and item.keep_boundary
            }
            sentence_start = 0
            for gap_index in range(len(refined.sentences) - 1):
                if gap_index not in kept_gaps:
                    continue
                chunks.append(
                    self._sentence_range_chunk(
                        markdown,
                        document_id,
                        source_path,
                        refined,
                        sentence_start,
                        gap_index + 1,
                        len(chunks),
                        document_created_at,
                        document_modified_at,
                    )
                )
                chunk_memberships.append({block.index})
                sentence_start = gap_index + 1
            chunks.append(
                self._sentence_range_chunk(
                    markdown,
                    document_id,
                    source_path,
                    refined,
                    sentence_start,
                    len(refined.sentences),
                    len(chunks),
                    document_created_at,
                    document_modified_at,
                )
            )
            chunk_memberships.append({block.index})

        chunks, chunk_memberships = self._merge_cross_block_candidates(
            markdown,
            chunks,
            chunk_memberships,
            decisions,
        )

        chunks = [
            replace(
                chunk,
                ordinal=index,
                previous_chunk_id=chunks[index - 1].id if index else None,
                next_chunk_id=chunks[index + 1].id if index + 1 < len(chunks) else None,
            )
            for index, chunk in enumerate(chunks)
        ]

        return ChunkingResult(
            document_id=document_id,
            source_path=source_path,
            chunks=tuple(chunks),
            structural_blocks=tuple(blocks),
            boundary_decisions=tuple(decisions),
            metadata={
                "chunker": "markdown-structural+gate-v2+small-v3-50-normal",
                "boundary_keep_threshold": self.config.boundary_keep_threshold,
                "candidate_budget": self.config.candidate_budget,
                "semantic_prose_block_count": len(refinable),
                "gate_window_size": self.config.gate_window_size,
            },
        )

    def _embed_refinable_blocks(
        self,
        document_id: str,
        specs: Sequence[tuple[StructuralBlock, tuple[SentenceSpan, ...]]],
    ) -> list[_RefinableBlock]:
        texts = [span.embedding_text for _, spans in specs for span in spans]
        if not texts:
            return []
        embeddings = self._load_or_encode(document_id, texts)
        if len(embeddings) != len(texts):
            raise ValueError("Embedding count does not match the refinable sentence count.")
        result: list[_RefinableBlock] = []
        offset = 0
        for block, spans in specs:
            end = offset + len(spans)
            result.append(_RefinableBlock(block, spans, embeddings[offset:end]))
            offset = end
        return result

    def _load_or_encode(self, document_id: str, texts: Sequence[str]) -> np.ndarray:
        identity = "\n".join((self.config.model_id, document_id, *texts)).encode("utf-8")
        cache_path = self.config.cache_directory / f"{hashlib.sha256(identity).hexdigest()}.npy"
        if cache_path.is_file():
            return np.load(cache_path, allow_pickle=False)
        embeddings = np.asarray(self._embedding_provider().encode(texts), dtype=np.float32)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings, allow_pickle=False)
        return embeddings

    def _decide_boundaries(self, refinable: Sequence[_RefinableBlock]) -> list[BoundaryDecision]:
        gaps: list[_Gap] = []
        block_map = {item.block.index: item for item in refinable}
        for stream in self._prose_streams(refinable):
            embeddings = np.concatenate([item.embeddings for item in stream], axis=0)
            references = [
                (item.block.index, sentence_index)
                for item in stream
                for sentence_index in range(len(item.sentences))
            ]
            similarities = contextual_cosine_curve(embeddings, self.config.gate_window_size)
            valleys = local_valleys(similarities)
            for gap_index, score in enumerate(similarities):
                left_block_index, left_sentence_index = references[gap_index]
                right_block_index, _ = references[gap_index + 1]
                gaps.append(
                    _Gap(
                        block_index=left_block_index,
                        right_block_index=(
                            right_block_index if right_block_index != left_block_index else None
                        ),
                        gap_index=left_sentence_index,
                        cosine=float(score),
                        local_valley=bool(valleys[gap_index]),
                    )
                )
        if not gaps:
            return []

        ranked = sorted(
            gaps,
            key=lambda item: (
                0 if item.local_valley else 1,
                item.cosine,
                item.block_index,
                item.right_block_index if item.right_block_index is not None else -1,
                item.gap_index,
            ),
        )
        candidate_count = min(len(ranked), max(1, ceil(len(ranked) * self.config.candidate_budget)))
        candidates = ranked[:candidate_count]
        ranks = {
            self._gap_key(item): rank
            for rank, item in enumerate(candidates, 1)
        }
        contexts = [
            self._boundary_context(block_map, item)
            for item in candidates
        ]
        scores = self._boundary_verifier().predict(contexts)
        if len(scores) != len(candidates):
            raise ValueError("Boundary verifier returned a probability count different from candidates.")
        probabilities = {
            self._gap_key(item): float(score)
            for item, score in zip(candidates, scores)
        }

        decisions: list[BoundaryDecision] = []
        for item in sorted(
            gaps,
            key=lambda gap: (
                gap.block_index,
                gap.gap_index,
                gap.right_block_index if gap.right_block_index is not None else -1,
            ),
        ):
            key = self._gap_key(item)
            probability = probabilities.get(key)
            left_sentences = block_map[item.block_index].sentences
            right_sentences = (
                block_map[item.right_block_index].sentences
                if item.right_block_index is not None
                else left_sentences
            )
            decisions.append(
                BoundaryDecision(
                    block_index=item.block_index,
                    right_block_index=item.right_block_index,
                    gap_index=item.gap_index,
                    left_sentence=left_sentences[item.gap_index].embedding_text,
                    right_sentence=(
                        right_sentences[0].embedding_text
                        if item.right_block_index is not None
                        else right_sentences[item.gap_index + 1].embedding_text
                    ),
                    contextual_cosine=item.cosine,
                    local_valley=item.local_valley,
                    candidate_rank=ranks.get(key),
                    is_candidate=key in probabilities,
                    boundary_probability=probability,
                    keep_boundary=(
                        probability is not None
                        and probability >= self.config.boundary_keep_threshold
                    ),
                    effective_threshold=self.config.boundary_keep_threshold,
                )
            )
        return decisions

    @staticmethod
    def _prose_streams(refinable: Sequence[_RefinableBlock]) -> list[list[_RefinableBlock]]:
        """Join adjacent prose blocks under the same heading for cross-paragraph gaps."""
        streams: list[list[_RefinableBlock]] = []
        for item in refinable:
            if streams:
                previous = streams[-1][-1]
                if (
                    previous.block.index + 1 == item.block.index
                    and previous.block.heading_path == item.block.heading_path
                ):
                    streams[-1].append(item)
                    continue
            streams.append([item])
        return streams

    @staticmethod
    def _gap_key(item: _Gap) -> tuple[int, int | None, int]:
        return item.block_index, item.right_block_index, item.gap_index

    def _boundary_context(
        self,
        block_map: dict[int, _RefinableBlock],
        item: _Gap,
    ):
        if item.right_block_index is None:
            return boundary_context(
                block_map[item.block_index].embeddings,
                item.gap_index,
                self.config.attention_context_window,
            )
        left = block_map[item.block_index].embeddings
        right = block_map[item.right_block_index].embeddings
        combined = np.concatenate((left, right), axis=0)
        return boundary_context(
            combined,
            len(left) - 1,
            self.config.attention_context_window,
        )

    def _merge_cross_block_candidates(
        self,
        markdown: str,
        chunks: list[Chunk],
        memberships: list[set[int]],
        decisions: Sequence[BoundaryDecision],
    ) -> tuple[list[Chunk], list[set[int]]]:
        """Merge adjacent prose blocks only when their candidate predicts MERGE."""
        merge_decisions = [
            decision
            for decision in decisions
            if decision.right_block_index is not None
            and decision.is_candidate
            and not decision.keep_boundary
        ]
        for decision in merge_decisions:
            right_block_index = decision.right_block_index
            assert right_block_index is not None
            for index in range(len(chunks) - 1):
                if (
                    decision.block_index in memberships[index]
                    and right_block_index in memberships[index + 1]
                ):
                    chunks[index] = self._merge_chunks(markdown, chunks[index], chunks[index + 1])
                    memberships[index] = memberships[index] | memberships[index + 1]
                    del chunks[index + 1]
                    del memberships[index + 1]
                    break
        return chunks, memberships

    @staticmethod
    def _merge_chunks(markdown: str, left: Chunk, right: Chunk) -> Chunk:
        start = left.source_start
        end = right.source_end
        source_text = markdown[start:end]
        text = normalize_chunk_text(source_text, left.kind)
        content_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
        identity = f"{left.document_id}|{start}|{end}|{content_hash}".encode("utf-8")
        return replace(
            left,
            id=f"chunk:{hashlib.sha1(identity).hexdigest()[:20]}",
            text=text,
            source_text=source_text,
            source_end=end,
            sentence_count=left.sentence_count + right.sentence_count,
            semantic_refined=True,
            content_hash=content_hash,
            previous_chunk_id=None,
            next_chunk_id=None,
        )

    def _sentence_range_chunk(
        self,
        markdown: str,
        document_id: str,
        source_path: str,
        item: _RefinableBlock,
        sentence_start: int,
        sentence_end: int,
        ordinal: int,
        document_created_at: str | None,
        document_modified_at: str | None,
    ) -> Chunk:
        start = item.block.source_start if sentence_start == 0 else item.sentences[sentence_start].source_start
        end = (
            item.block.source_end
            if sentence_end == len(item.sentences)
            else item.sentences[sentence_end].source_start
        )
        return self._make_chunk(
            markdown,
            document_id,
            source_path,
            item.block,
            start,
            end,
            sentence_count=sentence_end - sentence_start,
            semantic_refined=True,
            ordinal=ordinal,
            document_created_at=document_created_at,
            document_modified_at=document_modified_at,
        )

    @staticmethod
    def _make_chunk(
        markdown: str,
        document_id: str,
        source_path: str,
        block: StructuralBlock,
        start: int,
        end: int,
        *,
        sentence_count: int,
        semantic_refined: bool,
        ordinal: int,
        document_created_at: str | None,
        document_modified_at: str | None,
    ) -> Chunk:
        while start < end and markdown[start].isspace():
            start += 1
        while end > start and markdown[end - 1].isspace():
            end -= 1
        source_text = markdown[start:end]
        text = normalize_chunk_text(source_text, block.kind)
        content_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
        identity = f"{document_id}|{start}|{end}|{content_hash}".encode("utf-8")
        return Chunk(
            id=f"chunk:{hashlib.sha1(identity).hexdigest()[:20]}",
            document_id=document_id,
            source_path=source_path,
            ordinal=ordinal,
            kind=block.kind,
            text=text,
            source_text=source_text,
            source_start=start,
            source_end=end,
            heading_path=block.heading_path,
            chunk_heading=block.heading_path[-1] if block.heading_path else None,
            previous_chunk_id=None,
            next_chunk_id=None,
            document_created_at=document_created_at,
            document_modified_at=document_modified_at,
            sentence_count=sentence_count,
            semantic_refined=semantic_refined,
            content_hash=content_hash,
        )

    def _embedding_provider(self) -> EmbeddingProvider:
        if self._embedder is None:
            self._embedder = QwenSentenceEmbedder(
                self.config.model_id,
                device=self.config.device,
                batch_size=self.config.embedding_batch_size,
            )
        return self._embedder

    def _boundary_verifier(self) -> BoundaryVerifier:
        if self._verifier is None:
            self._verifier = SmallV3BoundaryVerifier(
                self.config.checkpoint_path,
                device=self.config.device,
                batch_size=self.config.inference_batch_size,
            )
        return self._verifier


def _document_timestamps(markdown: str, path: Path) -> tuple[str | None, str | None]:
    """Read frontmatter dates first, then fall back to filesystem timestamps."""
    values: dict[str, str] = {}
    lines = markdown.splitlines()
    if lines and lines[0].strip() == "---":
        for line in lines[1:]:
            if line.strip() == "---":
                break
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            values[key.strip().lower()] = value.strip().strip("\"'")

    created = next(
        (values[key] for key in ("created", "created_at", "date") if values.get(key)),
        None,
    )
    modified = next(
        (
            values[key]
            for key in ("modified", "modified_at", "updated", "updated_at")
            if values.get(key)
        ),
        None,
    )
    stat = path.stat()
    created = created or _format_timestamp(stat.st_ctime)
    modified = modified or _format_timestamp(stat.st_mtime)
    return created, modified


def _format_timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, tz=timezone.utc).astimezone().isoformat()
