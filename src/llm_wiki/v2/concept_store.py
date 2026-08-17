"""Build and read v2 document/chunk/concept artifacts."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Callable

from llm_wiki import build_index
from llm_wiki.paths import content_paths, content_root, page_hash, relative
from llm_wiki.v2 import artifacts
from llm_wiki.v2.chunking import chunk_document
from llm_wiki.v2.concept_extraction import extract
from llm_wiki.v2.llm_adapter import UserLLMAdapter, default_adapter
from llm_wiki.v2.models import Chunk, Concept, ConceptProposal, Document
from llm_wiki.v2.schemas import (CONCEPT_PROMPT_VERSION, CONCEPT_SCHEMA_VERSION,
                                  DEFAULT_CHUNK_TARGET_CHARS)


def document_id_for(path: Path) -> str:
    fm = build_index.parse_frontmatter(path) or {}
    return str(fm.get("id") or path.stem)


def collect_documents(vault: Path | None = None) -> list[Document]:
    docs = []
    for path in content_paths(vault):
        raw = path.read_text(encoding="utf-8")
        fm = build_index.parse_frontmatter(path) or {}
        docs.append(Document(
            id=str(fm.get("id") or path.stem),
            path=relative(path, vault),
            content_hash=page_hash(raw),
            metadata=fm,
            updated_at=str(fm.get("updated")) if fm.get("updated") is not None else None,
        ))
    return docs


def build_chunks(vault: Path | None = None, target_chars: int = DEFAULT_CHUNK_TARGET_CHARS) -> list[Chunk]:
    root = content_root(vault)
    chunks: list[Chunk] = []
    for doc in collect_documents(vault):
        path = root / doc.path
        chunks.extend(chunk_document(doc.id, doc.path, path.read_text(encoding="utf-8"), target_chars))
    return chunks


def _concept_id(chunk: Chunk, proposal: ConceptProposal) -> str:
    raw = f"{chunk.id}|{proposal.source_quote}|{proposal.text}"
    return f"concept:{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def concepts_from_chunks(
    chunks: list[Chunk],
    adapter: UserLLMAdapter | None = None,
    progress: Callable[[int, int, Chunk], None] | None = None,
) -> list[Concept]:
    adapter = adapter or default_adapter()
    concepts: list[Concept] = []
    for index, chunk in enumerate(chunks, start=1):
        for proposal in extract(chunk, adapter, getattr(adapter, "model_identity", "offline")):
            if not proposal.source_quote or proposal.source_quote not in chunk.text:
                continue
            start_in_chunk = chunk.text.index(proposal.source_quote)
            concepts.append(Concept(
                id=_concept_id(chunk, proposal),
                document_id=chunk.document_id,
                chunk_id=chunk.id,
                text=proposal.text,
                summary=proposal.summary,
                source_quote=proposal.source_quote,
                confidence=max(0.0, min(1.0, float(proposal.confidence))),
                chunk_hash=chunk.content_hash,
                source_start=chunk.source_start + start_in_chunk,
                source_end=chunk.source_start + start_in_chunk + len(proposal.source_quote),
                heading_path=chunk.heading_path,
            ))
        if progress:
            progress(index, len(chunks), chunk)
    return concepts


def build_concepts(
    vault: Path | None = None,
    target_chars: int = DEFAULT_CHUNK_TARGET_CHARS,
    adapter: UserLLMAdapter | None = None,
    changed_only: bool = False,
    progress: Callable[[int, int, Chunk], None] | None = None,
) -> tuple[list[Document], list[Chunk], list[Concept]]:
    artifacts.ensure_layout(vault)
    adapter = adapter or default_adapter(vault)
    previous_chunks = read_chunks(vault)
    previous_concepts = read_concepts(vault)
    identity = {
        "target_chars": target_chars,
        "model_identity": getattr(adapter, "model_identity", "offline"),
    }
    state_path = artifacts.artifact_path("concept_build_state.json", vault)
    previous_identity = None
    if state_path.exists():
        try:
            previous_identity = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous_identity = None
    incompatible = bool(
        artifacts.schema_staleness(vault)
        or (previous_chunks and any(chunk.target_chars != target_chars for chunk in previous_chunks))
        or (previous_concepts and any(
            concept.prompt_version != CONCEPT_PROMPT_VERSION
            or concept.schema_version != CONCEPT_SCHEMA_VERSION
            for concept in previous_concepts
        ))
        or (previous_identity is not None and previous_identity != identity)
    )
    # `--changed` is only safe when every non-source input is unchanged. An
    # incompatible prompt/schema/model/chunk target transparently becomes a full
    # rebuild so stale generated Concepts cannot survive.
    if changed_only and incompatible:
        changed_only = False
    docs = collect_documents(vault)
    previous_docs = {doc.id: doc for doc in read_documents(vault)} if changed_only else {}
    changed_ids = {doc.id for doc in docs if previous_docs.get(doc.id, None) is None
                   or previous_docs[doc.id].content_hash != doc.content_hash}
    if changed_only and not changed_ids:
        return docs, read_chunks(vault), read_concepts(vault)
    rebuilt = build_chunks(vault, target_chars)
    if changed_only:
        retained_chunks = [chunk for chunk in read_chunks(vault)
                           if chunk.document_id not in changed_ids and chunk.document_id in {doc.id for doc in docs}]
        chunks = retained_chunks + [chunk for chunk in rebuilt if chunk.document_id in changed_ids]
    else:
        chunks = rebuilt
    by_doc: dict[str, list[str]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.document_id, []).append(chunk.id)
    docs = [Document(**{**doc.to_dict(), "chunk_ids": by_doc.get(doc.id, [])}) for doc in docs]
    rebuilt_concepts = concepts_from_chunks(
        [chunk for chunk in chunks if chunk.document_id in changed_ids], adapter, progress)
    if changed_only:
        concepts = [concept for concept in read_concepts(vault)
                    if concept.document_id not in changed_ids and concept.document_id in {doc.id for doc in docs}] + rebuilt_concepts
    else:
        concepts = rebuilt_concepts
    prior_by_id = {concept.id: concept for concept in previous_concepts}
    doc_updated = {doc.id: doc.updated_at for doc in docs}
    reconciled = []
    for concept in concepts:
        prior = prior_by_id.get(concept.id)
        reconciled.append(replace(
            concept,
            created_at=prior.created_at if prior else concept.created_at,
            updated_at=doc_updated.get(concept.document_id),
            state=prior.state if prior else concept.state,
            primary_topic_id=prior.primary_topic_id if prior else concept.primary_topic_id,
            secondary_topic_ids=prior.secondary_topic_ids if prior else concept.secondary_topic_ids,
        ))
    concepts = reconciled
    root = artifacts.artifact_root(vault)
    artifacts.write_jsonl(root / "documents.jsonl", docs)
    artifacts.write_jsonl(root / "chunks.jsonl", sorted(chunks, key=lambda chunk: (chunk.document_id, chunk.ordinal)))
    artifacts.write_jsonl(root / "concepts.jsonl", concepts)
    state_path.write_text(json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8")
    artifacts.write_schema_manifest(vault)
    return docs, chunks, concepts


def read_documents(vault: Path | None = None) -> list[Document]:
    return artifacts.read_jsonl(artifacts.artifact_path("documents.jsonl", vault), Document.from_dict)


def read_chunks(vault: Path | None = None) -> list[Chunk]:
    return artifacts.read_jsonl(artifacts.artifact_path("chunks.jsonl", vault), Chunk.from_dict)


def read_concepts(vault: Path | None = None) -> list[Concept]:
    return artifacts.read_jsonl(artifacts.artifact_path("concepts.jsonl", vault), Concept.from_dict)


def write_concepts(concepts: list[Concept], vault: Path | None = None) -> None:
    artifacts.write_jsonl(artifacts.artifact_path("concepts.jsonl", vault), concepts)
