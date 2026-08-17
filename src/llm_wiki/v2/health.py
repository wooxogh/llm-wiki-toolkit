"""v2 health checks for provenance, NET integrity, and concept index freshness."""
from __future__ import annotations

import json
import os
from pathlib import Path

from llm_wiki import config
from llm_wiki.v2 import artifacts, concept_index
from llm_wiki.v2.concept_store import collect_documents, read_chunks, read_concepts, read_documents
from llm_wiki.v2.net_store import NetStore
from llm_wiki.v2.schemas import ConceptState


def check_v2_health(vault: Path | None = None) -> list[str]:
    issues: list[str] = []
    schema_issue = artifacts.schema_staleness(vault)
    if schema_issue:
        issues.append(schema_issue)
    docs = {d.id: d for d in read_documents(vault)}
    live_docs = {d.id: d for d in collect_documents(vault)}
    chunks = {c.id: c for c in read_chunks(vault)}
    concepts = read_concepts(vault)
    concept_state = artifacts.artifact_path("concept_build_state.json", vault)
    if concepts and not concept_state.exists():
        issues.append("v2 Concept build identity is missing; run a full wiki-concepts build")
    elif concepts:
        try:
            saved = json.loads(concept_state.read_text(encoding="utf-8"))
            expected_model = _expected_llm_identity(vault)
            if saved.get("target_chars") != config.load(vault).v2_chunk_target_chars:
                issues.append("v2 Concept chunk target identity is stale")
            if saved.get("model_identity") != expected_model:
                issues.append("v2 Concept LLM model identity is stale")
        except (OSError, json.JSONDecodeError):
            issues.append("v2 Concept build identity is invalid")
    if not chunks:
        issues.append("v2 chunks are missing")
    expected_target = config.load(vault).v2_chunk_target_chars
    for chunk in chunks.values():
        if chunk.target_chars != expected_target:
            issues.append(f"chunk {chunk.id} target config is stale ({chunk.target_chars} != {expected_target})")
    for doc_id, live in live_docs.items():
        stored = docs.get(doc_id)
        if not stored:
            issues.append(f"v2 document artifact missing for live document {doc_id}")
        elif stored.content_hash != live.content_hash or stored.path != live.path:
            issues.append(f"v2 document artifact stale for {doc_id}")
    for doc_id in docs:
        if doc_id not in live_docs:
            issues.append(f"v2 document artifact refers to deleted source {doc_id}")
    for chunk in chunks.values():
        if chunk.document_id not in docs:
            issues.append(f"chunk {chunk.id} points at missing document {chunk.document_id}")
        if not chunk.path or chunk.source_end < chunk.source_start:
            issues.append(f"chunk {chunk.id} has invalid source span")
        live = live_docs.get(chunk.document_id)
        if live and live.content_hash != docs.get(chunk.document_id, live).content_hash:
            issues.append(f"chunk {chunk.id} belongs to stale document artifact")
    for concept in concepts:
        chunk = chunks.get(concept.chunk_id)
        if concept.document_id not in docs:
            issues.append(f"concept {concept.id} points at missing document {concept.document_id}")
        if not chunk:
            issues.append(f"concept {concept.id} points at missing chunk {concept.chunk_id}")
            continue
        if concept.chunk_hash != chunk.content_hash:
            issues.append(f"concept {concept.id} chunk hash is stale")
        if concept.source_quote not in chunk.text:
            issues.append(f"concept {concept.id} source quote is not in its chunk")
        if concept.state not in {state.value for state in ConceptState}:
            issues.append(f"concept {concept.id} has invalid lifecycle state {concept.state}")
    stale = concept_index.is_stale(vault)
    if stale:
        issues.append(stale)
    store = NetStore(vault)
    net_state = artifacts.artifact_path("net_build_state.json", vault)
    if store.nodes() and not net_state.exists():
        issues.append("v2 NET build identity is missing; run wiki-net build")
    elif store.nodes():
        try:
            saved = json.loads(net_state.read_text(encoding="utf-8"))
            prompts = artifacts.current_schema_manifest()["prompt_versions"]
            if any(saved.get(f"{name}_prompt_version") != prompts[name]
                   for name in ("placement", "relation", "temporal")):
                issues.append("v2 NET prompt identity is stale; run wiki-net build")
            if saved.get("model_identity") != _expected_llm_identity(vault):
                issues.append("v2 NET LLM agent/model identity is stale; run wiki-net build")
        except (OSError, json.JSONDecodeError):
            issues.append("v2 NET build identity is invalid; run wiki-net build")
    issues.extend(store.health_issues())
    return sorted(set(issues))


def _expected_llm_identity(vault: Path | None) -> str:
    cfg = config.load(vault)
    fallback = (os.environ.get("WIKI_V2_LLM_COMMAND")
                or (f"{cfg.v2_agent}-cli-default" if cfg.v2_agent else "offline"))
    return os.environ.get("WIKI_V2_LLM_MODEL", fallback)
