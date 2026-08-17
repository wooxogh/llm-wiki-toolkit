"""Reconcile the generated NET layer without overwriting user-owned structure."""
from __future__ import annotations

import json
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Callable

from llm_wiki import config
from llm_wiki.v2 import artifacts
from llm_wiki.v2.concept_store import read_concepts, read_documents, write_concepts
from llm_wiki.v2.llm_adapter import UserLLMAdapter, default_adapter
from llm_wiki.v2.models import NetEdge, NetNode, Operation
from llm_wiki.v2.net_store import NetStore
from llm_wiki.v2.relation_candidates import discover
from llm_wiki.v2.relation_classifier import classify
from llm_wiki.v2.review import submit_relation_proposal
from llm_wiki.v2.schemas import ConceptState, EdgeType, NodeType
from llm_wiki.v2.temporal import resolve
from llm_wiki.v2.tree_ops import op_id

ROOT_TOPIC_ID = "topic:knowledge"
NetProgress = Callable[[str, int, int, str], None]


def build_net(vault: Path | None = None, adapter: UserLLMAdapter | None = None,
              allow_ai_topic_creation: bool = True,
              progress: NetProgress | None = None,
              changed_only: bool = False) -> NetStore:
    """Add/update source-derived nodes while retaining user tree and review state."""
    adapter = adapter or default_adapter(vault)
    cfg = config.load(vault)
    allow_ai_topic_creation = allow_ai_topic_creation and cfg.v2_allow_ai_topic_creation
    docs, concepts = read_documents(vault), read_concepts(vault)
    store = NetStore(vault)
    # Prompt-generated open proposals are rebuildable artifacts. Terminal human
    # decisions remain audit records, while stale open proposals are discarded
    # and regenerated under the current contract.
    terminal_proposals = [proposal for proposal in store.proposals()
                          if proposal.status in {"APPROVED", "REJECTED"}]
    terminal_ids = {proposal.id for proposal in terminal_proposals}
    store.write_proposals(terminal_proposals)
    store.write_review_items([item for item in store.review_items()
                              if item.proposal_id in terminal_ids and item.state != "OPEN"])
    approved_edge_ids = {f"edge:{proposal.id}" for proposal in terminal_proposals
                         if proposal.status == "APPROVED"}
    existing_nodes = {node.id: node for node in store.nodes()}
    dirty_ids = _dirty_concept_ids(concepts, existing_nodes) if changed_only else None
    existing_edges = [edge for edge in store.edges()
                      if edge.type != EdgeType.RELATES_TO.value or edge.id in approved_edge_ids]
    existing_document_placements = {
        edge.target for edge in existing_edges if edge.type == EdgeType.CONTAINS_DOCUMENT.value
    }
    document_ids = {f"document:{doc.id}" for doc in docs}
    concept_ids = {concept.id for concept in concepts}

    # Topics/collections are user-owned after creation. Keep their labels,
    # parents, archived state, and user-created collections across rebuilds.
    nodes = [node for node in existing_nodes.values()
             if node.type in {NodeType.TOPIC.value, NodeType.COLLECTION.value}]
    if ROOT_TOPIC_ID not in {node.id for node in nodes}:
        nodes.append(NetNode.topic("Knowledge", ROOT_TOPIC_ID, created_by="system"))
    for doc in docs:
        prior = existing_nodes.get(f"document:{doc.id}")
        nodes.append(NetNode(f"document:{doc.id}", NodeType.DOCUMENT.value, doc.id,
                             state=prior.state if prior else "ACTIVE", created_by="system",
                             attrs={"path": doc.path, "content_hash": doc.content_hash}))
    for concept in concepts:
        prior = existing_nodes.get(concept.id)
        nodes.append(NetNode(concept.id, NodeType.CONCEPT.value, concept.summary or concept.text,
                             state=prior.state if prior else "ACTIVE", created_by="system",
                             attrs={"document_id": concept.document_id, "chunk_id": concept.chunk_id,
                                    "concept_state": concept.state or ConceptState.ACTIVE.value,
                                    "chunk_hash": concept.chunk_hash,
                                    "text_hash": _concept_text_hash(concept)}))
    nodes = _dedupe_nodes(nodes)

    # Retain user structure and committed semantic edges, dropping only endpoints
    # that no longer exist in canonical source artifacts.
    node_ids = {node.id for node in nodes}
    edges = [edge for edge in existing_edges if edge.source in node_ids and edge.target in node_ids]
    edges = [edge for edge in edges if edge.type != EdgeType.DOCUMENT_HAS_CONCEPT.value]
    for concept in concepts:
        edge = NetEdge(f"edge:document-concept:{concept.document_id}:{concept.id}",
                       EdgeType.DOCUMENT_HAS_CONCEPT.value, f"document:{concept.document_id}", concept.id)
        edges.append(edge)

    # Every document belongs to exactly one visible tree location. Preserve a
    # user's Topic/Collection placement; otherwise give it the root default.
    for document_id in document_ids:
        memberships = [edge for edge in edges if edge.type == EdgeType.CONTAINS_DOCUMENT.value and edge.target == document_id]
        if not memberships:
            edges.append(NetEdge(f"edge:contains:{ROOT_TOPIC_ID}:{document_id}",
                                 EdgeType.CONTAINS_DOCUMENT.value, ROOT_TOPIC_ID, document_id))

    topic_ids = {node.id for node in nodes if node.type == NodeType.TOPIC.value}
    collection_ids = {node.id for node in nodes if node.type == NodeType.COLLECTION.value}
    membership_by_concept = {edge.target for edge in edges if edge.type == EdgeType.PRIMARY_TOPIC_OF.value}
    collection_decided_docs: set[str] = set()
    unplaced = [concept for concept in concepts if concept.id not in membership_by_concept]
    for index, concept in enumerate(unplaced, start=1):
        proposal = adapter.place_concept(concept, _placement_candidates(nodes, edges, concept))
        if proposal.concept_id != concept.id:
            if progress:
                progress("placement", index, len(unplaced), concept.summary or concept.id)
            continue
        topic_id = proposal.primary_topic_id if proposal.primary_topic_id in topic_ids else None
        if not topic_id and proposal.create_topic_label and allow_ai_topic_creation:
            topic_id = _topic_id(proposal.create_topic_label)
            if topic_id not in topic_ids:
                nodes.append(NetNode.topic(proposal.create_topic_label, topic_id, created_by="ai"))
                topic_ids.add(topic_id)
                edges.append(NetEdge(f"edge:parent:{ROOT_TOPIC_ID}:{topic_id}", EdgeType.PARENT_OF.value,
                                     ROOT_TOPIC_ID, topic_id))
        topic_id = topic_id or ROOT_TOPIC_ID
        collection_id = proposal.collection_id if proposal.collection_id in collection_ids else None
        if not collection_id and proposal.create_collection_label and allow_ai_topic_creation:
            collection_id = _collection_id(proposal.create_collection_label)
            if collection_id not in collection_ids:
                nodes.append(NetNode(collection_id, NodeType.COLLECTION.value,
                                     proposal.create_collection_label, created_by="ai",
                                     attrs={"collection_type": proposal.collection_type}
                                     if proposal.collection_type else {}))
                collection_ids.add(collection_id)
                edges.append(NetEdge(f"edge:parent:{topic_id}:{collection_id}", EdgeType.PARENT_OF.value,
                                     topic_id, collection_id))
        document_id = f"document:{concept.document_id}"
        if (collection_id and document_id not in existing_document_placements
                and document_id not in collection_decided_docs):
            edges = [edge for edge in edges if not (
                edge.type == EdgeType.CONTAINS_DOCUMENT.value and edge.target == document_id
            )]
            edges.append(NetEdge(f"edge:contains:{collection_id}:{document_id}",
                                 EdgeType.CONTAINS_DOCUMENT.value, collection_id, document_id))
            collection_decided_docs.add(document_id)
        edges.append(NetEdge(f"edge:primary:{topic_id}:{concept.id}", EdgeType.PRIMARY_TOPIC_OF.value,
                             topic_id, concept.id, confidence=proposal.confidence))
        for secondary_id in proposal.secondary_topic_ids:
            if secondary_id in topic_ids and secondary_id != topic_id:
                edges.append(NetEdge(f"edge:secondary:{secondary_id}:{concept.id}", EdgeType.SECONDARY_TOPIC_OF.value,
                                     secondary_id, concept.id, confidence=proposal.confidence))
        if progress:
            progress("placement", index, len(unplaced), concept.summary or concept.id)

    store.replace_graph(_dedupe_nodes(nodes), _dedupe_edges(edges))
    _sync_concept_membership(store, concepts, vault)
    if not store.operations():
        store.append_operation(Operation(op_id("BUILD_NET", str(len(concepts))), "BUILD_NET", "system", {}, {"concepts": len(concepts)}))
    _discover_relations(store, concepts, adapter, cfg, progress, dirty_ids)
    artifacts.artifact_path("net_build_state.json", vault).write_text(json.dumps({
        "placement_prompt_version": artifacts.current_schema_manifest()["prompt_versions"]["placement"],
        "relation_prompt_version": artifacts.current_schema_manifest()["prompt_versions"]["relation"],
        "temporal_prompt_version": artifacts.current_schema_manifest()["prompt_versions"]["temporal"],
        "model_identity": getattr(adapter, "model_identity", "offline"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    artifacts.write_schema_manifest(vault)
    return store


def _discover_relations(store: NetStore, concepts, adapter, cfg,
                        progress: NetProgress | None = None,
                        dirty_ids: set[str] | None = None) -> None:
    terminal = {proposal.id for proposal in store.proposals() if proposal.status in {"APPROVED", "REJECTED"}}
    pairs = _candidate_pairs(store.vault, concepts, cfg.v2_relation_candidate_topk,
                             cfg.v2_relation_candidate_min_score, progress)
    if dirty_ids is not None:
        pairs = [(source, target) for source, target in pairs
                if source.id in dirty_ids or target.id in dirty_ids]
    total = len(pairs)
    # Every submit_relation_proposal() call below happens on THIS thread, no
    # matter how _compute_relation_proposals computes its results (serially
    # today, optionally concurrently once a worker count is threaded in) —
    # NetStore's append/upsert methods do an unlocked read-modify-write and
    # are not safe to call from more than one thread.
    for index, (source, target, proposals) in enumerate(
        _compute_relation_proposals(pairs, adapter, store.vault, cfg.v2_relation_concurrency), start=1
    ):
        for proposal in proposals:
            if proposal.id not in terminal:
                submit_relation_proposal(store, proposal, cfg.v2_safe_relation_min_confidence,
                                         cfg.v2_require_user_approval)
        if progress:
            progress("relations", index, total, f"{source.id} -> {target.id}")


def _classify_pair(adapter, vault, source, target):
    relation = classify(adapter, source, target, vault=vault)
    temporal = resolve(adapter, source, target, relation, vault=vault)
    return source, target, [p for p in (relation, temporal) if p is not None]


def _compute_relation_proposals(pairs, adapter, vault, workers: int = 1):
    if workers <= 1:
        for source, target in pairs:
            yield _classify_pair(adapter, vault, source, target)
        return
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = [pool.submit(_classify_pair, adapter, vault, source, target) for source, target in pairs]
        for future in as_completed(futures):
            yield future.result()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def _candidate_pairs(vault, concepts, topk: int, min_score: float,
                     progress: NetProgress | None = None) -> list[tuple]:
    pairs = []
    for index, source in enumerate(concepts, start=1):
        for score, target in discover(vault, source, concepts, topk):
            if score >= min_score:
                pairs.append((source, target))
        if progress:
            progress("candidates", index, len(concepts), source.summary or source.id)
    return pairs


def _concept_text_hash(concept) -> str:
    text = " ".join((concept.text, concept.summary, concept.source_quote))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _dirty_concept_ids(concepts, existing_nodes: dict) -> set[str]:
    current = {concept.id: concept for concept in concepts}
    dirty: set[str] = set()
    for concept in concepts:
        prior = existing_nodes.get(concept.id)
        if not prior or prior.type != NodeType.CONCEPT.value:
            dirty.add(concept.id)
            continue
        if (prior.attrs.get("chunk_hash") != concept.chunk_hash
                or prior.attrs.get("text_hash") != _concept_text_hash(concept)):
            dirty.add(concept.id)
    return dirty


def _sync_concept_membership(store: NetStore, concepts, vault: Path | None) -> None:
    primary = {edge.target: edge.source for edge in store.edges() if edge.type == EdgeType.PRIMARY_TOPIC_OF.value}
    secondary: dict[str, list[str]] = {}
    for edge in store.edges():
        if edge.type == EdgeType.SECONDARY_TOPIC_OF.value:
            secondary.setdefault(edge.target, []).append(edge.source)
    write_concepts([replace(concept, primary_topic_id=primary.get(concept.id),
                            secondary_topic_ids=sorted(secondary.get(concept.id, []))) for concept in concepts], vault)


def _topic_id(label: str) -> str:
    slug = "-".join("".join(ch.lower() if ch.isalnum() else " " for ch in label).split())
    return f"topic:{slug or 'untitled'}"


def _collection_id(label: str) -> str:
    return _topic_id(label).replace("topic:", "collection:", 1)


def _placement_candidates(nodes: list[NetNode], edges: list[NetEdge], concept) -> list[dict]:
    parent = {edge.target: edge.source for edge in edges if edge.type == EdgeType.PARENT_OF.value}
    document_id = f"document:{concept.document_id}"
    document_parent = next((edge.source for edge in edges
                            if edge.type == EdgeType.CONTAINS_DOCUMENT.value and edge.target == document_id), None)
    rows = []
    for node in nodes:
        if node.type not in {NodeType.TOPIC.value, NodeType.COLLECTION.value}:
            continue
        rows.append({
            "id": node.id,
            "type": node.type,
            "label": node.label,
            "parent_id": parent.get(node.id),
            "state": node.state,
            "collection_type": node.attrs.get("collection_type"),
            "contains_source_document": node.id == document_parent,
        })
    return sorted(rows, key=lambda row: (row["type"], row["id"]))


def _dedupe_nodes(nodes: list[NetNode]) -> list[NetNode]:
    return sorted({node.id: node for node in nodes}.values(), key=lambda node: node.id)


def _dedupe_edges(edges: list[NetEdge]) -> list[NetEdge]:
    return sorted({edge.id: edge for edge in edges}.values(), key=lambda edge: edge.id)
