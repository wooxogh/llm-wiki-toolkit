"""Validated, reversible user operations over the NET tree."""
from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

from llm_wiki.v2.models import NetEdge, NetNode, Operation
from llm_wiki.v2.net_store import NetIntegrityError, NetStore
from llm_wiki.v2.schemas import EdgeType, NodeType


def op_id(op: str, payload: str = "") -> str:
    return f"op:{op}:{uuid4().hex[:12]}"


def _snapshot(nodes: list[NetNode], edges: list[NetEdge]) -> dict:
    return {"nodes": [node.to_dict() for node in nodes], "edges": [edge.to_dict() for edge in edges]}


def _commit(store: NetStore, op: str, actor: str, before: dict,
            nodes: list[NetNode], edges: list[NetEdge], detail: dict | None = None) -> None:
    store.replace_graph(nodes, _dedupe_edges(edges))
    after = _snapshot(nodes, _dedupe_edges(edges))
    if detail:
        after["detail"] = detail
    store.append_operation(Operation(op_id(op), op, actor, before, after))


def _node(store: NetStore, node_id: str, *types: str) -> NetNode:
    node = next((candidate for candidate in store.nodes() if candidate.id == node_id), None)
    if node is None:
        raise KeyError(f"unknown NET node {node_id}")
    if types and node.type not in types:
        raise NetIntegrityError(f"{node_id} must be one of {types}, got {node.type}")
    return node


def rename_topic(store: NetStore, topic_id: str, label: str, actor: str = "user") -> None:
    _node(store, topic_id, NodeType.TOPIC.value)
    if not label.strip():
        raise ValueError("topic label cannot be empty")
    nodes, edges = store.nodes(), store.edges()
    before = _snapshot(nodes, edges)
    nodes = [replace(node, label=label.strip()) if node.id == topic_id else node for node in nodes]
    _commit(store, "RENAME_TOPIC", actor, before, nodes, edges, {"topic": topic_id, "label": label.strip()})


def move_topic(store: NetStore, topic_id: str, new_parent_id: str, actor: str = "user") -> None:
    _node(store, topic_id, NodeType.TOPIC.value)
    _node(store, new_parent_id, NodeType.TOPIC.value)
    nodes, edges = store.nodes(), store.edges()
    before = _snapshot(nodes, edges)
    edges = [edge for edge in edges
             if not (edge.type == EdgeType.PARENT_OF.value and edge.target == topic_id)]
    edges.append(NetEdge(f"edge:parent:{new_parent_id}:{topic_id}", EdgeType.PARENT_OF.value,
                         new_parent_id, topic_id))
    _commit(store, "MOVE_TOPIC", actor, before, nodes, edges,
            {"topic": topic_id, "parent": new_parent_id})


def merge_topic(store: NetStore, source_topic_id: str, target_topic_id: str,
                actor: str = "user") -> None:
    if source_topic_id == target_topic_id:
        raise ValueError("cannot merge a topic into itself")
    _node(store, source_topic_id, NodeType.TOPIC.value)
    _node(store, target_topic_id, NodeType.TOPIC.value)
    nodes, original_edges = store.nodes(), store.edges()
    before = _snapshot(nodes, original_edges)
    edges: list[NetEdge] = []
    for edge in original_edges:
        source = target_topic_id if edge.source == source_topic_id else edge.source
        target = target_topic_id if edge.target == source_topic_id else edge.target
        if source == target:
            continue
        edges.append(replace(edge, id=_edge_id(edge, source, target), source=source, target=target))
    nodes = [replace(node, state="ARCHIVED", attrs={**node.attrs, "merged_into": target_topic_id})
             if node.id == source_topic_id else node for node in nodes]
    _commit(store, "MERGE_TOPIC", actor, before, nodes, edges,
            {"source": source_topic_id, "target": target_topic_id})
    _sync_concept_membership(store)


def delete_topic(store: NetStore, topic_id: str, actor: str = "user") -> None:
    _set_topic_state(store, topic_id, "ARCHIVED", "DELETE_TOPIC", actor)


def restore_topic(store: NetStore, topic_id: str, actor: str = "user") -> None:
    _set_topic_state(store, topic_id, "ACTIVE", "RESTORE_TOPIC", actor)


def move_document(store: NetStore, document_id: str, topic_or_collection_id: str,
                  actor: str = "user") -> None:
    document_id = document_id if document_id.startswith("document:") else f"document:{document_id}"
    _node(store, document_id, NodeType.DOCUMENT.value)
    _node(store, topic_or_collection_id, NodeType.TOPIC.value, NodeType.COLLECTION.value)
    nodes, edges = store.nodes(), store.edges()
    before = _snapshot(nodes, edges)
    edges = [edge for edge in edges
             if not (edge.type == EdgeType.CONTAINS_DOCUMENT.value and edge.target == document_id)]
    edges.append(NetEdge(f"edge:contains:{topic_or_collection_id}:{document_id}",
                         EdgeType.CONTAINS_DOCUMENT.value, topic_or_collection_id, document_id))
    _commit(store, "MOVE_DOCUMENT", actor, before, nodes, edges,
            {"document": document_id, "parent": topic_or_collection_id})


def create_collection(store: NetStore, collection_id: str, label: str, parent_topic_id: str,
                      collection_type: str | None = None, actor: str = "user") -> None:
    collection_id = collection_id if collection_id.startswith("collection:") else f"collection:{collection_id}"
    if any(node.id == collection_id for node in store.nodes()):
        raise ValueError(f"collection already exists: {collection_id}")
    _node(store, parent_topic_id, NodeType.TOPIC.value)
    nodes, edges = store.nodes(), store.edges()
    before = _snapshot(nodes, edges)
    node = NetNode(collection_id, NodeType.COLLECTION.value, label.strip(), created_by=actor,
                   attrs={"collection_type": collection_type} if collection_type else {})
    nodes.append(node)
    edges.append(NetEdge(f"edge:parent:{parent_topic_id}:{collection_id}",
                         EdgeType.PARENT_OF.value, parent_topic_id, collection_id))
    _commit(store, "CREATE_COLLECTION", actor, before, nodes, edges,
            {"collection": collection_id, "parent": parent_topic_id})


def change_primary_topic(store: NetStore, concept_id: str, topic_id: str,
                         actor: str = "user") -> None:
    _replace_membership(store, concept_id, topic_id, EdgeType.PRIMARY_TOPIC_OF.value,
                        "CHANGE_PRIMARY_TOPIC", actor)


def add_secondary_topic(store: NetStore, concept_id: str, topic_id: str,
                        actor: str = "user") -> None:
    _node(store, concept_id, NodeType.CONCEPT.value)
    _node(store, topic_id, NodeType.TOPIC.value)
    nodes, edges = store.nodes(), store.edges()
    if any(edge.type == EdgeType.PRIMARY_TOPIC_OF.value and edge.source == topic_id
           and edge.target == concept_id for edge in edges):
        raise NetIntegrityError("primary topic cannot also be a secondary topic")
    before = _snapshot(nodes, edges)
    edge = NetEdge(f"edge:secondary:{topic_id}:{concept_id}", EdgeType.SECONDARY_TOPIC_OF.value,
                   topic_id, concept_id)
    edges = [item for item in edges if item.id != edge.id] + [edge]
    _commit(store, "ADD_SECONDARY_TOPIC", actor, before, nodes, edges,
            {"concept": concept_id, "topic": topic_id})
    _sync_concept_membership(store)


def remove_secondary_topic(store: NetStore, concept_id: str, topic_id: str,
                           actor: str = "user") -> None:
    _node(store, concept_id, NodeType.CONCEPT.value)
    _node(store, topic_id, NodeType.TOPIC.value)
    nodes, edges = store.nodes(), store.edges()
    before = _snapshot(nodes, edges)
    edges = [edge for edge in edges if not (
        edge.type == EdgeType.SECONDARY_TOPIC_OF.value
        and edge.source == topic_id and edge.target == concept_id
    )]
    _commit(store, "REMOVE_SECONDARY_TOPIC", actor, before, nodes, edges,
            {"concept": concept_id, "topic": topic_id})
    _sync_concept_membership(store)


def _set_topic_state(store: NetStore, topic_id: str, state: str, op: str, actor: str) -> None:
    _node(store, topic_id, NodeType.TOPIC.value)
    nodes, edges = store.nodes(), store.edges()
    before = _snapshot(nodes, edges)
    nodes = [replace(node, state=state) if node.id == topic_id else node for node in nodes]
    _commit(store, op, actor, before, nodes, edges, {"topic": topic_id, "state": state})


def _replace_membership(store: NetStore, concept_id: str, topic_id: str,
                        edge_type: str, op: str, actor: str) -> None:
    _node(store, concept_id, NodeType.CONCEPT.value)
    _node(store, topic_id, NodeType.TOPIC.value)
    nodes, edges = store.nodes(), store.edges()
    before = _snapshot(nodes, edges)
    edges = [edge for edge in edges if not (edge.type == edge_type and edge.target == concept_id)]
    edges = [edge for edge in edges if not (
        edge.type == EdgeType.SECONDARY_TOPIC_OF.value and edge.source == topic_id
        and edge.target == concept_id
    )]
    edges.append(NetEdge(f"edge:primary:{topic_id}:{concept_id}", edge_type, topic_id, concept_id))
    _commit(store, op, actor, before, nodes, edges, {"concept": concept_id, "topic": topic_id})
    _sync_concept_membership(store)


def _edge_id(edge: NetEdge, source: str, target: str) -> str:
    relation = f":{edge.relation}" if edge.relation else ""
    return f"edge:{edge.type.lower()}:{source}:{target}{relation}"


def _dedupe_edges(edges: list[NetEdge]) -> list[NetEdge]:
    deduped: dict[tuple, NetEdge] = {}
    for edge in edges:
        key = (edge.type, edge.source, edge.target, edge.relation)
        deduped[key] = edge
    return sorted(deduped.values(), key=lambda edge: edge.id)


def _sync_concept_membership(store: NetStore) -> None:
    from llm_wiki.v2.concept_store import read_concepts, write_concepts

    concepts = read_concepts(store.vault)
    if not concepts:
        return
    primary = {edge.target: edge.source for edge in store.edges()
               if edge.type == EdgeType.PRIMARY_TOPIC_OF.value}
    secondary: dict[str, list[str]] = {}
    for edge in store.edges():
        if edge.type == EdgeType.SECONDARY_TOPIC_OF.value:
            secondary.setdefault(edge.target, []).append(edge.source)
    write_concepts([replace(concept, primary_topic_id=primary.get(concept.id),
                            secondary_topic_ids=sorted(secondary.get(concept.id, [])))
                    for concept in concepts], store.vault)


def undo_last(store: NetStore, actor: str = "user") -> Operation:
    """Restore the exact graph snapshot for the latest not-yet-undone user operation."""
    operations = store.operations()
    undone = {operation.after.get("undone_operation_id") for operation in operations
              if operation.op == "UNDO"}
    operation = next((candidate for candidate in reversed(operations)
                      if candidate.op != "UNDO" and candidate.actor != "system"
                      and candidate.id not in undone), None)
    if operation is None:
        raise KeyError("no reversible user operation in the operation log")
    rows = operation.before
    if not isinstance(rows.get("nodes"), list) or not isinstance(rows.get("edges"), list):
        raise NetIntegrityError(f"operation {operation.id} has no complete undo snapshot")
    nodes = [NetNode.from_dict(row) for row in rows["nodes"]]
    edges = [NetEdge.from_dict(row) for row in rows["edges"]]
    store.replace_graph(nodes, edges)
    undo = Operation(op_id("UNDO"), "UNDO", actor, _snapshot(store.nodes(), store.edges()),
                     {"undone_operation_id": operation.id})
    store.append_operation(undo)
    _sync_concept_membership(store)
    return operation
