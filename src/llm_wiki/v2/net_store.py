"""Committed NET store and integrity checks."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from llm_wiki.v2 import artifacts
from llm_wiki.v2.models import NetEdge, NetNode, Operation, RelationProposal, ReviewItem
from llm_wiki.v2.schemas import EdgeType, NodeType, RISKY_RELATIONS, RelationType


class NetIntegrityError(ValueError):
    pass


class NetStore:
    def __init__(self, vault: Path | None = None):
        self.vault = vault
        artifacts.ensure_layout(vault)

    @property
    def root(self) -> Path:
        return artifacts.artifact_path("net", self.vault)

    def nodes(self) -> list[NetNode]:
        return artifacts.read_jsonl(self.root / "nodes.jsonl", NetNode.from_dict)

    def edges(self) -> list[NetEdge]:
        return artifacts.read_jsonl(self.root / "edges.jsonl", NetEdge.from_dict)

    def proposals(self) -> list[RelationProposal]:
        return artifacts.read_jsonl(self.root / "proposals.jsonl", RelationProposal.from_dict)

    def review_items(self) -> list[ReviewItem]:
        return artifacts.read_jsonl(self.root / "review_queue.jsonl", ReviewItem.from_dict)

    def operations(self) -> list[Operation]:
        return artifacts.read_jsonl(self.root / "operations.jsonl", Operation.from_dict)

    def write_nodes(self, nodes: list[NetNode]) -> None:
        artifacts.write_jsonl(self.root / "nodes.jsonl", nodes)

    def write_edges(self, edges: list[NetEdge]) -> None:
        artifacts.write_jsonl(self.root / "edges.jsonl", edges)

    def replace_graph(self, nodes: list[NetNode], edges: list[NetEdge]) -> None:
        """Validate a complete graph in memory, then replace it atomically enough for local files."""
        by_id = {node.id: node for node in nodes}
        if len(by_id) != len(nodes):
            raise NetIntegrityError("duplicate NET node id")
        validated: list[NetEdge] = []
        seen_ids: set[str] = set()
        for edge in edges:
            if edge.id in seen_ids:
                raise NetIntegrityError(f"duplicate NET edge id {edge.id}")
            if edge.source not in by_id or edge.target not in by_id:
                raise NetIntegrityError(f"edge {edge.id} has dangling endpoint")
            self._validate_edge(edge, validated, by_id)
            validated.append(edge)
            seen_ids.add(edge.id)
        self.write_nodes(sorted(nodes, key=lambda node: node.id))
        self.write_edges(sorted(edges, key=lambda edge: edge.id))

    def write_proposals(self, proposals: list[RelationProposal]) -> None:
        artifacts.write_jsonl(self.root / "proposals.jsonl", proposals)

    def write_review_items(self, items: list[ReviewItem]) -> None:
        artifacts.write_jsonl(self.root / "review_queue.jsonl", items)

    def write_operations(self, operations: list[Operation]) -> None:
        artifacts.write_jsonl(self.root / "operations.jsonl", operations)

    def upsert_node(self, node: NetNode) -> None:
        nodes = [n for n in self.nodes() if n.id != node.id]
        nodes.append(node)
        self.write_nodes(sorted(nodes, key=lambda n: n.id))

    def upsert_edge(self, edge: NetEdge) -> None:
        nodes = {n.id: n for n in self.nodes()}
        if edge.source not in nodes or edge.target not in nodes:
            raise NetIntegrityError(f"edge {edge.id} has dangling endpoint")
        edges = [e for e in self.edges() if e.id != edge.id]
        self._validate_edge(edge, edges, nodes)
        edges.append(edge)
        self.write_edges(sorted(edges, key=lambda e: e.id))

    def update_concept_state(self, concept_id: str, state: str) -> None:
        nodes = []
        changed = False
        for node in self.nodes():
            if node.id == concept_id:
                attrs = dict(node.attrs)
                attrs["concept_state"] = state
                nodes.append(replace(node, attrs=attrs))
                changed = True
            else:
                nodes.append(node)
        if changed:
            self.write_nodes(nodes)

    def append_proposal(self, proposal: RelationProposal) -> None:
        proposals = [p for p in self.proposals() if p.id != proposal.id]
        proposals.append(proposal)
        self.write_proposals(sorted(proposals, key=lambda p: p.id))

    def append_review_item(self, item: ReviewItem) -> None:
        items = [i for i in self.review_items() if i.id != item.id]
        items.append(item)
        self.write_review_items(sorted(items, key=lambda i: i.id))

    def append_operation(self, operation: Operation) -> None:
        operations = self.operations()
        operations.append(operation)
        self.write_operations(operations)

    def _validate_edge(self, edge: NetEdge, existing: list[NetEdge], nodes: dict[str, NetNode]) -> None:
        if edge.type == EdgeType.PARENT_OF.value:
            if nodes[edge.source].type != NodeType.TOPIC.value:
                raise NetIntegrityError("PARENT_OF source must be TOPIC")
            if nodes[edge.target].type not in (NodeType.TOPIC.value, NodeType.COLLECTION.value):
                raise NetIntegrityError("PARENT_OF target must be TOPIC or COLLECTION")
            if _would_cycle(existing + [edge], edge.source, edge.target):
                raise NetIntegrityError("PARENT_OF would create a cycle")
            if any(e.type == EdgeType.PARENT_OF.value and e.target == edge.target for e in existing):
                raise NetIntegrityError("tree node may have only one parent")
        if edge.type == EdgeType.CONTAINS_DOCUMENT.value:
            if nodes[edge.source].type not in (NodeType.TOPIC.value, NodeType.COLLECTION.value):
                raise NetIntegrityError("CONTAINS_DOCUMENT source must be TOPIC or COLLECTION")
            if nodes[edge.target].type != NodeType.DOCUMENT.value:
                raise NetIntegrityError("CONTAINS_DOCUMENT target must be DOCUMENT")
            if any(e.type == EdgeType.CONTAINS_DOCUMENT.value and e.target == edge.target for e in existing):
                raise NetIntegrityError("document may have only one Topic/Collection location")
        if edge.type == EdgeType.DOCUMENT_HAS_CONCEPT.value:
            if nodes[edge.source].type != NodeType.DOCUMENT.value or nodes[edge.target].type != NodeType.CONCEPT.value:
                raise NetIntegrityError("DOCUMENT_HAS_CONCEPT must connect DOCUMENT to CONCEPT")
        if edge.type in (EdgeType.PRIMARY_TOPIC_OF.value, EdgeType.SECONDARY_TOPIC_OF.value):
            if nodes[edge.source].type != NodeType.TOPIC.value or nodes[edge.target].type != NodeType.CONCEPT.value:
                raise NetIntegrityError("concept membership must connect TOPIC to CONCEPT")
            if edge.type == EdgeType.PRIMARY_TOPIC_OF.value and any(
                    e.type == EdgeType.PRIMARY_TOPIC_OF.value and e.target == edge.target for e in existing):
                raise NetIntegrityError("concept may have only one primary topic")
            opposite = (EdgeType.SECONDARY_TOPIC_OF.value if edge.type == EdgeType.PRIMARY_TOPIC_OF.value
                        else EdgeType.PRIMARY_TOPIC_OF.value)
            if any(e.type == opposite and e.source == edge.source and e.target == edge.target for e in existing):
                raise NetIntegrityError("primary topic cannot also be a secondary topic")
            if any(e.type == edge.type and e.source == edge.source and e.target == edge.target for e in existing):
                raise NetIntegrityError("duplicate concept membership")
        if edge.type == EdgeType.RELATES_TO.value:
            if nodes[edge.source].type != NodeType.CONCEPT.value or nodes[edge.target].type != NodeType.CONCEPT.value:
                raise NetIntegrityError("semantic relation must connect Concept nodes")
            if edge.source == edge.target:
                raise NetIntegrityError("semantic relation cannot be a self-edge")
            relation = RelationType(edge.relation or "")
            if relation in RISKY_RELATIONS and not edge.approved_by:
                raise NetIntegrityError(f"{relation.value} requires user approval")

    def health_issues(self) -> list[str]:
        issues: list[str] = []
        nodes = {n.id: n for n in self.nodes()}
        edges = self.edges()
        proposals = {f"edge:{proposal.id}": proposal for proposal in self.proposals()}
        for edge in edges:
            if edge.source not in nodes or edge.target not in nodes:
                issues.append(f"edge {edge.id} has dangling endpoint")
            if edge.type == EdgeType.RELATES_TO.value and edge.relation in {r.value for r in RISKY_RELATIONS}:
                if not edge.approved_by:
                    issues.append(f"risky relation {edge.id} committed without approval")
                proposal = proposals.get(edge.id)
                if not proposal or proposal.status != "APPROVED" or not proposal.approved_by:
                    issues.append(f"risky relation {edge.id} has no approved proposal trace")
            if edge.type == EdgeType.RELATES_TO.value and edge.source == edge.target:
                issues.append(f"relation {edge.id} is a self-edge")
        for edge in edges:
            if edge.type == EdgeType.PARENT_OF.value and _would_cycle([e for e in edges if e.id != edge.id] + [edge], edge.source, edge.target):
                issues.append("topic hierarchy contains a cycle")
                break
        parent_counts: dict[str, int] = {}
        for edge in edges:
            if edge.type == EdgeType.PARENT_OF.value:
                parent_counts[edge.target] = parent_counts.get(edge.target, 0) + 1
        for node_id, count in parent_counts.items():
            if count > 1:
                issues.append(f"tree node {node_id} has {count} parents")
        for node in nodes.values():
            if (node.type in {NodeType.TOPIC.value, NodeType.COLLECTION.value}
                    and node.id != "topic:knowledge" and node.state == "ACTIVE"
                    and node.id not in parent_counts):
                issues.append(f"active tree node {node.id} has no parent")
        if _relation_cycle(edges, RelationType.SUPERSEDES.value):
            issues.append("SUPERSEDES relation contains a cycle")
        superseded = [n for n in nodes.values()
                      if n.type == NodeType.CONCEPT.value and n.attrs.get("concept_state") == "SUPERSEDED"]
        successors = {e.target for e in edges if e.type == EdgeType.RELATES_TO.value and e.relation == RelationType.SUPERSEDES.value}
        for node in superseded:
            if node.id not in successors:
                issues.append(f"SUPERSEDED concept {node.id} has no successor")
        disputed = [n for n in nodes.values()
                    if n.type == NodeType.CONCEPT.value and n.attrs.get("concept_state") == "DISPUTED"]
        disputed_targets = {e.source for e in edges if e.type == EdgeType.RELATES_TO.value and e.relation == RelationType.CONTRADICTS.value}
        disputed_targets |= {e.target for e in edges if e.type == EdgeType.RELATES_TO.value and e.relation == RelationType.CONTRADICTS.value}
        for node in disputed:
            if node.id not in disputed_targets and not node.attrs.get("dispute_reason"):
                issues.append(f"DISPUTED concept {node.id} has no contradiction or reason")
        if not (self.root / "operations.jsonl").exists():
            issues.append("operation log missing")
        primary: dict[str, int] = {}
        for edge in edges:
            if edge.type == EdgeType.PRIMARY_TOPIC_OF.value:
                primary[edge.target] = primary.get(edge.target, 0) + 1
        for concept_id, count in primary.items():
            if count != 1:
                issues.append(f"concept {concept_id} has {count} primary topics")
        for node in nodes.values():
            if node.type == NodeType.CONCEPT.value and node.id not in primary:
                issues.append(f"concept {node.id} has no primary topic")
        contains: dict[str, int] = {}
        for edge in edges:
            if edge.type == EdgeType.CONTAINS_DOCUMENT.value:
                contains[edge.target] = contains.get(edge.target, 0) + 1
        for node in nodes.values():
            if node.type == NodeType.DOCUMENT.value:
                count = contains.get(node.id, 0)
                if count != 1:
                    issues.append(f"document {node.id} has {count} tree locations; expected exactly one")
        operation_ids = {operation.id for operation in self.operations()}
        for operation in self.operations():
            if operation.op == "UNDO" and operation.after.get("undone_operation_id") not in operation_ids:
                issues.append(f"undo operation {operation.id} has missing target")
        return issues


def _would_cycle(edges: list[NetEdge], source: str, target: str) -> bool:
    children: dict[str, list[str]] = {}
    for edge in edges:
        if edge.type == EdgeType.PARENT_OF.value:
            children.setdefault(edge.source, []).append(edge.target)
    seen = set()

    def visit(node: str) -> bool:
        if node == source:
            return True
        if node in seen:
            return False
        seen.add(node)
        return any(visit(child) for child in children.get(node, []))

    return visit(target)


def _relation_cycle(edges: list[NetEdge], relation: str) -> bool:
    graph: dict[str, list[str]] = {}
    for edge in edges:
        if edge.type == EdgeType.RELATES_TO.value and edge.relation == relation:
            graph.setdefault(edge.source, []).append(edge.target)
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        for child in graph.get(node, []):
            if visit(child):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node) for node in graph)
