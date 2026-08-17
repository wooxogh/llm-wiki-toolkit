"""Proposal review and approved relation commit logic."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from llm_wiki.v2.concept_store import read_concepts, write_concepts
from llm_wiki.v2.models import NetEdge, RelationProposal, ReviewItem, utc_now
from llm_wiki.v2.net_store import NetStore
from llm_wiki.v2.schemas import ConceptState, RelationType, RISKY_RELATIONS, SAFE_RELATIONS


def submit_relation_proposal(
    store: NetStore,
    proposal: RelationProposal,
    safe_min_confidence: float = 0.90,
    require_user_approval: tuple[str, ...] = tuple(r.value for r in RISKY_RELATIONS),
) -> str:
    relation = RelationType(proposal.relation)
    _validate_for_persistence(store, proposal)
    requires = relation.value in require_user_approval
    existing = next((item for item in store.proposals() if item.id == proposal.id), None)
    if existing and existing.status in {"APPROVED", "REJECTED"}:
        return existing.status.lower()
    proposal = replace(proposal, requires_approval=requires)
    store.append_proposal(proposal)
    if requires or relation not in SAFE_RELATIONS or proposal.confidence < safe_min_confidence:
        store.append_review_item(ReviewItem(
            id=f"review:{proposal.id}",
            kind="RELATION",
            proposal_id=proposal.id,
            reason="requires user approval" if requires else "below safe confidence threshold",
        ))
        return "review"
    approved = replace(proposal, status="APPROVED", approved_by="policy", approved_at=utc_now())
    store.append_proposal(approved)
    store.upsert_edge(NetEdge.relation_edge(approved))
    return "committed"


def approve_review_item(store: NetStore, review_id: str, actor: str = "user", vault: Path | None = None) -> None:
    items = store.review_items()
    item = next((i for i in items if i.id == review_id), None)
    if not item:
        raise KeyError(f"unknown review item {review_id}")
    if item.state != "OPEN":
        raise ValueError(f"review item {review_id} is {item.state}, not OPEN")
    proposal = next((p for p in store.proposals() if p.id == item.proposal_id), None)
    if not proposal:
        raise KeyError(f"unknown proposal {item.proposal_id}")
    _validate_for_persistence(store, proposal)
    approved = replace(proposal, status="APPROVED", approved_by=actor, approved_at=utc_now())
    store.append_proposal(approved)
    store.upsert_edge(NetEdge.relation_edge(approved))
    _apply_lifecycle(store, approved, vault)
    store.write_review_items([replace(i, state="APPROVED") if i.id == review_id else i for i in items])


def reject_review_item(store: NetStore, review_id: str, actor: str = "user") -> None:
    items = store.review_items()
    item = next((i for i in items if i.id == review_id), None)
    if not item:
        raise KeyError(f"unknown review item {review_id}")
    if item.state != "OPEN":
        raise ValueError(f"review item {review_id} is {item.state}, not OPEN")
    proposals = [replace(p, status="REJECTED") if p.id == item.proposal_id else p for p in store.proposals()]
    store.write_proposals(proposals)
    store.write_review_items([replace(i, state="REJECTED") if i.id == review_id else i for i in items])


def resolve_review_item(store: NetStore, review_id: str, decision: str,
                        actor: str = "user", vault: Path | None = None) -> None:
    """Apply the explicit contradiction-review choices from the v2 contract."""
    if decision in {"different-scope", "unrelated"}:
        reject_review_item(store, review_id, actor)
        return
    item = next((candidate for candidate in store.review_items() if candidate.id == review_id), None)
    if not item or item.state != "OPEN":
        raise ValueError(f"review item {review_id} is not OPEN")
    proposal = next((candidate for candidate in store.proposals() if candidate.id == item.proposal_id), None)
    if not proposal:
        raise KeyError(f"unknown proposal {item.proposal_id}")
    if decision == "target-current":
        proposal = replace(proposal, source_concept_id=proposal.target_concept_id,
                           target_concept_id=proposal.source_concept_id,
                           relation=RelationType.OVERRIDES.value, same_subject=True, same_scope=True,
                           temporal_change_possible=False, reason="user selected target as current")
    elif decision == "source-current":
        proposal = replace(proposal, relation=RelationType.OVERRIDES.value,
                           same_subject=True, same_scope=True, temporal_change_possible=False,
                           reason="user selected source as current")
    elif decision == "disputed":
        proposal = replace(proposal, relation=RelationType.CONTRADICTS.value,
                           same_subject=True, same_scope=True, temporal_change_possible=False,
                           reason="user confirmed an unresolved contradiction")
    else:
        raise ValueError(f"unknown review decision {decision}")
    store.append_proposal(proposal)
    approve_review_item(store, review_id, actor, vault)


def _apply_lifecycle(store: NetStore, proposal: RelationProposal, vault: Path | None) -> None:
    relation = RelationType(proposal.relation)
    concepts = read_concepts(vault)
    changed = []
    for concept in concepts:
        if relation == RelationType.SUPERSEDES and concept.id == proposal.target_concept_id:
            changed.append(replace(concept, state=ConceptState.SUPERSEDED.value))
            store.update_concept_state(concept.id, ConceptState.SUPERSEDED.value)
        elif relation == RelationType.CONTRADICTS and concept.id in {proposal.source_concept_id, proposal.target_concept_id}:
            changed.append(replace(concept, state=ConceptState.DISPUTED.value))
            store.update_concept_state(concept.id, ConceptState.DISPUTED.value)
        elif relation == RelationType.DUPLICATE_OF and concept.id == proposal.source_concept_id:
            changed.append(replace(concept, state=ConceptState.DUPLICATE.value))
            store.update_concept_state(concept.id, ConceptState.DUPLICATE.value)
        else:
            changed.append(concept)
    if concepts:
        write_concepts(changed, vault)


def _validate_for_persistence(store: NetStore, proposal: RelationProposal) -> None:
    relation = RelationType(proposal.relation)
    if proposal.source_concept_id == proposal.target_concept_id:
        raise ValueError("relation proposal cannot target itself")
    nodes = {node.id: node for node in store.nodes()}
    if proposal.source_concept_id not in nodes or proposal.target_concept_id not in nodes:
        raise ValueError("relation proposal has a missing Concept endpoint")
    if not 0 <= proposal.confidence <= 1 or not proposal.evidence.strip():
        raise ValueError("relation proposal needs confidence in [0,1] and evidence")
    concepts = {concept.id: concept for concept in read_concepts(store.vault)}
    source = concepts.get(proposal.source_concept_id)
    target = concepts.get(proposal.target_concept_id)
    if source and target and proposal.evidence not in "\n".join(
            (source.text, source.source_quote, target.text, target.source_quote)):
        raise ValueError("relation proposal evidence must quote one of its source Concepts")
    if relation in RISKY_RELATIONS:
        if proposal.same_subject is not True or proposal.same_scope is not True:
            raise ValueError(f"{relation.value} requires same_subject=true and same_scope=true")
        if not proposal.reason.strip():
            raise ValueError(f"{relation.value} requires a rationale")
    if relation == RelationType.SUPERSEDES:
        if proposal.temporal_change_possible is not True:
            raise ValueError("SUPERSEDES requires temporal_change_possible=true")
        if not source or not target:
            raise ValueError("SUPERSEDES requires persisted source and target Concepts")
        from llm_wiki.v2.temporal import _has_revision_evidence, _source_is_newer
        if not (_source_is_newer(source, target) or _has_revision_evidence(source, proposal)):
            raise ValueError("SUPERSEDES requires newer metadata or explicit revision evidence")
