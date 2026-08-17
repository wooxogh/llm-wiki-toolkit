"""Semantic classification is an adapter proposal, policy is applied elsewhere."""
from __future__ import annotations

from dataclasses import replace

from llm_wiki.v2.llm_adapter import UserLLMAdapter
from llm_wiki.v2.models import Concept, RelationProposal
from llm_wiki.v2.schemas import RELATION_PROMPT_VERSION, RISKY_RELATIONS, RelationType


def classify(adapter: UserLLMAdapter, source: Concept, target: Concept) -> RelationProposal | None:
    proposal = adapter.classify_relation(source, target)
    if proposal is None:
        return None
    if proposal.source_concept_id != source.id or proposal.target_concept_id != target.id:
        return None
    try:
        RelationType(proposal.relation)
    except ValueError:
        return None
    if not 0 <= proposal.confidence <= 1:
        return None
    if not proposal.evidence or proposal.evidence not in (source.text + "\n" + source.source_quote + "\n" +
                                                           target.text + "\n" + target.source_quote):
        return None
    if RelationType(proposal.relation) in RISKY_RELATIONS and (
        proposal.same_subject is None or proposal.same_scope is None
        or proposal.temporal_change_possible is None or not proposal.reason.strip()
    ):
        return None
    return replace(proposal, prompt_version=RELATION_PROMPT_VERSION)
