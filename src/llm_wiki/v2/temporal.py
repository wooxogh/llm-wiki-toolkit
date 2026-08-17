"""Turn plausible relation pairs into temporal proposals without mutating state."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime

from llm_wiki.v2.llm_adapter import UserLLMAdapter
from llm_wiki.v2.models import Concept, RelationProposal
from llm_wiki.v2 import relation_cache
from llm_wiki.v2.schemas import TEMPORAL_PROMPT_VERSION, RelationType


def resolve(adapter: UserLLMAdapter, source: Concept, target: Concept,
            relation: RelationProposal | None = None, vault=None) -> RelationProposal | None:
    if relation is None or relation.relation not in {RelationType.CONTRADICTS.value, RelationType.SUPERSEDES.value, RelationType.OVERRIDES.value}:
        return None
    return relation_cache.cached_call(
        "resolve", source, target, TEMPORAL_PROMPT_VERSION,
        getattr(adapter, "model_identity", "offline"),
        lambda: _resolve_uncached(adapter, source, target), vault,
    )


def _resolve_uncached(adapter: UserLLMAdapter, source: Concept, target: Concept) -> RelationProposal | None:
    proposal = adapter.resolve_temporal(source, target)
    if proposal is None:
        return None
    if proposal.source_concept_id != source.id or proposal.target_concept_id != target.id:
        return None
    if proposal.relation not in {RelationType.SUPERSEDES.value, RelationType.OVERRIDES.value}:
        return None
    if not 0 <= proposal.confidence <= 1:
        return None
    if proposal.same_subject is not True or proposal.same_scope is not True:
        return None
    if not proposal.reason.strip() or not proposal.evidence:
        return None
    evidence_space = "\n".join((source.text, source.source_quote, target.text, target.source_quote))
    if proposal.evidence not in evidence_space:
        return None
    if proposal.relation == RelationType.SUPERSEDES.value:
        if proposal.temporal_change_possible is not True:
            return None
        if not (_source_is_newer(source, target) or _has_revision_evidence(source, proposal)):
            return None
    return replace(proposal, prompt_version=TEMPORAL_PROMPT_VERSION)


def _source_is_newer(source: Concept, target: Concept) -> bool:
    if not source.updated_at or not target.updated_at:
        return False
    try:
        return datetime.fromisoformat(source.updated_at.replace("Z", "+00:00")) > datetime.fromisoformat(
            target.updated_at.replace("Z", "+00:00")
        )
    except ValueError:
        return False


def _has_revision_evidence(source: Concept, proposal: RelationProposal) -> bool:
    text = f"{source.text} {source.source_quote} {proposal.evidence}".lower()
    markers = ("replaces", "supersedes", "no longer", "deprecated", "version", "revision",
               "대체", "변경", "개정", "최신", "더 이상", "버전")
    return any(marker in text for marker in markers)
