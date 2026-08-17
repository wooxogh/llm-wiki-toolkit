from pathlib import Path

from llm_wiki.v2.models import Concept, RelationProposal
from llm_wiki.v2.schemas import RelationType
from llm_wiki.v2 import relation_classifier, temporal


def _vault(tmp_path: Path) -> Path:
    (tmp_path / "wiki.toml").write_text(
        '[vault]\ncontent_dirs = ["domain"]\n\n[v2]\nenabled = true\n', encoding="utf-8")
    (tmp_path / "domain").mkdir()
    return tmp_path


def _concept(cid: str, text: str) -> Concept:
    return Concept(
        id=cid, document_id="doc", chunk_id="doc:chunk:0001:aaa", text=text,
        summary=text, source_quote=text, confidence=0.9, chunk_hash="hash",
        source_start=0, source_end=len(text),
    )


class _CountingClassifyAdapter:
    model_identity = "test-model"

    def __init__(self, proposal):
        self.proposal = proposal
        self.calls = 0

    def classify_relation(self, source, target):
        self.calls += 1
        return self.proposal


class _CountingTemporalAdapter:
    model_identity = "test-model"

    def __init__(self, proposal):
        self.proposal = proposal
        self.calls = 0

    def resolve_temporal(self, source, target):
        self.calls += 1
        return self.proposal


def test_classify_relation_is_cached_across_calls(tmp_path):
    vault = _vault(tmp_path)
    source, target = _concept("concept:a", "Frontend uses React"), _concept("concept:b", "Frontend used Vue")
    proposal = RelationProposal(
        id="proposal:a:SUPPORTS:b", source_concept_id=source.id, target_concept_id=target.id,
        relation=RelationType.SUPPORTS.value, confidence=0.9, evidence=source.source_quote,
    )
    adapter = _CountingClassifyAdapter(proposal)
    first = relation_classifier.classify(adapter, source, target, vault=vault)
    second = relation_classifier.classify(adapter, source, target, vault=vault)
    assert first == second
    assert first.relation == RelationType.SUPPORTS.value
    assert adapter.calls == 1


def test_resolve_temporal_is_cached_across_calls(tmp_path):
    vault = _vault(tmp_path)
    source, target = _concept("concept:a", "Frontend no longer uses Vue"), _concept("concept:b", "Frontend uses Vue")
    proposal = RelationProposal(
        id="proposal:a:SUPERSEDES:b", source_concept_id=source.id, target_concept_id=target.id,
        relation=RelationType.SUPERSEDES.value, confidence=0.9, evidence=source.source_quote,
        same_subject=True, same_scope=True, temporal_change_possible=True, reason="explicit replacement",
    )
    relation = RelationProposal(
        id="proposal:a:SUPERSEDES:b", source_concept_id=source.id, target_concept_id=target.id,
        relation=RelationType.SUPERSEDES.value, confidence=0.9, evidence=source.source_quote,
    )
    adapter = _CountingTemporalAdapter(proposal)
    first = temporal.resolve(adapter, source, target, relation=relation, vault=vault)
    second = temporal.resolve(adapter, source, target, relation=relation, vault=vault)
    assert first == second
    assert first.relation == RelationType.SUPERSEDES.value
    assert adapter.calls == 1
