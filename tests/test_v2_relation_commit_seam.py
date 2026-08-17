from pathlib import Path

from llm_wiki.v2 import net_builder
from llm_wiki.v2.models import Concept, RelationProposal
from llm_wiki.v2.schemas import RelationType


def _concept(cid: str, text: str) -> Concept:
    return Concept(
        id=cid, document_id="doc", chunk_id="doc:chunk:0001:aaa", text=text,
        summary=text, source_quote=text, confidence=0.9, chunk_hash="hash",
        source_start=0, source_end=len(text),
    )


class _StubAdapter:
    model_identity = "stub"

    def __init__(self, relation_by_pair):
        self.relation_by_pair = relation_by_pair

    def classify_relation(self, source, target):
        return self.relation_by_pair.get((source.id, target.id))

    def resolve_temporal(self, source, target):
        return None


def test_classify_pair_never_touches_a_store(tmp_path: Path):
    source, target = _concept("concept:a", "Frontend uses React"), _concept("concept:b", "Frontend used Vue")
    proposal = RelationProposal(
        id="proposal:a:SUPPORTS:b", source_concept_id=source.id, target_concept_id=target.id,
        relation=RelationType.SUPPORTS.value, confidence=0.9, evidence=source.source_quote,
    )
    adapter = _StubAdapter({(source.id, target.id): proposal})
    result_source, result_target, proposals = net_builder._classify_pair(adapter, tmp_path, source, target)
    assert result_source is source and result_target is target
    assert [p.relation for p in proposals] == [RelationType.SUPPORTS.value]


def test_compute_relation_proposals_covers_every_pair(tmp_path: Path):
    a, b, c = _concept("concept:a", "A"), _concept("concept:b", "B"), _concept("concept:c", "C")
    adapter = _StubAdapter({})
    results = list(net_builder._compute_relation_proposals([(a, b), (b, c)], adapter, tmp_path))
    seen_pairs = {(source.id, target.id) for source, target, _ in results}
    assert seen_pairs == {("concept:a", "concept:b"), ("concept:b", "concept:c")}
