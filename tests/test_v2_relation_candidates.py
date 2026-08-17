from pathlib import Path

from llm_wiki.v2 import net_builder, relation_candidates
from llm_wiki.v2.models import Concept


def _concept(cid: str) -> Concept:
    return Concept(
        id=cid, document_id="doc", chunk_id="doc:chunk:0001:aaa", text=cid,
        summary=cid, source_quote=cid, confidence=0.9, chunk_hash="hash",
        source_start=0, source_end=len(cid),
    )


def test_discover_returns_score_alongside_each_candidate(monkeypatch, tmp_path: Path):
    seed, high, low = _concept("concept:seed"), _concept("concept:high"), _concept("concept:low")
    monkeypatch.setattr(
        relation_candidates.concept_index, "search",
        lambda vault, text, k, concepts: [(0.05, high), (0.001, low)],
    )
    result = relation_candidates.discover(tmp_path, seed, [seed, high, low], top_k=10)
    assert result[0][0] >= result[1][0]
    assert {concept.id for _, concept in result} == {"concept:high", "concept:low"}


def test_candidate_pairs_drops_scores_below_the_configured_floor(monkeypatch):
    seed, high, low = _concept("concept:seed"), _concept("concept:high"), _concept("concept:low")
    monkeypatch.setattr(
        net_builder, "discover",
        lambda vault, source, concepts, top_k: [(0.05, high), (0.001, low)],
    )
    pairs = net_builder._candidate_pairs(None, [seed], topk=10, min_score=0.01)
    assert pairs == [(seed, high)]
