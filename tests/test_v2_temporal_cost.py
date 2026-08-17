from pathlib import Path

from llm_wiki.v2.models import Concept, RelationProposal
from llm_wiki.v2.schemas import RelationType
from llm_wiki.v2 import temporal


def _vault(tmp_path: Path) -> Path:
    (tmp_path / "wiki.toml").write_text(
        '[vault]\ncontent_dirs = ["domain"]\n\n[v2]\nenabled = true\n', encoding="utf-8")
    (tmp_path / "domain").mkdir()
    return tmp_path


def _concept(cid: str) -> Concept:
    return Concept(
        id=cid, document_id="doc", chunk_id="doc:chunk:0001:aaa", text=f"{cid} text",
        summary=cid, source_quote=f"{cid} text", confidence=0.9, chunk_hash="hash",
        source_start=0, source_end=10,
    )


class _CountingAdapter:
    def __init__(self):
        self.calls = 0

    def resolve_temporal(self, source, target):
        self.calls += 1
        return None


def test_resolve_skips_the_llm_when_no_relation_was_classified(tmp_path):
    vault = _vault(tmp_path)
    adapter = _CountingAdapter()
    result = temporal.resolve(adapter, _concept("concept:a"), _concept("concept:b"), relation=None, vault=vault)
    assert result is None
    assert adapter.calls == 0


def test_resolve_still_calls_the_llm_for_a_plausible_relation(tmp_path):
    vault = _vault(tmp_path)
    adapter = _CountingAdapter()
    relation = RelationProposal(
        id="proposal:a:CONTRADICTS:b", source_concept_id="concept:a", target_concept_id="concept:b",
        relation=RelationType.CONTRADICTS.value, confidence=0.8, evidence="a text",
    )
    result = temporal.resolve(adapter, _concept("concept:a"), _concept("concept:b"), relation=relation, vault=vault)
    assert result is None  # adapter returns None here, only the CALL matters
    assert adapter.calls == 1
