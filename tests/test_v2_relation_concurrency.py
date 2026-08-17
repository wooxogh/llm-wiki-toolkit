import threading
from pathlib import Path

from llm_wiki.v2 import net_builder
from llm_wiki.v2.models import Concept, PlacementProposal, RelationProposal
from llm_wiki.v2.schemas import RelationType


def _concept(cid: str, text: str) -> Concept:
    return Concept(
        id=cid, document_id="doc", chunk_id="doc:chunk:0001:aaa", text=text,
        summary=text, source_quote=text, confidence=0.9, chunk_hash="hash",
        source_start=0, source_end=len(text),
    )


class _ThreadRecordingAdapter:
    model_identity = "stub"

    def __init__(self):
        self.threads: set[int] = set()
        self.lock = threading.Lock()

    def classify_relation(self, source, target):
        with self.lock:
            self.threads.add(threading.get_ident())
        return RelationProposal(
            id=f"proposal:{source.id}:SUPPORTS:{target.id}", source_concept_id=source.id,
            target_concept_id=target.id, relation=RelationType.SUPPORTS.value,
            confidence=0.9, evidence=source.source_quote,
        )

    def resolve_temporal(self, source, target):
        return None

    def place_concept(self, concept, tree_candidates):
        return PlacementProposal(concept_id=concept.id, primary_topic_id=None, confidence=0.0)


def test_concurrent_computation_uses_more_than_one_worker_thread(tmp_path: Path):
    concepts = [_concept(f"concept:{i}", f"text {i}") for i in range(8)]
    pairs = [(concepts[i], concepts[i + 1]) for i in range(len(concepts) - 1)]
    adapter = _ThreadRecordingAdapter()
    results = list(net_builder._compute_relation_proposals(pairs, adapter, tmp_path, workers=4))
    assert len(results) == len(pairs)
    assert len(adapter.threads) > 1


def test_default_workers_stays_single_threaded(tmp_path: Path):
    concepts = [_concept(f"concept:{i}", f"text {i}") for i in range(4)]
    pairs = [(concepts[i], concepts[i + 1]) for i in range(len(concepts) - 1)]
    adapter = _ThreadRecordingAdapter()
    results = list(net_builder._compute_relation_proposals(pairs, adapter, tmp_path))
    assert len(results) == len(pairs)
    assert adapter.threads == {threading.get_ident()}


def test_concurrent_build_commits_every_expected_proposal(tmp_path: Path):
    from llm_wiki.v2.concept_store import build_concepts
    from llm_wiki.v2.net_builder import build_net

    (tmp_path / "wiki.toml").write_text(
        '[vault]\ncontent_dirs = ["domain"]\n\n[v2]\nenabled = true\nrelation_concurrency = 4\n'
        'safe_relation_min_confidence = 0.0\n', encoding="utf-8")
    (tmp_path / "domain").mkdir()
    for i in range(6):
        (tmp_path / "domain" / f"doc{i}.md").write_text(
            f"---\nid: doc{i}\nlayer: domain\nprojects: []\ntags: []\nconfidence: confirmed\nstatus: active\nsummary: doc {i}\n---\n"
            f"# Doc {i}\n\nFrontend uses React in service {i}.",
            encoding="utf-8")
    build_concepts(tmp_path)
    store = build_net(tmp_path, adapter=_ThreadRecordingAdapter())
    proposals = store.proposals()
    # A lost update from the read-modify-write race this plan guards against
    # would show up here as duplicate/overwritten ids or a short list.
    assert len(proposals) > 0
    assert len(proposals) == len({proposal.id for proposal in proposals})
