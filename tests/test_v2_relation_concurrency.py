import re
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


def _build_fixture_vault(vault: Path, relation_concurrency: int) -> None:
    from llm_wiki.v2.concept_store import build_concepts
    from llm_wiki.v2.net_builder import build_net

    vault.mkdir(parents=True, exist_ok=True)
    (vault / "wiki.toml").write_text(
        '[vault]\ncontent_dirs = ["domain"]\n\n[v2]\nenabled = true\n'
        f'relation_concurrency = {relation_concurrency}\n'
        'safe_relation_min_confidence = 0.0\n', encoding="utf-8")
    (vault / "domain").mkdir()
    for i in range(6):
        (vault / "domain" / f"doc{i}.md").write_text(
            f"---\nid: doc{i}\nlayer: domain\nprojects: []\ntags: []\nconfidence: confirmed\nstatus: active\nsummary: doc {i}\n---\n"
            f"# Doc {i}\n\nFrontend uses React in service {i}.",
            encoding="utf-8")
    build_concepts(vault)
    build_net(vault, adapter=_ThreadRecordingAdapter())


_VOLATILE_TIMESTAMP = re.compile(r'"(created_at|approved_at|updated_at)": "[^"]*"')


def _stable(text: str) -> str:
    return _VOLATILE_TIMESTAMP.sub(r'"\1": "<ts>"', text)


def test_concurrent_build_commits_every_expected_proposal(tmp_path: Path):
    serial_vault = tmp_path / "serial"
    concurrent_vault = tmp_path / "concurrent"
    _build_fixture_vault(serial_vault, relation_concurrency=1)
    _build_fixture_vault(concurrent_vault, relation_concurrency=4)

    serial_proposals = (serial_vault / ".llm_wiki_v2" / "net" / "proposals.jsonl").read_text(encoding="utf-8")
    concurrent_proposals = (concurrent_vault / ".llm_wiki_v2" / "net" / "proposals.jsonl").read_text(encoding="utf-8")
    serial_edges = (serial_vault / ".llm_wiki_v2" / "net" / "edges.jsonl").read_text(encoding="utf-8")
    concurrent_edges = (concurrent_vault / ".llm_wiki_v2" / "net" / "edges.jsonl").read_text(encoding="utf-8")

    # NetStore sorts proposals/edges by id before every write, so a lost
    # update from the read-modify-write race this plan guards against would
    # show up here as a byte-level divergence between the two runs (missing
    # or overwritten lines), not merely as duplicate ids.
    assert serial_proposals != ""
    assert _stable(serial_proposals) == _stable(concurrent_proposals)
    assert serial_edges != ""
    assert _stable(serial_edges) == _stable(concurrent_edges)
