from pathlib import Path

from llm_wiki.v2.concept_index import build_index
from llm_wiki.v2.concept_store import build_concepts
from llm_wiki.v2.net_builder import build_net
from llm_wiki.v2.net_store import NetStore
from llm_wiki.v2.query import auto_decision, recall
from llm_wiki.v2 import concept_index, query as query_module


def _vault(tmp_path: Path) -> Path:
    (tmp_path / "wiki.toml").write_text(
        '[vault]\ncontent_dirs = ["domain"]\n\n[v2]\nenabled = true\n', encoding="utf-8")
    (tmp_path / "domain").mkdir()
    (tmp_path / "domain" / "stack.md").write_text(
        "---\nid: stack\nlayer: domain\nprojects: []\ntags: []\nconfidence: confirmed\nstatus: active\nsummary: Stack decisions\n---\n"
        "# Stack\n\nFrontend no longer uses Vue and now uses React. Backend uses Spring.",
        encoding="utf-8")
    build_concepts(tmp_path)
    build_index(tmp_path)
    build_net(tmp_path)
    return tmp_path


def test_recall_reads_the_net_graph_at_most_once_per_call(tmp_path, monkeypatch):
    vault = _vault(tmp_path)
    # _fail_closed() builds its own separate NetStore for health checks; that
    # cost is out of scope here — this test isolates recall()'s OWN graph
    # reads, which used to be 2 + one-per-result-row and should now be 1 each.
    monkeypatch.setattr(query_module, "_fail_closed", lambda vault: None)
    calls = {"nodes": 0, "edges": 0}
    real_nodes, real_edges = NetStore.nodes, NetStore.edges

    def counting_nodes(self):
        calls["nodes"] += 1
        return real_nodes(self)

    def counting_edges(self):
        calls["edges"] += 1
        return real_edges(self)

    monkeypatch.setattr(NetStore, "nodes", counting_nodes)
    monkeypatch.setattr(NetStore, "edges", counting_edges)
    rows = recall(vault, "React", k=5)
    assert rows
    assert calls["nodes"] == 1
    assert calls["edges"] == 1


def test_auto_decision_computes_signals_only_once(tmp_path, monkeypatch):
    vault = _vault(tmp_path)
    calls = {"count": 0}
    real_search_with_signals = concept_index.search_with_signals

    def counting(vault_arg, query, k=8, concepts=None):
        calls["count"] += 1
        return real_search_with_signals(vault_arg, query, k=k, concepts=concepts)

    monkeypatch.setattr(concept_index, "search_with_signals", counting)
    payload = auto_decision(vault, "React", k=5)
    assert payload["decision"] in {"answer", "review", "none"}
    assert calls["count"] == 1
