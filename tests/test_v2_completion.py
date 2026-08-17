import json
from dataclasses import replace

import numpy as np

from llm_wiki.v2.chunking import chunk_document
from llm_wiki.v2.concept_extraction import extract
from llm_wiki.v2 import concept_index
from llm_wiki.v2.concept_index import build_index
from llm_wiki.v2.concept_store import build_concepts, read_concepts
from llm_wiki.v2.evaluation import evaluate
from llm_wiki.v2.models import ConceptProposal
from llm_wiki.v2.net_builder import build_net
from llm_wiki.v2.net_report import export_html, export_mermaid, render_tree
from llm_wiki.v2.net_store import NetStore
from llm_wiki.v2.tree_ops import rename_topic, undo_last


def vault(tmp_path):
    (tmp_path / "wiki.toml").write_text('[vault]\ncontent_dirs = ["domain"]\n[v2]\nenabled = true\n', encoding="utf-8")
    (tmp_path / "domain").mkdir()
    (tmp_path / "domain" / "policy.md").write_text(
        "---\nid: policy\nlayer: domain\nprojects: []\ntags: []\nconfidence: confirmed\nstatus: active\nsummary: retry policy\n---\n# Policy\n\nRetry count is 2. Frontend uses React.",
        encoding="utf-8")
    return tmp_path


def test_extraction_validation_rejects_ungrounded_output():
    chunk = chunk_document("doc", "domain/doc.md", "# D\n\nRetry count is 2.")[0]

    class Adapter:
        def extract_concepts(self, _):
            return [ConceptProposal("invented", "bad", "not in chunk", 0.9)]

    assert extract(chunk, Adapter()) == []


def test_net_export_and_undo(tmp_path):
    root = vault(tmp_path)
    build_concepts(root)
    store = build_net(root)
    topic = next(node for node in store.nodes() if node.id != "topic:knowledge" and node.type == "TOPIC")
    rename_topic(store, topic.id, "Renamed")
    undo_last(store)
    assert next(node for node in store.nodes() if node.id == topic.id).label == topic.label
    report = export_mermaid(root)
    assert "graph TD" in report.read_text(encoding="utf-8")


def test_net_tree_and_self_contained_html_visualization(tmp_path):
    root = vault(tmp_path)
    build_concepts(root)
    store = build_net(root)

    tree = render_tree(root, show_concepts=True, show_ids=True, ascii_only=True)
    assert "Knowledge [TOPIC] <topic:knowledge>" in tree
    assert "policy [DOCUMENT] <document:policy>" in tree
    assert "[CONCEPT]" in tree
    assert "-- " in tree

    report = export_html(root)
    html = report.read_text(encoding="utf-8")
    assert report.name == "NET.html"
    assert "LLM Wiki NET Explorer" in html
    assert '"nodes":[' in html
    assert '"edges":[' in html
    assert "script src=" not in html
    assert "https://" not in html


def test_net_tree_cli(tmp_path, monkeypatch, capsys):
    root = vault(tmp_path)
    build_concepts(root)
    build_net(root)
    from llm_wiki.v2 import net_cli

    monkeypatch.setattr("sys.argv", ["wiki-net", "tree", "--vault", str(root),
                                     "--show-concepts", "--ascii"])
    assert net_cli.main() == 0
    output = capsys.readouterr().out
    assert "NET (" in output
    assert "[DOCUMENT]" in output


def test_v2_eval_measures_current_query_and_approval_safety(tmp_path):
    root = vault(tmp_path)
    build_concepts(root)
    build_index(root)
    build_net(root)
    concept = read_concepts(root)[0]
    gold = {
        "concepts": [{"concept_id": concept.id}],
        "relations": [],
        "queries": [{"query": "retry count", "expect": [concept.id], "historical": False}],
    }
    path = root / ".llm_wiki_v2" / "v2_gold.json"
    path.write_text(json.dumps(gold), encoding="utf-8")
    metrics = evaluate(root)
    assert metrics["current_historical_hit_at_k"] == 1.0
    assert metrics["risky_unapproved"] == 0


def test_changed_build_preserves_unchanged_document_concepts(tmp_path):
    root = vault(tmp_path)
    (root / "domain" / "second.md").write_text(
        "---\nid: second\nlayer: domain\nprojects: []\ntags: []\nconfidence: confirmed\nstatus: active\nsummary: second\n---\n# Second\n\nBackend uses Spring.",
        encoding="utf-8")
    build_concepts(root)
    before = {c.id for c in read_concepts(root) if c.document_id == "second"}
    policy = root / "domain" / "policy.md"
    policy.write_text(policy.read_text(encoding="utf-8") + "\nCache uses Redis.", encoding="utf-8")
    build_concepts(root, changed_only=True)
    after = {c.id for c in read_concepts(root) if c.document_id == "second"}
    assert after == before


def test_changed_index_embeds_only_new_concepts(tmp_path, monkeypatch):
    root = vault(tmp_path)
    build_concepts(root)
    build_index(root)
    index_root = root / ".llm_wiki_v2" / "concept_embeddings"
    before_meta = json.loads((index_root / "meta.json").read_text(encoding="utf-8"))
    before_vectors = np.load(index_root / "vectors.npy").copy()

    (root / "domain" / "new.md").write_text(
        "---\nid: new\nlayer: domain\nprojects: []\ntags: []\nconfidence: confirmed\n"
        "status: active\nsummary: new\n---\n# New\n\nNew service uses Redis.",
        encoding="utf-8",
    )
    build_concepts(root, changed_only=True)

    original = concept_index._embed_passages
    embedded_batches = []

    def tracked_embed(texts, vault=None, show_progress=False):
        embedded_batches.append(len(texts))
        return original(texts, vault, show_progress)

    monkeypatch.setattr(concept_index, "_embed_passages", tracked_embed)
    build_index(root, changed_only=True)

    after_meta = json.loads((index_root / "meta.json").read_text(encoding="utf-8"))
    after_vectors = np.load(index_root / "vectors.npy")
    before_by_id = {row["concept_id"]: index for index, row in enumerate(before_meta)}
    after_by_id = {row["concept_id"]: index for index, row in enumerate(after_meta)}
    unchanged_ids = set(before_by_id) & set(after_by_id)
    assert embedded_batches == [1]
    assert all(np.array_equal(before_vectors[before_by_id[id_]], after_vectors[after_by_id[id_]])
               for id_ in unchanged_ids)


def test_changed_index_rebuilds_when_index_identity_changes(tmp_path, monkeypatch):
    root = vault(tmp_path)
    build_concepts(root)
    build_index(root)
    identity_path = root / ".llm_wiki_v2" / "concept_embeddings" / "identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity["indexed_text_schema"] = "changed"
    identity_path.write_text(json.dumps(identity), encoding="utf-8")

    original = concept_index._embed_passages
    embedded_batches = []

    def tracked_embed(texts, vault=None, show_progress=False):
        embedded_batches.append(len(texts))
        return original(texts, vault, show_progress)

    monkeypatch.setattr(concept_index, "_embed_passages", tracked_embed)
    build_index(root, changed_only=True)

    assert embedded_batches == [len(read_concepts(root))]
