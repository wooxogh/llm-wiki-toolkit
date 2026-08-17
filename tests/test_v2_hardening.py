from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_wiki.v2.concept_extraction import _validate
from llm_wiki.v2.concept_index import build_index
from llm_wiki.v2.concept_store import build_concepts, read_concepts
from llm_wiki.v2.llm_adapter import CommandUserLLMAdapter
from llm_wiki.v2.models import ConceptProposal
from llm_wiki.v2.net_builder import build_net
from llm_wiki.v2.net_report import export_mermaid
from llm_wiki.v2.net_store import NetStore
from llm_wiki.v2.query import recall
from llm_wiki.v2.review import approve_review_item, reject_review_item, submit_relation_proposal
from llm_wiki.v2.schemas import RelationType
from llm_wiki.v2.tree_ops import move_document, rename_topic
from llm_wiki.v2.chunking import chunk_document


def _vault(tmp_path: Path) -> Path:
    (tmp_path / "wiki.toml").write_text('[vault]\ncontent_dirs = ["domain"]\n[v2]\nenabled = true\n', encoding="utf-8")
    (tmp_path / "domain").mkdir()
    (tmp_path / "domain" / "stack.md").write_text(
        "---\nid: stack\nlayer: domain\nprojects: []\ntags: []\nconfidence: confirmed\nstatus: active\nsummary: stack\n---\n# Stack\n\nFrontend no longer uses Vue and now uses React. Backend uses Spring.",
        encoding="utf-8")
    return tmp_path


def test_command_adapter_sends_versioned_atomic_contract(monkeypatch):
    captured = {}

    def run(*args, **kwargs):
        captured["request"] = kwargs["input"]
        return SimpleNamespace(returncode=0, stdout='{"concepts": []}', stderr="")

    monkeypatch.setattr("llm_wiki.v2.llm_adapter.subprocess.run", run)
    chunk = chunk_document("doc", "domain/doc.md", "# D\n\nFrontend uses React.")[0]
    assert CommandUserLLMAdapter("bridge").extract_concepts(chunk) == []
    assert "Atomic Concept" in captured["request"]
    assert "json_schema" in captured["request"]
    assert "prompt_version" in captured["request"]


def test_quote_without_semantic_overlap_is_rejected():
    chunk = chunk_document("doc", "domain/doc.md", "# D\n\nBackend uses Spring.")[0]
    assert _validate([ConceptProposal("The moon is green", "moon", "Backend uses Spring.", 0.9)], chunk) == []


def test_net_rebuild_preserves_renamed_topic_and_document_placement(tmp_path):
    vault = _vault(tmp_path)
    build_concepts(vault)
    store = build_net(vault)
    topic = next(node for node in store.nodes() if node.id != "topic:knowledge" and node.type == "TOPIC")
    rename_topic(store, topic.id, "User Frontend")
    move_document(store, "stack", topic.id)
    rebuilt = build_net(vault)
    assert next(node for node in rebuilt.nodes() if node.id == topic.id).label == "User Frontend"
    assert any(edge.type == "CONTAINS_DOCUMENT" and edge.source == topic.id and edge.target == "document:stack"
               for edge in rebuilt.edges())


def test_rejected_review_cannot_be_approved_later(tmp_path):
    vault = _vault(tmp_path)
    build_concepts(vault)
    store = build_net(vault)
    first, second = read_concepts(vault)[:2]
    from llm_wiki.v2.models import RelationProposal
    proposal = RelationProposal("p:reject", first.id, second.id, RelationType.SUPERSEDES.value,
                                0.99, first.source_quote, same_subject=True, same_scope=True,
                                temporal_change_possible=True, reason="explicit revision")
    assert submit_relation_proposal(store, proposal) == "review"
    reject_review_item(store, "review:p:reject")
    with pytest.raises(ValueError, match="not OPEN"):
        approve_review_item(store, "review:p:reject", vault=vault)
    assert submit_relation_proposal(store, proposal) == "rejected"


def test_live_markdown_change_fails_health_and_query(tmp_path):
    vault = _vault(tmp_path)
    build_concepts(vault)
    build_index(vault)
    page = vault / "domain" / "stack.md"
    page.write_text(page.read_text(encoding="utf-8").replace("React", "Svelte"), encoding="utf-8")
    from llm_wiki.v2.health import check_v2_health
    assert any("stale" in issue for issue in check_v2_health(vault))
    with pytest.raises(RuntimeError, match="refused"):
        recall(vault, "React")


def test_mermaid_fence_is_closed(tmp_path):
    vault = _vault(tmp_path)
    build_concepts(vault)
    build_net(vault)
    assert export_mermaid(vault).read_text(encoding="utf-8").rstrip().endswith("```")


def test_v2_eval_rejects_empty_gold():
    from llm_wiki.v2.evaluation import validate_gold
    errors = validate_gold({"concepts": [], "relations": [], "queries": []})
    assert any("at least one" in error for error in errors)
