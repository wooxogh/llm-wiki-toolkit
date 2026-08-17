from dataclasses import replace
from pathlib import Path

import pytest

from llm_wiki.v2 import artifacts
from llm_wiki.v2.concept_index import build_index
from llm_wiki.v2.concept_store import build_concepts, read_concepts, write_concepts
from llm_wiki.v2.models import NetEdge, NetNode, RelationProposal
from llm_wiki.v2.net_builder import build_net
from llm_wiki.v2.net_store import NetIntegrityError, NetStore
from llm_wiki.v2.query import recall
from llm_wiki.v2.review import approve_review_item, submit_relation_proposal
from llm_wiki.v2.schemas import ConceptState, EdgeType, NodeType, RelationType
from llm_wiki.v2.tree_ops import delete_topic, restore_topic
from llm_wiki import wiki_health


def _vault(tmp_path: Path) -> Path:
    (tmp_path / "wiki.toml").write_text(
        '[vault]\ncontent_dirs = ["domain"]\n\n[v2]\nenabled = true\n',
        encoding="utf-8",
    )
    (tmp_path / "domain").mkdir()
    (tmp_path / "domain" / "stack.md").write_text(
        "---\nid: stack\nlayer: domain\nprojects: []\ntags: []\nconfidence: confirmed\nstatus: active\nsummary: Stack decisions\n---\n"
        "# Stack\n\nFrontend no longer uses Vue and now uses React. Backend uses Spring.",
        encoding="utf-8",
    )
    return tmp_path


def test_net_build_creates_ai_topics_and_delete_preserves_concepts(tmp_path):
    vault = _vault(tmp_path)
    build_concepts(vault)
    store = build_net(vault)
    topics = [n for n in store.nodes() if n.type == NodeType.TOPIC.value]
    concepts = [n for n in store.nodes() if n.type == NodeType.CONCEPT.value]
    assert len(topics) >= 2
    assert concepts
    topic_id = next(t.id for t in topics if t.id != "topic:knowledge")
    delete_topic(store, topic_id)
    restore_topic(store, topic_id)
    assert [n for n in store.nodes() if n.type == NodeType.CONCEPT.value] == concepts
    assert any(op.op == "DELETE_TOPIC" for op in store.operations())
    assert any(op.op == "RESTORE_TOPIC" for op in store.operations())


def test_net_build_reports_each_long_running_phase(tmp_path):
    vault = _vault(tmp_path)
    build_concepts(vault)
    events = []

    build_net(vault, progress=lambda phase, done, total, label: events.append(
        (phase, done, total, label)))

    for phase in ("placement", "candidates", "relations"):
        phase_events = [event for event in events if event[0] == phase]
        assert phase_events
        assert phase_events[-1][1] == phase_events[-1][2]
        assert phase_events[-1][3]


def test_cycle_is_rejected(tmp_path):
    vault = _vault(tmp_path)
    store = NetStore(vault)
    store.write_nodes([
        NetNode("topic:a", NodeType.TOPIC.value, "A"),
        NetNode("topic:b", NodeType.TOPIC.value, "B"),
    ])
    store.upsert_edge(NetEdge("edge:a:b", EdgeType.PARENT_OF.value, "topic:a", "topic:b"))
    with pytest.raises(NetIntegrityError):
        store.upsert_edge(NetEdge("edge:b:a", EdgeType.PARENT_OF.value, "topic:b", "topic:a"))


def test_safe_relation_commits_but_risky_relation_waits_for_review(tmp_path):
    vault = _vault(tmp_path)
    build_concepts(vault)
    store = build_net(vault)
    c1, c2 = read_concepts(vault)[:2]
    safe = RelationProposal("p:safe", c1.id, c2.id, RelationType.SUPPORTS.value, 0.95, c1.source_quote)
    assert submit_relation_proposal(store, safe) == "committed"
    risky = RelationProposal("p:risky", c1.id, c2.id, RelationType.SUPERSEDES.value, 0.99,
                             c1.source_quote, same_subject=True, same_scope=True,
                             temporal_change_possible=True, reason="explicit revision")
    assert submit_relation_proposal(store, risky) == "review"
    assert any(e.relation == RelationType.SUPPORTS.value for e in store.edges())
    assert not any(e.relation == RelationType.SUPERSEDES.value for e in store.edges())
    approve_review_item(store, "review:p:risky", actor="louis", vault=vault)
    assert any(e.relation == RelationType.SUPERSEDES.value and e.approved_by == "louis" for e in store.edges())
    old = next(c for c in read_concepts(vault) if c.id == c2.id)
    assert old.state == ConceptState.SUPERSEDED.value


def test_query_prefers_active_and_historical_includes_superseded(tmp_path):
    vault = _vault(tmp_path)
    build_concepts(vault)
    concepts = read_concepts(vault)
    write_concepts([replace(concepts[0], state=ConceptState.SUPERSEDED.value), concepts[1]], vault)
    build_index(vault)
    current = recall(vault, "Frontend React", k=5)
    historical = recall(vault, "Frontend React", k=5, historical=True)
    assert all(row["state"] != ConceptState.SUPERSEDED.value for row in current)
    assert any(row["state"] == ConceptState.SUPERSEDED.value for row in historical)


def test_health_detects_missing_chunk_and_stale_index(tmp_path):
    vault = _vault(tmp_path)
    build_concepts(vault)
    build_index(vault)
    root = artifacts.artifact_root(vault)
    (root / "chunks.jsonl").write_text("", encoding="utf-8")
    from llm_wiki.v2.health import check_v2_health
    issues = check_v2_health(vault)
    assert any("missing chunk" in issue or "chunks are missing" in issue for issue in issues)


def test_plain_markdown_vault_is_healthy_through_v2_only_pipeline(tmp_path):
    (tmp_path / "wiki.toml").write_text(
        '[vault]\ncontent_dirs = ["."]\n[v2]\nenabled = true\n', encoding="utf-8")
    (tmp_path / "notes.md").write_text(
        "# Decisions\n\nRetry count is 2. Frontend uses React.\n", encoding="utf-8")

    build_concepts(tmp_path)
    build_index(tmp_path)
    build_net(tmp_path)

    issues = wiki_health.check_health(tmp_path, mode="ci", v2_only=True)
    assert [issue for issue in issues if issue.severity == "error"] == []
