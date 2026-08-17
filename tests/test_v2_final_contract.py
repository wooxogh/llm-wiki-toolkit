import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_wiki import config
from llm_wiki.v2 import artifacts
from llm_wiki.v2.chunking import chunk_document
from llm_wiki.v2 import concept_store
from llm_wiki.v2.concept_index import build_index
from llm_wiki.v2.concept_store import build_concepts, read_concepts
from llm_wiki.v2.health import check_v2_health
from llm_wiki.v2.llm_adapter import (AgentCLIUserLLMAdapter, CommandUserLLMAdapter,
                                     RuleBasedUserLLMAdapter, _parse_agent_json,
                                     default_adapter)
from llm_wiki.v2.models import Concept, Document, NetEdge, NetNode, PlacementProposal, RelationProposal
from llm_wiki.v2.net_builder import build_net
from llm_wiki.v2.net_store import NetIntegrityError, NetStore
from llm_wiki.v2.query import auto_decision, recall
from llm_wiki.v2.review import submit_relation_proposal
from llm_wiki.v2.schemas import (CONCEPT_PROMPT_VERSION, EdgeType, NodeType,
                                 PLACEMENT_PROMPT_VERSION, RELATION_PROMPT_VERSION,
                                 TEMPORAL_PROMPT_VERSION, RelationType)
from llm_wiki.v2.tree_ops import (add_secondary_topic, create_collection, merge_topic,
                                  remove_secondary_topic, undo_last)


def _vault(tmp_path: Path) -> Path:
    (tmp_path / "wiki.toml").write_text(
        '[vault]\ncontent_dirs = ["domain"]\n[v2]\nenabled = true\n', encoding="utf-8"
    )
    (tmp_path / "domain").mkdir()
    (tmp_path / "domain" / "policy.md").write_text(
        "---\nid: policy\nlayer: domain\nprojects: []\ntags: []\nconfidence: confirmed\n"
        "status: active\nsummary: policy\n---\n# Policy\n\nRetry count is 2. Backend uses Spring.",
        encoding="utf-8",
    )
    return tmp_path


def _graph(store: NetStore) -> tuple[list[dict], list[dict]]:
    return ([node.to_dict() for node in store.nodes()], [edge.to_dict() for edge in store.edges()])


def test_schema_manifest_is_not_silently_overwritten_and_changed_build_recovers(tmp_path):
    vault = _vault(tmp_path)
    build_concepts(vault)
    path = artifacts.artifact_path("schemas.json", vault)
    stale = json.loads(path.read_text(encoding="utf-8"))
    stale["prompt_versions"]["concept_extraction"] = "concept-extraction.old"
    path.write_text(json.dumps(stale), encoding="utf-8")

    artifacts.ensure_layout(vault)
    assert json.loads(path.read_text(encoding="utf-8"))["prompt_versions"]["concept_extraction"] == "concept-extraction.old"
    assert any("schema stale" in issue for issue in check_v2_health(vault))

    build_concepts(vault, changed_only=True)
    assert artifacts.read_schema_manifest(vault)["prompt_versions"]["concept_extraction"] == CONCEPT_PROMPT_VERSION
    assert all(concept.prompt_version == CONCEPT_PROMPT_VERSION for concept in read_concepts(vault))


def test_changed_build_rebuilds_when_chunk_target_changes(tmp_path):
    vault = _vault(tmp_path)
    build_concepts(vault, target_chars=700)
    _, chunks, _ = build_concepts(vault, target_chars=321, changed_only=True)
    assert chunks and all(chunk.target_chars == 321 for chunk in chunks)


def test_changed_build_does_not_chunk_unchanged_documents(tmp_path, monkeypatch):
    vault = _vault(tmp_path)
    second = vault / "domain" / "other.md"
    second.write_text(
        "---\nid: other\nlayer: domain\nprojects: []\ntags: []\nconfidence: confirmed\n"
        "status: active\nsummary: other\n---\n# Other\n\nOther value is stable.",
        encoding="utf-8",
    )
    build_concepts(vault)
    policy = vault / "domain" / "policy.md"
    policy.write_text(policy.read_text(encoding="utf-8") + "\nNew policy value is 3.", encoding="utf-8")

    original = concept_store.chunk_document
    calls = []

    def tracked_chunk(document_id, path, raw, target_chars):
        calls.append(path)
        return original(document_id, path, raw, target_chars)

    monkeypatch.setattr(concept_store, "chunk_document", tracked_chunk)
    build_concepts(vault, changed_only=True)

    assert calls == ["domain/policy.md"]


def test_changed_build_removes_deleted_document_artifacts(tmp_path):
    vault = _vault(tmp_path)
    second = vault / "domain" / "other.md"
    second.write_text(
        "---\nid: other\nlayer: domain\nprojects: []\ntags: []\nconfidence: confirmed\n"
        "status: active\nsummary: other\n---\n# Other\n\nOther value is stable.",
        encoding="utf-8",
    )
    build_concepts(vault)
    second.unlink()

    docs, chunks, concepts = build_concepts(vault, changed_only=True)

    assert {doc.id for doc in docs} == {"policy"}
    assert all(chunk.document_id == "policy" for chunk in chunks)
    assert all(concept.document_id == "policy" for concept in concepts)


def test_full_then_incremental_build_remains_healthy(tmp_path):
    vault = _vault(tmp_path)
    build_concepts(vault)
    build_index(vault)
    build_net(vault)

    (vault / "domain" / "new.md").write_text(
        "---\nid: new\nlayer: domain\nprojects: []\ntags: []\nconfidence: confirmed\n"
        "status: active\nsummary: new\n---\n# New\n\nNew service uses Redis.",
        encoding="utf-8",
    )
    build_concepts(vault, changed_only=True)
    build_index(vault, changed_only=True)
    build_net(vault, changed_only=True)

    assert check_v2_health(vault) == []


def test_models_reject_unknown_artifact_fields():
    with pytest.raises(ValueError, match="unknown field"):
        NetNode.from_dict({"id": "topic:a", "type": "TOPIC", "label": "A", "future": True})


def test_yaml_date_metadata_serializes_as_iso_text():
    document = Document("doc", "domain/doc.md", "hash", metadata={"updated": date(2026, 8, 17)})
    assert document.to_dict()["metadata"]["updated"] == "2026-08-17"


def test_create_merge_and_secondary_operations_restore_exact_snapshots(tmp_path):
    vault = _vault(tmp_path)
    build_concepts(vault)
    store = build_net(vault)

    before_create = _graph(store)
    create_collection(store, "monthly", "Monthly", "topic:knowledge")
    undo_last(store)
    assert _graph(store) == before_create

    nodes, edges = store.nodes(), store.edges()
    nodes.extend([NetNode.topic("A", "topic:a"), NetNode.topic("B", "topic:b")])
    edges.extend([
        NetEdge("edge:parent:root:a", EdgeType.PARENT_OF.value, "topic:knowledge", "topic:a"),
        NetEdge("edge:parent:root:b", EdgeType.PARENT_OF.value, "topic:knowledge", "topic:b"),
    ])
    store.replace_graph(nodes, edges)
    before_merge = _graph(store)
    merge_topic(store, "topic:a", "topic:b")
    assert next(node for node in store.nodes() if node.id == "topic:a").state == "ARCHIVED"
    undo_last(store)
    assert _graph(store) == before_merge

    concept_id = read_concepts(vault)[0].id
    before_add = _graph(store)
    add_secondary_topic(store, concept_id, "topic:a")
    undo_last(store)
    assert _graph(store) == before_add
    add_secondary_topic(store, concept_id, "topic:a")
    before_remove = _graph(store)
    remove_secondary_topic(store, concept_id, "topic:a")
    undo_last(store)
    assert _graph(store) == before_remove


def test_document_cannot_have_two_tree_locations(tmp_path):
    vault = _vault(tmp_path)
    build_concepts(vault)
    store = build_net(vault)
    store.upsert_node(NetNode.topic("Other", "topic:other"))
    with pytest.raises(NetIntegrityError, match="only one"):
        store.upsert_edge(NetEdge("edge:contains:other", EdgeType.CONTAINS_DOCUMENT.value,
                                  "topic:other", "document:policy"))


def test_collection_cannot_parent_another_tree_node(tmp_path):
    vault = _vault(tmp_path)
    store = NetStore(vault)
    store.write_nodes([
        NetNode.topic("Root", "topic:knowledge"),
        NetNode("collection:a", NodeType.COLLECTION.value, "A"),
        NetNode.topic("Child", "topic:child"),
    ])
    with pytest.raises(NetIntegrityError, match="source must be TOPIC"):
        store.upsert_edge(NetEdge("edge:bad", EdgeType.PARENT_OF.value,
                                  "collection:a", "topic:child"))


def test_command_adapter_sends_all_versioned_semantic_contracts(monkeypatch):
    requests = []

    def run(*args, **kwargs):
        request = json.loads(kwargs["input"])
        requests.append(request)
        task = request["task"]
        if task == "place_concept":
            payload = {"concept_id": "concept:a", "primary_topic_id": "topic:knowledge",
                       "secondary_topic_ids": [], "confidence": 0.8, "reason": "fit"}
        else:
            payload = {"proposal": None}
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr("llm_wiki.v2.llm_adapter.subprocess.run", run)
    adapter = CommandUserLLMAdapter("bridge")
    concept = Concept("concept:a", "doc", "chunk", "Retry is 2", "retry", "Retry is 2",
                      0.9, "hash", 0, 10)
    adapter.place_concept(concept, [{"id": "topic:knowledge", "label": "Knowledge"}])
    adapter.classify_relation(concept, Concept(**{**concept.to_dict(), "id": "concept:b"}))
    adapter.resolve_temporal(concept, Concept(**{**concept.to_dict(), "id": "concept:b"}))

    by_task = {request["task"]: request["payload"] for request in requests}
    assert by_task["place_concept"]["prompt_version"] == PLACEMENT_PROMPT_VERSION
    assert by_task["classify_relation"]["prompt_version"] == RELATION_PROMPT_VERSION
    assert by_task["resolve_temporal"]["prompt_version"] == TEMPORAL_PROMPT_VERSION
    assert all("instruction" in payload and "json_schema" in payload for payload in by_task.values())


@pytest.mark.parametrize("agent", ["codex", "claude"])
def test_v2_agent_config_selects_authenticated_cli(tmp_path, agent):
    vault = _vault(tmp_path)
    config_path = vault / "wiki.toml"
    config_path.write_text(config_path.read_text(encoding="utf-8") + f'agent = "{agent}"\n', encoding="utf-8")
    selected = default_adapter(vault)
    assert isinstance(selected, AgentCLIUserLLMAdapter)
    assert selected.agent == agent
    assert selected.model_identity == f"{agent}-cli-default"


def test_invalid_v2_agent_is_rejected(tmp_path):
    vault = _vault(tmp_path)
    config_path = vault / "wiki.toml"
    config_path.write_text(config_path.read_text(encoding="utf-8") + 'agent = "other"\n', encoding="utf-8")
    with pytest.raises(config.ConfigError, match="codex.*claude"):
        config.load(vault)


def test_codex_agent_uses_ephemeral_read_only_structured_output(monkeypatch, tmp_path):
    captured = {}

    def run(argv, **kwargs):
        captured["argv"] = argv
        captured["prompt"] = kwargs["input"]
        output = Path(argv[argv.index("--output-last-message") + 1])
        output.write_text('{"concepts": []}', encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("llm_wiki.v2.llm_adapter.shutil.which", lambda name: f"C:/bin/{name}.exe")
    monkeypatch.setattr("llm_wiki.v2.llm_adapter.subprocess.run", run)
    adapter = AgentCLIUserLLMAdapter("codex", tmp_path)
    chunk = chunk_document("doc", "domain/doc.md", "# D\n\nRetry count is 2.")[0]
    assert adapter.extract_concepts(chunk) == []
    assert captured["argv"][1:3] == ["exec", "--cd"]
    assert "--sandbox" in captured["argv"] and "read-only" in captured["argv"]
    assert "--ephemeral" in captured["argv"]
    assert "--output-schema" in captured["argv"]
    assert "Do not inspect files" in captured["prompt"]


def test_codex_placement_schema_removes_unsupported_unique_items(monkeypatch, tmp_path):
    captured = {}

    def run(argv, **kwargs):
        schema = Path(argv[argv.index("--output-schema") + 1])
        captured["schema"] = json.loads(schema.read_text(encoding="utf-8"))
        captured["prompt"] = kwargs["input"]
        output = Path(argv[argv.index("--output-last-message") + 1])
        output.write_text(json.dumps({
            "concept_id": "concept:a",
            "primary_topic_id": "topic:knowledge",
            "secondary_topic_ids": ["topic:other", "topic:other"],
            "create_topic_label": None,
            "collection_id": None,
            "create_collection_label": None,
            "collection_type": None,
            "confidence": 0.8,
            "reason": "fit",
        }), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("llm_wiki.v2.llm_adapter.shutil.which", lambda name: f"C:/bin/{name}.exe")
    monkeypatch.setattr("llm_wiki.v2.llm_adapter.subprocess.run", run)
    adapter = AgentCLIUserLLMAdapter("codex", tmp_path)
    concept = Concept("concept:a", "doc", "chunk", "Retry is 2", "retry", "Retry is 2",
                      0.9, "hash", 0, 10)

    proposal = adapter.place_concept(concept, [{"id": "topic:knowledge", "label": "Knowledge"}])

    secondary_schema = captured["schema"]["properties"]["secondary_topic_ids"]
    assert "uniqueItems" not in secondary_schema
    assert '"uniqueItems": true' in captured["prompt"]
    assert proposal.secondary_topic_ids == ["topic:other"]


def test_claude_structured_output_envelope_is_unwrapped():
    raw = json.dumps({"result": "ok", "structured_output": {"concepts": []}})
    assert _parse_agent_json(raw, "Claude") == {"concepts": []}


def test_ai_collection_proposal_places_source_document(tmp_path):
    vault = _vault(tmp_path)

    class Adapter(RuleBasedUserLLMAdapter):
        def place_concept(self, concept, tree_candidates):
            assert all("label" in row and "parent_id" in row for row in tree_candidates)
            return PlacementProposal(concept.id, "topic:knowledge",
                                     create_collection_label="Monthly Reports",
                                     collection_type="monthly", confidence=0.9,
                                     reason="recurring report")

    adapter = Adapter()
    build_concepts(vault, adapter=adapter)
    store = build_net(vault, adapter=adapter)
    assert any(node.id == "collection:monthly-reports" for node in store.nodes())
    assert any(edge.type == EdgeType.CONTAINS_DOCUMENT.value
               and edge.source == "collection:monthly-reports" and edge.target == "document:policy"
               for edge in store.edges())


def test_supersedes_without_temporal_evidence_is_rejected(tmp_path):
    vault = _vault(tmp_path)
    build_concepts(vault)
    store = build_net(vault)
    source, target = read_concepts(vault)[:2]
    proposal = RelationProposal("invalid", source.id, target.id, RelationType.SUPERSEDES.value,
                                0.99, source.source_quote, same_subject=True, same_scope=True,
                                temporal_change_possible=True, reason="claimed update")
    with pytest.raises(ValueError, match="newer metadata or explicit revision evidence"):
        submit_relation_proposal(store, proposal)


def test_hash_embedding_auto_mode_fails_closed_to_review(tmp_path):
    vault = _vault(tmp_path)
    build_concepts(vault)
    build_index(vault)
    build_net(vault)
    thresholds = vault / "thresholds.json"
    thresholds.write_text('{"score": -1, "margin": -1, "none": -1, "rerank": -1}', encoding="utf-8")
    decision = auto_decision(vault, "retry count", thresholds_path=thresholds)
    assert decision["decision"] == "review"
    assert decision["reason"] == "uncalibrated-concept-embedding"


def test_historical_year_is_a_soft_ranking_signal(tmp_path):
    vault = _vault(tmp_path)
    first = vault / "domain" / "policy.md"
    first.write_text(first.read_text(encoding="utf-8").replace("summary: policy", "summary: policy\nupdated: 2025-01-01"),
                     encoding="utf-8")
    (vault / "domain" / "new.md").write_text(
        "---\nid: new\nlayer: domain\nprojects: []\ntags: []\nconfidence: confirmed\n"
        "status: active\nsummary: new\nupdated: 2026-01-01\n---\n# Policy\n\nRetry count is 2.",
        encoding="utf-8",
    )
    build_concepts(vault)
    build_index(vault)
    build_net(vault)
    rows = recall(vault, "2025 retry count", historical=True)
    assert rows[0]["document_id"] == "policy"


def test_utf8_bom_wiki_config_is_supported(tmp_path):
    (tmp_path / "wiki.toml").write_text('\ufeff[vault]\ncontent_dirs = ["notes"]\n', encoding="utf-8")
    assert config.load(tmp_path).content_dirs == ("notes",)
