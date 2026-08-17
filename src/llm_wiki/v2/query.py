"""Lifecycle-aware v2 query orchestration: tree + dense + text + graph + RRF."""
from __future__ import annotations

import json
import re
from pathlib import Path

from llm_wiki.paths import content_root
from llm_wiki.retrieval import retrieval_policy
from llm_wiki.v2 import concept_index
from llm_wiki.v2.concept_store import read_concepts
from llm_wiki.v2.models import Concept
from llm_wiki.v2.net_store import NetStore
from llm_wiki.v2.schemas import ConceptState, EdgeType, RelationType

RRF_K = 60


def recall(vault: Path | None, query: str, k: int = 8, historical: bool = False,
           rerank: int = 0) -> list[dict]:
    """Soft-route through NET, then fuse semantic/text seeds and graph evidence.

    Tree routing deliberately only adds a rank signal. A bad auto-placement must
    never make an otherwise relevant concept unreachable.
    """
    concepts = read_concepts(vault)
    if not concepts:
        return []
    _fail_closed(vault)
    seeds = concept_index.search(vault, query, k=max(k * 5, 30))
    scores = _rrf_scores(seeds)
    store = NetStore(vault)
    nodes = {node.id: node for node in store.nodes()}
    tree = _tree_scores(query, store)
    for concept_id, value in tree.items():
        scores[concept_id] = scores.get(concept_id, 0.0) + value
    for concept_id, value in _graph_scores(scores, store, historical).items():
        scores[concept_id] = scores.get(concept_id, 0.0) + value
    by_id = {concept.id: concept for concept in concepts}
    if historical:
        years = set(re.findall(r"\b(?:19|20)\d{2}\b", query))
        for concept in concepts:
            if concept.updated_at and years.intersection(re.findall(r"\b(?:19|20)\d{2}\b", concept.updated_at)):
                scores[concept.id] = scores.get(concept.id, 0.0) + 0.02
    rows: list[dict] = []
    for concept_id, score in sorted(scores.items(), key=lambda item: (-item[1], item[0])):
        concept = by_id.get(concept_id)
        if not concept or (not historical and concept.state in {ConceptState.SUPERSEDED.value, ConceptState.ARCHIVED.value, ConceptState.DUPLICATE.value}):
            continue
        evidence = _evidence(concept_id, store)
        warning = None
        if concept.state == ConceptState.DISPUTED.value:
            warning = "DISPUTED: review contradiction evidence before relying on this concept"
        elif any(item["relation"] == RelationType.OVERRIDES.value for item in evidence):
            warning = "OVERRIDES: verify scope/priority before treating this as universal"
        rows.append(_row(score, concept, warning, evidence))
        if len(rows) >= max(k, rerank):
            break
    if rerank and rows:
        try:
            from llm_wiki.retrieval._rerank import rerank_scores
            values = rerank_scores(query, [row["text"] for row in rows])
            for row, value in zip(rows, values):
                row["rerank_score"] = value
            rows.sort(key=lambda row: (-row["rerank_score"], -row["score"], row["id"]))
        except (ImportError, OSError, RuntimeError):
            pass
    return rows[:k]


def _fail_closed(vault: Path | None) -> None:
    # A stale concept index or illegally committed risky edge must never look like
    # a normal successful query. Other health checks remain visible via health.
    stale = concept_index.is_stale(vault)
    if stale:
        raise RuntimeError(f"v2 query refused: {stale}; run wiki-concepts build")
    from llm_wiki.v2.health import check_v2_health
    issues = check_v2_health(vault)
    blocking = [issue for issue in issues if any(marker in issue for marker in (
        "artifact", "stale", "missing chunk", "source quote", "risky relation",
        "cycle", "has no primary", "dangling", "invalid lifecycle", "tree locations",
        "build identity",
    ))]
    if blocking:
        raise RuntimeError(f"v2 query refused: {blocking[0]}; run wiki-concepts build / wiki-net build")


def _rrf_scores(seeds: list[tuple[float, Concept]]) -> dict[str, float]:
    # concept_index already fuses dense and BM25 values; rank fusion keeps query
    # composition scale-free when tree and graph signals are added below.
    return {concept.id: 2.0 / (RRF_K + rank) for rank, (_, concept) in enumerate(seeds)}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\w가-힣]+", text.lower()))


def _tree_scores(query: str, store: NetStore) -> dict[str, float]:
    q = _tokens(query)
    if not q:
        return {}
    topic_relevance = {node.id: len(q & _tokens(node.label)) for node in store.nodes()}
    out: dict[str, float] = {}
    for edge in store.edges():
        if edge.type in {EdgeType.PRIMARY_TOPIC_OF.value, EdgeType.SECONDARY_TOPIC_OF.value}:
            rel = topic_relevance.get(edge.source, 0)
            if rel:
                out[edge.target] = out.get(edge.target, 0.0) + min(0.01, rel / 200)
    return out


def _graph_scores(seed_scores: dict[str, float], store: NetStore, historical: bool) -> dict[str, float]:
    out: dict[str, float] = {}
    allowed = {RelationType.SUPPORTS.value, RelationType.COMPLEMENTS.value, RelationType.DUPLICATE_OF.value}
    if historical:
        allowed.add(RelationType.SUPERSEDES.value)
    top = {concept_id for concept_id, _ in sorted(seed_scores.items(), key=lambda row: -row[1])[:12]}
    for edge in store.edges():
        if edge.type != EdgeType.RELATES_TO.value or edge.relation not in allowed:
            continue
        if edge.source in top:
            out[edge.target] = out.get(edge.target, 0.0) + 0.004
        if edge.target in top and edge.relation != RelationType.SUPERSEDES.value:
            out[edge.source] = out.get(edge.source, 0.0) + 0.004
    return out


def _evidence(concept_id: str, store: NetStore) -> list[dict]:
    return [{"relation": edge.relation, "concept_id": edge.target if edge.source == concept_id else edge.source}
            for edge in store.edges()
            if edge.type == EdgeType.RELATES_TO.value and concept_id in {edge.source, edge.target}]


def auto_decision(vault: Path | None, query: str, k: int = 8, historical: bool = False,
                  thresholds_path: Path | None = None) -> dict:
    rows = recall(vault, query, k=max(k, 10), historical=historical)
    if not rows:
        return {"decision": "none", "reason": "no-candidates", "results": []}
    signals = {row["concept"].id: row for row in concept_index.search_with_signals(
        vault, query, k=max(k * 5, 30)
    )}
    candidates = [retrieval_policy.Candidate(
        id=row["id"], score=float(signals.get(row["id"], {}).get("dense_score", 0.0)), meta=row
    ) for row in rows]
    threshold_file = retrieval_policy.resolve_thresholds_path(thresholds_path, content_root(vault))
    thresholds = retrieval_policy.load_thresholds(threshold_file)
    model_file = concept_index.artifacts.artifact_path("concept_embeddings/model.txt", vault)
    semantic_index = model_file.exists() and model_file.read_text(encoding="utf-8").strip().startswith("Qwen/")
    if not semantic_index:
        decision = retrieval_policy.decide(candidates, thresholds, reranker_available=False)
        if decision.kind == "answer":  # defensive: decide currently cannot answer without rerank
            decision = retrieval_policy.Decision("review", tuple(candidates[:5]), "uncalibrated-concept-embedding")
        elif decision.kind == "review":
            decision = retrieval_policy.Decision("review", decision.candidates,
                                                   "uncalibrated-concept-embedding")
        return _decision_payload(decision)
    try:
        from llm_wiki.retrieval._rerank import rerank_scores
        reranked = rerank_scores(query, [candidate.meta["text"] for candidate in candidates])
    except (ImportError, OSError, RuntimeError):
        return _decision_payload(retrieval_policy.decide(candidates, thresholds, False))
    candidates = [retrieval_policy.Candidate(candidate.id, candidate.score, candidate.meta, score)
                  for candidate, score in zip(candidates, reranked)]
    candidates.sort(key=lambda candidate: candidate.rerank_score, reverse=True)
    decision = retrieval_policy.decide(candidates, thresholds, True)
    if decision.kind == "answer" and any(candidate.meta.get("state") == ConceptState.DISPUTED.value
                                          for candidate in candidates[:3]):
        decision = retrieval_policy.Decision("review", tuple(candidates[:5]), "disputed-evidence")
    return _decision_payload(decision)


def _decision_payload(decision: retrieval_policy.Decision) -> dict:
    return {"decision": decision.kind, "reason": decision.reason,
            "results": [{**candidate.meta, "confidence": round(candidate.score, 6),
                         "rerank_score": candidate.rerank_score}
                        for candidate in decision.candidates]}


def run_cli_query(query: str, vault: Path | None = None, k: int = 8, historical: bool = False,
                  as_json: bool = False, auto: bool = False,
                  thresholds_path: Path | None = None, rerank: int = 0) -> int:
    if auto:
        payload = auto_decision(vault, query, k, historical, thresholds_path)
        if as_json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(f"decision: {payload['decision']} ({payload['reason']})")
            for row in payload["results"]:
                print(f"  [{row['confidence']:.4f}] {row['id']} ({row['state']})")
        return 0
    rows = recall(vault, query, k=k, historical=historical, rerank=rerank)
    if as_json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    for row in rows:
        warn = f"\n        ! {row['warning']}" if row.get("warning") else ""
        print(f"[{row['score']:.4f}] {row['id']} ({row['state']})\n        {row['summary']}\n        quote: {row['source_quote']}{warn}")
    if not rows:
        print("No v2 concepts matched.")
    return 0


def _row(score: float, concept: Concept, warning: str | None, evidence: list[dict]) -> dict:
    return {"id": concept.id, "score": round(float(score), 6), "document_id": concept.document_id,
            "chunk_id": concept.chunk_id, "state": concept.state, "text": concept.text,
            "summary": concept.summary, "source_quote": concept.source_quote,
            "heading_path": concept.heading_path, "warning": warning, "evidence": evidence}
