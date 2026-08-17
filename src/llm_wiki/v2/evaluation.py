"""Versioned v2 evaluation for extraction, placement, relations, lifecycle, and safety."""
from __future__ import annotations

import json
from pathlib import Path

from llm_wiki.v2 import artifacts
from llm_wiki.v2.concept_store import read_concepts
from llm_wiki.v2.net_store import NetStore
from llm_wiki.v2.query import auto_decision, recall
from llm_wiki.v2.schemas import EdgeType, RISKY_RELATIONS, RelationType


def gold_path(vault: Path | None, name: str | None = None) -> Path:
    return Path(name) if name and Path(name).is_absolute() else artifacts.artifact_root(vault) / (name or "v2_gold.json")


def validate_gold(data: dict) -> list[str]:
    errors: list[str] = []
    for key in ("concepts", "relations", "queries"):
        if key not in data:
            errors.append(f"missing required {key} array")
        elif not isinstance(data[key], list):
            errors.append(f"{key} must be a list")
    for key in ("placements", "supersessions"):
        if key in data and not isinstance(data[key], list):
            errors.append(f"{key} must be a list")
    if not data.get("concepts"):
        errors.append("concepts must contain at least one gold case")
    if not data.get("queries"):
        errors.append("queries must contain at least one gold case")
    for row in data.get("concepts", []):
        if not isinstance(row, dict) or not any(row.get(key) for key in ("concept_id", "text", "source_quote")):
            errors.append("each concept case needs concept_id, text, or source_quote")
    for row in data.get("relations", []):
        if not isinstance(row, dict) or not all(row.get(key) for key in ("source", "target", "relation")):
            errors.append("each relation needs source, target, and relation")
    for row in data.get("queries", []):
        if not isinstance(row, dict) or not isinstance(row.get("query"), str) or not isinstance(row.get("expect", []), list):
            errors.append("each query needs query:string and expect:list")
    return errors


def evaluate(vault: Path | None, name: str | None = None, k: int = 8) -> dict:
    data = json.loads(gold_path(vault, name).read_text(encoding="utf-8"))
    errors = validate_gold(data)
    if errors:
        raise ValueError("; ".join(errors))
    concepts = {concept.id: concept for concept in read_concepts(vault)}
    store = NetStore(vault)
    relation_edges = [edge for edge in store.edges()
                      if edge.type == EdgeType.RELATES_TO.value and edge.relation]
    relation_set = {(edge.source, edge.target, edge.relation) for edge in relation_edges}

    concept_cases = data["concepts"]
    concept_hits = sum(_concept_match(case, concepts) for case in concept_cases)
    expected_ids = {case.get("concept_id") for case in concept_cases if case.get("concept_id")}
    scoped_predictions = [concept for concept in concepts.values()
                          if not expected_ids or concept.id in expected_ids]
    concept_precision = _rate(sum(concept.id in expected_ids for concept in scoped_predictions),
                              len(scoped_predictions)) if expected_ids else _rate(concept_hits, len(concept_cases))
    faithfulness = sum(_faithful(case, concepts) for case in concept_cases)

    placement_cases = data.get("placements", [])
    placement_hits = sum(concepts.get(case.get("concept_id")) is not None and
                         concepts[case["concept_id"]].primary_topic_id in set(
                             case.get("expected_topics", [case.get("primary_topic_id")]))
                         for case in placement_cases)
    placement_topk_hits = 0
    for case in placement_cases:
        concept = concepts.get(case.get("concept_id"))
        expected = set(case.get("expected_topics", [case.get("primary_topic_id")]))
        routed = ({concept.primary_topic_id, *concept.secondary_topic_ids} if concept else set())
        placement_topk_hits += bool(expected & routed)

    expected_relations = {(case["source"], case["target"], case["relation"])
                          for case in data.get("relations", [])}
    relation_hits = len(expected_relations & relation_set)
    relation_precision = _rate(relation_hits, len(relation_set)) if relation_set else (1.0 if not expected_relations else 0.0)
    relation_recall = _rate(relation_hits, len(expected_relations))
    per_type = {}
    for relation in RelationType:
        expected_type = {row for row in expected_relations if row[2] == relation.value}
        actual_type = {row for row in relation_set if row[2] == relation.value}
        hits = len(expected_type & actual_type)
        type_precision = _rate(hits, len(actual_type)) if actual_type else (1.0 if not expected_type else 0.0)
        type_recall = _rate(hits, len(expected_type))
        per_type[f"relation_{relation.value}_precision"] = type_precision
        per_type[f"relation_{relation.value}_recall"] = type_recall
        per_type[f"relation_{relation.value}_f1"] = (
            2 * type_precision * type_recall / (type_precision + type_recall)
            if type_precision + type_recall else 0.0
        )

    expected_supersessions = {(case["source"], case["target"], RelationType.SUPERSEDES.value)
                              for case in data.get("supersessions", [])}
    actual_supersessions = {row for row in relation_set if row[2] == RelationType.SUPERSEDES.value}
    supersession_hits = len(expected_supersessions & actual_supersessions)
    supersession_precision = (_rate(supersession_hits, len(actual_supersessions))
                              if actual_supersessions else (1.0 if not expected_supersessions else 0.0))
    supersession_recall = _rate(supersession_hits, len(expected_supersessions))

    current_ranks: list[int | None] = []
    historical_ranks: list[int | None] = []
    current_top1 = 0
    outdated = false_auto = 0
    for case in data["queries"]:
        historical = bool(case.get("historical"))
        rows = recall(vault, case["query"], k=k, historical=historical)
        expected = set(case.get("expect", []))
        rank = next((index + 1 for index, row in enumerate(rows) if row["id"] in expected), None)
        (historical_ranks if historical else current_ranks).append(rank)
        if not historical and rank == 1:
            current_top1 += 1
        if not historical and rows and rows[0]["state"] == "SUPERSEDED":
            outdated += 1
        if case.get("auto") or case.get("expect_none"):
            decision = auto_decision(vault, case["query"], k=k, historical=historical)
            ids = [row["id"] for row in decision["results"]]
            if decision["decision"] == "answer" and (case.get("expect_none") or not expected.intersection(ids[:1])):
                false_auto += 1

    review_items = store.review_items()
    proposals = store.proposals()
    approved = sum(proposal.status == "APPROVED" for proposal in proposals)
    open_reviews = sum(item.state == "OPEN" for item in review_items)
    total_queries = len(data["queries"])
    precision = relation_precision
    recall_rate = relation_recall
    metrics = {
        "concept_presence": _rate(concept_hits, len(concept_cases)),
        "concept_precision": concept_precision,
        "concept_recall": _rate(concept_hits, len(concept_cases)),
        "concept_faithfulness": _rate(faithfulness, len(concept_cases)),
        "placement_primary_accuracy": _rate(placement_hits, len(placement_cases)),
        "placement_topk_route_recall": _rate(placement_topk_hits, len(placement_cases)),
        "relation_precision": precision,
        "relation_recall": recall_rate,
        "relation_f1": 2 * precision * recall_rate / (precision + recall_rate) if precision + recall_rate else 0.0,
        "supersession_precision": supersession_precision,
        "supersession_recall": supersession_recall,
        "false_supersession_rate": 1.0 - supersession_precision,
        "current_hit_at_k": _rank_hit(current_ranks),
        "current_mrr": _mrr(current_ranks),
        "current_fact_accuracy": _rate(current_top1, len(current_ranks)),
        "historical_accuracy": _rank_hit(historical_ranks),
        "current_historical_hit_at_k": _rank_hit(current_ranks + historical_ranks),
        "mrr": _mrr(current_ranks + historical_ranks),
        "outdated_answer_rate": _rate(outdated, total_queries),
        "false_auto_answers": false_auto,
        "risky_unapproved": sum(1 for issue in store.health_issues() if "risky relation" in issue),
        "human_review_rate": _rate(len(review_items), len(proposals)),
        "human_approval_rate": _rate(approved, len(proposals)),
        "open_review_items": open_reviews,
        "n": total_queries,
    }
    metrics.update(per_type)
    return metrics


def _concept_match(case: dict, concepts: dict) -> bool:
    if case.get("concept_id"):
        return case["concept_id"] in concepts
    return any((not case.get("text") or concept.text == case["text"]) and
               (not case.get("source_quote") or concept.source_quote == case["source_quote"])
               for concept in concepts.values())


def _faithful(case: dict, concepts: dict) -> bool:
    candidates = [concepts[case["concept_id"]]] if case.get("concept_id") in concepts else list(concepts.values())
    return any(concept.source_quote and
               (not case.get("source_quote") or concept.source_quote == case["source_quote"])
               for concept in candidates)


def _rank_hit(ranks: list[int | None]) -> float:
    return _rate(sum(rank is not None for rank in ranks), len(ranks))


def _mrr(ranks: list[int | None]) -> float:
    return sum(1 / rank for rank in ranks if rank) / len(ranks) if ranks else 1.0


def _rate(hit: int, total: int) -> float:
    return hit / total if total else 1.0
