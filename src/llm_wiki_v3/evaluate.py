"""Retrieval ablation evaluation for LLM-Wiki V3."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

from .config import load
from .io import configure_stdio, read_json, write_json
from .search import SearchEngine


BASE_VARIANTS = {
    "text_dense": ("text", "dense"),
    "text_dense_tree": ("text", "dense", "tree"),
    "text_dense_knn": ("text", "dense", "knn"),
    "full": ("text", "dense", "tree", "knn"),
}


def validate_gold(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, list) or not payload:
        raise ValueError("gold must be a non-empty JSON array")
    rows = []
    for index, row in enumerate(payload):
        if not isinstance(row, dict) or not str(row.get("query") or "").strip():
            raise ValueError(f"gold[{index}] requires a non-empty query")
        chunk_ids = row.get("relevant_chunk_ids") or []
        document_ids = row.get("relevant_document_ids") or []
        if not isinstance(chunk_ids, list) or not isinstance(document_ids, list):
            raise ValueError(f"gold[{index}] relevant ids must be arrays")
        if not chunk_ids and not document_ids and not row.get("expect_none", False):
            raise ValueError(f"gold[{index}] needs relevant ids or expect_none=true")
        if row.get("range_years") is not None and (
            not isinstance(row["range_years"], int) or row["range_years"] <= 0
        ):
            raise ValueError(f"gold[{index}] range_years must be a positive integer")
        rows.append(row)
    return rows


def _relevance_key(hit: dict[str, Any], row: dict[str, Any]) -> tuple[str, str] | None:
    chunk_id = str(hit.get("id") or "")
    document_id = str(hit.get("document_id") or "")
    if chunk_id in set(map(str, row.get("relevant_chunk_ids") or [])):
        return "chunk", chunk_id
    if document_id in set(map(str, row.get("relevant_document_ids") or [])):
        return "document", document_id
    return None


def _metrics(results: list[tuple[dict[str, Any], list[dict[str, Any]]]], k: int) -> dict[str, float]:
    hits, reciprocal, recalls, ndcgs = [], [], [], []
    for gold, returned in results:
        expect_none = bool(gold.get("expect_none", False))
        seen_relevance = set()
        flags = []
        for chunk in returned:
            relevance = _relevance_key(chunk, gold)
            is_new_relevant = relevance is not None and relevance not in seen_relevance
            flags.append(is_new_relevant)
            if relevance is not None:
                seen_relevance.add(relevance)
        if expect_none:
            correct = not returned
            hits.append(float(correct))
            reciprocal.append(float(correct))
            recalls.append(float(correct))
            ndcgs.append(float(correct))
            continue
        hit_positions = [index for index, flag in enumerate(flags, 1) if flag]
        hits.append(float(bool(hit_positions)))
        reciprocal.append(1.0 / hit_positions[0] if hit_positions else 0.0)
        relevant_total = len(set(gold.get("relevant_chunk_ids") or [])) + len(set(gold.get("relevant_document_ids") or []))
        recalls.append(min(sum(flags) / max(relevant_total, 1), 1.0))
        dcg = sum(1.0 / math.log2(index + 1) for index, flag in enumerate(flags, 1) if flag)
        ideal = sum(1.0 / math.log2(index + 1) for index in range(1, min(relevant_total, k) + 1))
        ndcgs.append(dcg / ideal if ideal else 0.0)
    count = max(len(results), 1)
    return {
        f"hit_at_{k}": sum(hits) / count,
        "mrr": sum(reciprocal) / count,
        f"recall_at_{k}": sum(recalls) / count,
        f"ndcg_at_{k}": sum(ndcgs) / count,
    }


def evaluate(engine: SearchEngine, gold: list[dict[str, Any]], *, k: int, rerank_pool: int = 0) -> dict[str, Any]:
    variants = dict(BASE_VARIANTS)
    if rerank_pool:
        variants["full_rerank"] = BASE_VARIANTS["full"]
    embedding_started = perf_counter()
    engine.prepare_queries(item["query"] for item in gold)
    embedding_seconds = perf_counter() - embedding_started
    report = {}
    for name, channels in variants.items():
        started = perf_counter()
        rows = []
        for item in gold:
            response = engine.search(
                item["query"],
                k=k,
                years=item.get("range_years"),
                rerank_pool=rerank_pool if name == "full_rerank" else 0,
                channels=channels,
            )
            rows.append((item, [hit.chunk for hit in response.hits]))
        elapsed = perf_counter() - started
        report[name] = {
            **_metrics(rows, k),
            "query_count": len(gold),
            "total_seconds": round(elapsed, 6),
            "mean_latency_ms": round(elapsed * 1000 / max(len(gold), 1), 4),
            "channels": list(channels),
            "rerank_pool": rerank_pool if name == "full_rerank" else 0,
        }
    return {
        "query_embedding_total_seconds": round(embedding_seconds, 6),
        "query_embedding_mean_ms": round(embedding_seconds * 1000 / max(len(gold), 1), 4),
        "variants": report,
    }


def main() -> int:
    configure_stdio()
    parser = argparse.ArgumentParser(description="Evaluate LLM-Wiki V3 retrieval variants")
    parser.add_argument("--vault", type=Path, default=None)
    parser.add_argument("--gold", type=Path, default=None)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--rerank", nargs="?", const=20, default=0, type=int, metavar="N")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        config = load(args.vault)
        gold_path = args.gold or (config.root / "eval_gold_v3.json")
        gold = validate_gold(read_json(gold_path))
        if args.validate_only:
            payload = {"ok": True, "gold": str(gold_path), "query_count": len(gold)}
        else:
            evaluation = evaluate(SearchEngine(config), gold, k=args.k, rerank_pool=args.rerank)
            payload = {
                "gold": str(gold_path),
                "k": args.k,
                **evaluation,
            }
            output = args.output or (config.artifact_dir / "eval" / "results.json")
            write_json(output, payload)
            payload["output"] = str(output)
    except (FileNotFoundError, ImportError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"wiki-eval: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, ensure_ascii=False) if args.json else json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
