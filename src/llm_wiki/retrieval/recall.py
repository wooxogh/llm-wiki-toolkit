#!/usr/bin/env python3
"""Semantic recall over the vault. Local, offline, NO API key.

  wiki-recall "how does the cache warm" [--k 8] [--layer domain] [--domain research]
              [--project project-a] [--status any] [--mode hybrid|dense]
              [--json] [--auto]

Hybrid (default) = dense embeddings + BM25 sparse, RRF-fused (see _retrieve.py).
Ask in either language; cross-lingual matching works.

`--auto` is for non-interactive consumption: instead of a result list it emits a
single answer/review/none decision (see retrieval_policy.py), failing closed to
`review` whenever confidence is insufficient or the reranker is unavailable.

Superseded pages are excluded by default; pass `--status any` to include them.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
from llm_wiki import telemetry
from llm_wiki.paths import content_root

CONTENT_ROOT = content_root()
from llm_wiki.retrieval import retrieval_policy
from llm_wiki.retrieval._retrieve import search

# Re-exported for callers/tests that want the packaged default without
# reaching into retrieval_policy directly. The actual resolution precedence
# (explicit -> vault-root -> packaged) lives in
# retrieval_policy.resolve_thresholds_path, used below and by eval.py/eval_gate.py
# so all three tools agree on which file "no override given" means.
THRESHOLDS_PATH = retrieval_policy.PACKAGED_THRESHOLDS_PATH
AUTO_RERANK_POOL = 10  # --auto always reranks; see run_auto()


def run_auto(args) -> int:
    from llm_wiki.retrieval._retrieve import search_with_confidence

    status = None if args.status == "any" else args.status
    # --auto ALWAYS reranks. The policy's guarantee is "answer only when the
    # cross-encoder also agrees", so running without it and then reporting the
    # reranker as available would make the fail-closed contract vacuous.
    pool = args.rerank or AUTO_RERANK_POOL
    started = time.perf_counter()
    result = search_with_confidence(
        args.query, k=args.k, layer=args.layer, domain=args.domain, mode=args.mode,
        rerank=pool, project=args.project, status=status,
        confidence=args.confidence)
    latency_ms = (time.perf_counter() - started) * 1000
    thresholds_path = retrieval_policy.resolve_thresholds_path(args.thresholds, CONTENT_ROOT)
    thresholds = retrieval_policy.load_thresholds(thresholds_path)
    # Order comes from the reranker; score is the absolute cosine confidence.
    candidates = [retrieval_policy.Candidate(
                      id=m["id"], score=float(conf), meta=m,
                      rerank_score=float(rank) if result.reranked else None)
                  for rank, conf, m in result.hits]
    decision = retrieval_policy.decide(
        candidates, thresholds, reranker_available=result.reranked
    )
    payload = decision.as_json()
    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(f"decision: {payload['decision']}  ({payload['reason']})")
        for row in payload["results"]:
            print(f"  [{row['score']:.4f}] {row['id']}")
    record_event(args, [m["id"] for _, _, m in result.hits], latency_ms, decision.kind)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--v2", action="store_true", help="recall over v2 atomic concepts")
    ap.add_argument("--vault", type=Path, help="v2 vault path (legacy mode uses WIKI_VAULT)")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--layer", help="filter: domain|pattern|entity|raw")
    ap.add_argument("--domain", help="filter: e.g. research|tooling (vault-specific)")
    ap.add_argument("--project", help="filter: only pages tagged for this repo")
    ap.add_argument("--confidence", help="filter: confirmed|provisional")
    ap.add_argument("--status", default="active",
                    help="active (default) | superseded | any")
    ap.add_argument("--historical", action="store_true",
                    help="v2 only: include superseded and archived concepts")
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "dense"])
    ap.add_argument("--rerank", type=int, default=0,
                    help="candidate pool for local cross-encoder rerank (0=off; e.g. 10)")
    ap.add_argument("--full", action="store_true", help="print path list only")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--auto", action="store_true",
                    help="emit one answer/review/none decision instead of a list")
    ap.add_argument("--thresholds", type=Path, default=None,
                    help="threshold file for --auto, used as given; without this, "
                         "the vault's own auto_thresholds.json is used if present, "
                         "else the packaged default")
    ap.add_argument("--telemetry", action="store_true",
                    help="record this recall event (also via WIKI_RECALL_TELEMETRY=1)")
    ap.add_argument("--label", choices=list(telemetry.LABELS),
                    help="annotate the recorded event with an outcome label")
    args = ap.parse_args()

    try:
        if args.v2:
            from llm_wiki.v2.query import run_cli_query
            return run_cli_query(args.query, vault=args.vault, k=args.k,
                                 historical=args.historical or args.status == "any",
                                 as_json=args.json, auto=args.auto,
                                 thresholds_path=args.thresholds, rerank=args.rerank)
        if args.auto:
            return run_auto(args)

        status = None if args.status == "any" else args.status
        started = time.perf_counter()
        ranked = search(args.query, k=args.k, layer=args.layer, domain=args.domain,
                        mode=args.mode, rerank=args.rerank, project=args.project,
                        status=status, confidence=args.confidence)
    except RuntimeError as exc:
        print(f"wiki-recall: {exc}", file=sys.stderr)
        return 1

    latency_ms = (time.perf_counter() - started) * 1000

    if args.json:
        print(json.dumps([
            {"id": m["id"], "score": round(float(sc), 6), "layer": m.get("layer"),
             "domain": m.get("domain"), "path": m["path"], "status": m.get("status"),
             "updated": m.get("updated"), "snippet": m.get("snippet", "")}
            for sc, m in ranked
        ], ensure_ascii=False))
    else:
        for sc, m in ranked:
            loc = f"{m['layer']}/{m.get('domain') or '-'}"
            if args.full:
                print(f"{sc:.4f}  {CONTENT_ROOT}/{m['path']}")
            else:
                print(f"[{sc:.4f}] {m['id']} ({loc})\n        {m['snippet']}\n"
                      f"        → {CONTENT_ROOT}/{m['path']}")
    record_event(args, [m["id"] for _, m in ranked], latency_ms, "list")
    return 0


def record_event(args, result_ids, latency_ms: float, decision: str) -> None:
    """Opt-in only — see telemetry.py. Records IDs, never page text."""
    if not telemetry.enabled(args.telemetry):
        return
    filters = {k: v for k, v in (("layer", args.layer), ("domain", args.domain),
                                 ("project", args.project), ("status", args.status),
                                 ("confidence", args.confidence)) if v}
    telemetry.append_event(telemetry.default_path(), telemetry.RecallEvent(
        query=args.query,
        filters=filters,
        result_ids=tuple(result_ids),
        latency_ms=round(latency_ms, 2),
        decision=decision,
        useful=args.label,
    ))


if __name__ == "__main__":
    raise SystemExit(main())
