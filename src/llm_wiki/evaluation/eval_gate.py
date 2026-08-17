#!/usr/bin/env python3
"""Fail when retrieval quality regresses against a committed baseline.

Why: on 2026-07-31 a ranking change (page-level RRF instead of chunk-level)
dropped pattern Hit@8 from 89.5% to 65.8% and entity Hit@3 from 60% to 20%.
Nothing caught it except a human re-running the eval by hand. CI validates the
gold schema but never measures ranking, so the next such change merges silently.

  wiki-gate                 # measure and compare (exit 1 on regression)
  wiki-gate --update        # rewrite the baseline (justify it!)
  wiki-gate --json

NOT a pytest test: measuring means embedding potentially hundreds of queries,
which would drag the model into a suite that must stay torch-free. The
comparison arithmetic *is* unit-tested (tests/test_eval_gate.py); this file is
the runner.

Run it before pushing a change to anything under the retrieval path
(`_retrieve.py`, `embed_index.py`, `retrieval_policy.py`, chunking, the gold set).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from llm_wiki import config
from llm_wiki.paths import VAULT_ROOT, content_root

BASELINE_PATH = content_root() / "eval_baseline.json"
DEFAULT_TOLERANCE_PP = 3.0
# Baselines are stored rounded to 6 decimals, so an exact one-case drop can come
# out as 0.333334 vs a 1/3 tolerance and fire spuriously. Absorb the rounding.
EPSILON = 1e-6
RATE_METRICS = ("hit@1", "hit@3", "hit@8")
# Safety property, not a quality metric: no tolerance whatsoever.
ZERO_TOLERANCE_METRICS = ("false_answers_gated",)


@dataclass(frozen=True)
class Regression:
    slice: str
    metric: str
    detail: str


def tolerance_for(n: int, tolerance_pp: float = DEFAULT_TOLERANCE_PP) -> float:
    """Allowed drop for a slice of `n` cases.

    A flat percentage is wrong for small slices: entity has n=5, so a single
    case is 20 percentage points and a 3pp gate would fire on noise. Allow
    whichever is larger — one case, or the flat floor.
    """
    flat = tolerance_pp / 100.0
    if n <= 0:
        return flat
    return max(flat, 1.0 / n)


def compare(baseline: dict, current: dict,
            tolerance_pp: float = DEFAULT_TOLERANCE_PP) -> list:
    """Regressions of `current` against `baseline`. Empty list == pass."""
    out: list = []
    for name, base in baseline.items():
        cur = current.get(name)
        if cur is None:
            out.append(Regression(name, "-", f"slice missing from the new run (baseline n={base.get('n')})"))
            continue

        base_n, cur_n = int(base.get("n", 0)), int(cur.get("n", 0))
        if cur_n < base_n:
            # A shrinking corpus hides regressions: the rate can hold while
            # coverage falls, so treat lost cases as a regression of its own.
            out.append(Regression(name, "n", f"gold cases fell {base_n} -> {cur_n}"))

        for metric in ZERO_TOLERANCE_METRICS:
            if metric in base and cur.get(metric, 0) > base[metric]:
                out.append(Regression(name, metric,
                                      f"{base[metric]} -> {cur[metric]} (no tolerance: safety property)"))

        tol = tolerance_for(base_n, tolerance_pp)
        for metric in RATE_METRICS:
            if metric not in base or metric not in cur:
                continue
            drop = float(base[metric]) - float(cur[metric])
            if drop > tol + EPSILON:
                cases = drop * base_n
                out.append(Regression(
                    name, metric,
                    f"{base[metric]:.1%} -> {cur[metric]:.1%} "
                    f"(-{drop:.1%} = {cases:.0f} case(s); tolerance {tol:.1%})"))
    return out


def load_baseline(path: Path = BASELINE_PATH) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"no baseline at {p} — create it with `wiki-gate --update`")
    return json.loads(p.read_text(encoding="utf-8"))["slices"]


def write_baseline(path: Path, slices: dict, note: str = "") -> None:
    import datetime as dt

    payload = {
        "generated": dt.date.today().isoformat(),
        "note": note or "regenerate only with a measured justification — see the commit log",
        "slices": slices,
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")


# --------------------------------------------------------------------------
# measurement (imports the real retrieval path)
# --------------------------------------------------------------------------


def measure(gold: str = None, k: int = 8, mode: str = "hybrid",
            rerank: int = 0, thresholds: str = None) -> dict:
    from llm_wiki.evaluation import eval as eval_mod
    from llm_wiki.evaluation import eval_schema

    gold = gold or config.load(VAULT_ROOT).gold
    cases = eval_schema.load_gold(eval_mod.resolve_path(gold))
    results = []
    for case in cases:
        ids = eval_mod.ranked_ids(case.q, k, mode, rerank)
        results.append((case, eval_schema.evaluate_case(case, ids)))

    slices = {"overall": eval_schema.aggregate(results)}
    for axis, key in (("layer", lambda c: c.layer), ("difficulty", lambda c: c.difficulty)):
        for name, agg in eval_schema.slice_by(results, key).items():
            slices[f"{axis}:{name}"] = agg

    # Safety gate: false answers on the categories pinned at zero — negative and
    # ambiguous. See ZERO_TOLERANCE_METRICS: this one is a safety property, so it
    # is compared with no tolerance at all, unlike the quality rates above.
    gated = _gated_false_answers([c for c in cases if c.split == "test"], k, mode, rerank, thresholds)
    slices["auto"] = gated
    return {name: {m: round(v, 6) if isinstance(v, float) else v for m, v in s.items()}
            for name, s in slices.items()}


def _gated_false_answers(cases: list, k: int, mode: str, rerank: int,
                         thresholds: str = None) -> dict:
    from llm_wiki.retrieval._retrieve import search_with_confidence
    from llm_wiki.retrieval import retrieval_policy

    # Same resolution rule as `wiki-recall --auto` and `wiki-eval --auto`
    # (see retrieval_policy.resolve_thresholds_path): an explicit override,
    # else the vault's own calibration, else the packaged default. Without
    # this the gate could only ever read the packaged file, even after a
    # user ran `wiki-eval --calibrate` on their own gold set — exactly what
    # the packaged file's own "_note" tells them to do.
    thresholds_path = retrieval_policy.resolve_thresholds_path(thresholds, content_root())
    thresholds = retrieval_policy.load_thresholds(thresholds_path)
    pool = rerank or 10
    gated = total = 0
    for case in cases:
        result = search_with_confidence(case.q, k=k, mode=mode, rerank=pool)
        cands = [retrieval_policy.Candidate(
            id=m["id"], score=float(conf), meta=m,
            rerank_score=float(rank) if result.reranked else None)
            for rank, conf, m in result.hits]
        decision = retrieval_policy.decide(cands, thresholds, result.reranked)
        if case.category not in ("negative", "ambiguous"):
            continue
        total += 1
        if decision.kind == "answer" and (
                case.expect_none or decision.candidates[0].id not in set(case.expect)):
            gated += 1
    return {"n": total, "false_answers_gated": gated}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", action="store_true", help="gate Concept/NET v2 metrics")
    ap.add_argument("--gold", default=None,
                    help="gold file (absolute path, or relative to the content root); "
                         "defaults to the vault's configured [eval] gold")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "dense"])
    ap.add_argument("--rerank", type=int, default=0)
    ap.add_argument("--thresholds", type=Path, default=None,
                    help="threshold file for the false-answer safety gate, used as given; "
                         "defaults to the vault's own auto_thresholds.json if present, "
                         "else the packaged default (see retrieval_policy.resolve_thresholds_path)")
    ap.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    ap.add_argument("--tolerance-pp", type=float, default=DEFAULT_TOLERANCE_PP)
    ap.add_argument("--update", action="store_true", help="rewrite the baseline from this run")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.v2:
        from llm_wiki.v2 import artifacts
        from llm_wiki.v2.evaluation import evaluate
        path = args.baseline if args.baseline != BASELINE_PATH else artifacts.artifact_path("v2_eval_baseline.json")
        current = evaluate(VAULT_ROOT, args.gold, args.k)
        if args.update:
            path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"wrote v2 baseline: {path}")
            return 0
        if not path.exists():
            print(f"no v2 baseline at {path}; create it with wiki-gate --v2 --update")
            return 1
        baseline = json.loads(path.read_text(encoding="utf-8"))
        quality_metrics = (
            "concept_precision", "concept_recall", "concept_faithfulness",
            "placement_primary_accuracy", "placement_topk_route_recall",
            "relation_precision", "relation_recall", "relation_f1",
            "supersession_precision", "supersession_recall", "current_hit_at_k", "current_mrr",
            "current_fact_accuracy", "historical_accuracy", "current_historical_hit_at_k", "mrr",
        )
        quality_metrics += tuple(name for name in baseline
                                 if name.startswith("relation_")
                                 and name.endswith(("_precision", "_recall", "_f1")))
        tolerance = args.tolerance_pp / 100
        regressions = []
        for name in quality_metrics:
            current_value, baseline_value = current.get(name), baseline.get(name)
            if isinstance(current_value, (int, float)) and isinstance(baseline_value, (int, float)):
                if current_value + tolerance < baseline_value:
                    regressions.append(name)
        if (current.get("outdated_answer_rate", 0) > baseline.get("outdated_answer_rate", 0)
                or current.get("false_supersession_rate", 0) > baseline.get("false_supersession_rate", 0)
                or current.get("false_auto_answers", 0) > baseline.get("false_auto_answers", 0)
                or current.get("risky_unapproved", 0)):
            regressions.append("safety")
        print(json.dumps({"current": current, "regressions": regressions}, ensure_ascii=False, indent=2))
        return 1 if regressions else 0

    current = measure(args.gold, args.k, args.mode, args.rerank, args.thresholds)

    if args.update:
        write_baseline(args.baseline, current)
        print(f"OK: baseline written to {args.baseline} ({len(current)} slices)")
        print("  WARNING: a baseline change must be justified by measurement - record it in the commit log")
        return 0

    regressions = compare(load_baseline(args.baseline), current, args.tolerance_pp)

    if args.json:
        print(json.dumps({"ok": not regressions, "current": current,
                          "regressions": [r.__dict__ for r in regressions]}, ensure_ascii=False))
        return 1 if regressions else 0

    for r in regressions:
        print(f"ERROR [{r.slice}] {r.metric}: {r.detail}", file=sys.stderr)
    if regressions:
        print(f"\nERROR: {len(regressions)} retrieval regression(s) vs {args.baseline.name}",
              file=sys.stderr)
        return 1
    print(f"OK: no retrieval regression ({len(current)} slices vs {args.baseline.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
