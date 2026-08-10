#!/usr/bin/env python3
"""Retrieval eval harness for the vault's semantic recall.

Runs a gold set of (query -> expected page id) cases through the same retrieval
path as `wiki-recall` and scores Hit@k, MRR, and nDCG@3 — overall and per slice
(difficulty, layer, domain, category). Use it to measure the effect of any
change (model swap, reranker, chunking, hybrid) with numbers instead of vibes.

  wiki-eval                                       # the vault's configured gold file
  wiki-eval --gold eval_gold.json                 # a specific gold file
  wiki-eval --gold eval_gold.json --validate-only # no embedding
  wiki-eval --gold eval_gold.json --split test

A gold file naming neither `split` nor `difficulty` still loads — absent fields
are defaulted rather than rejected (see eval_schema).

`--validate-only` is schema/coverage checking only: it never embeds a query, so
it is safe in CI where the embedding store does not exist.

Exit code: 0 for measurement runs; 1 when --validate-only finds errors.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
from llm_wiki import config
from llm_wiki.evaluation import eval_schema
from llm_wiki.paths import VAULT_ROOT, content_root, index_path
import yaml

VAULT = VAULT_ROOT              # config root — `config.load(VAULT)` must find wiki.toml
CONTENT_ROOT = content_root()   # content root — where the vault's own data files live


def resolve_path(name: str) -> Path:
    """Find a named gold file: by absolute path first, else relative to the content root.

    There is no package-local `scripts/` directory to fall back to here — a
    gold file is per-vault data, not part of this package. (Thresholds files
    have their own resolution rule with a packaged fallback — see
    `retrieval_policy.resolve_thresholds_path`.)
    """
    path = Path(name)
    if path.is_absolute():
        return path
    return CONTENT_ROOT / name


def index_entries(vault: Path = VAULT) -> list:
    data = yaml.safe_load(index_path(vault).read_text(encoding="utf-8"))
    return data.get("entries", [])


def ranked_ids(query: str, depth: int, mode: str, rerank: int = 0) -> list:
    from llm_wiki.retrieval._retrieve import search

    return [m["id"] for _, m in search(query, k=depth, mode=mode, rerank=rerank)]


def _fmt(agg: dict) -> str:
    return (f"n={agg['n']:<4} Hit@1={agg['hit@1']:6.1%}  Hit@3={agg['hit@3']:6.1%}  "
            f"Hit@8={agg['hit@8']:6.1%}  MRR={agg['mrr']:.3f}  nDCG@3={agg['ndcg@3']:.3f}")


def print_validation(cases: list, entries: list, minimums: dict = None) -> int:
    errors = eval_schema.validate_gold(cases, entries)
    leaks = eval_schema.leaked_ids(cases)
    report = eval_schema.coverage_report(cases, entries)
    shortfalls = eval_schema.coverage_shortfalls(report, minimums)

    print(f"cases: {report['total']}  |  pages referenced: "
          f"{report['pages_referenced']}/{report['pages_total']}")
    for axis in ("by_split", "by_layer", "by_domain", "by_category", "by_difficulty"):
        print(f"  {axis[3:]:<11}: {report[axis]}")
    print(f"  {'project':<11}: {report['by_project']}")
    print(f"  recent-window: {report['recent_cases']} case(s) touch pages updated "
          f"since {report['recent_window_start']}")

    for label, items in (("validation error", errors), ("id leakage", leaks),
                         ("coverage shortfall", shortfalls)):
        if items:
            print(f"\n❌ {len(items)} {label}(s):")
            for item in items:
                print(f"  - {item}")
    ok = not (errors or leaks or shortfalls)
    print("\n✓ gold set valid and complete" if ok else "\n❌ gold set not ready")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10, help="ranking depth")
    ap.add_argument("--gold", default=config.load(VAULT_ROOT).gold,
                    help="gold file (absolute path, or relative to the content root)")
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "dense"])
    ap.add_argument("--rerank", type=int, default=0,
                    help="candidate pool for local cross-encoder rerank (0=off)")
    ap.add_argument("--validate-only", action="store_true",
                    help="schema + coverage check; embeds nothing")
    ap.add_argument("--split", choices=list(eval_schema.SPLITS),
                    help="restrict measurement to one split")
    ap.add_argument("--json", action="store_true", help="machine-readable metrics")
    ap.add_argument("--auto", action="store_true",
                    help="also score automatic answer/review/none decisions")
    ap.add_argument("--thresholds", type=Path, default=None,
                    help="threshold file, used as given; defaults to the vault's own "
                         "auto_thresholds.json if present, else the packaged default "
                         "(see retrieval_policy.resolve_thresholds_path)")
    ap.add_argument("--calibrate", metavar="OUT",
                    help="fit thresholds on the calibration split and write them to OUT")
    args = ap.parse_args()

    gold_path = resolve_path(args.gold)
    cases = eval_schema.load_gold(gold_path)
    entries = index_entries()

    if args.validate_only:
        return print_validation(cases, entries, config.load(VAULT).minimums)

    print(f"gold file: {gold_path.name}  |  mode: {args.mode}"
          + (f"  |  rerank pool={args.rerank}" if args.rerank else "")
          + (f"  |  split={args.split}" if args.split else ""))

    if args.calibrate:
        return run_calibration(cases, args)

    selected = [c for c in cases if not args.split or c.split == args.split]
    results = []
    for case in selected:
        ids = ranked_ids(case.q, args.k, args.mode, args.rerank)
        results.append((case, eval_schema.evaluate_case(case, ids)))

    overall = eval_schema.aggregate(results)
    slices = {
        "difficulty": eval_schema.slice_by(results, lambda c: c.difficulty),
        "layer": eval_schema.slice_by(results, lambda c: c.layer),
        "domain": eval_schema.slice_by(results, lambda c: c.domain),
        "category": eval_schema.slice_by(results, lambda c: c.category),
    }
    auto = run_auto(selected, args) if args.auto else None

    if args.json:
        print(json.dumps({"overall": overall, "slices": slices, "auto": auto},
                         ensure_ascii=False))
        return 0

    print(f"\nOVERALL  {_fmt(overall)}")
    for axis, buckets in slices.items():
        print(f"\nby {axis}:")
        for name, agg in buckets.items():
            print(f"  {name:<12} {_fmt(agg)}")

    misses = [(c, m) for c, m in results if not c.expect_none and (m.rank is None or m.rank > 3)]
    if misses:
        print(f"\n{len(misses)} low-ranked/missed case(s) (rank>3 or miss):")
        for c, m in misses:
            print(f"  [rank={m.rank}] {c.q}")
            print(f"        expect={list(c.expect)}")
    if auto:
        print_auto(auto)
    return 0


# --------------------------------------------------------------------------
# automatic-decision scoring (see retrieval_policy.py)
# --------------------------------------------------------------------------


AUTO_RERANK_POOL = 10  # must match recall.AUTO_RERANK_POOL


def _decision_records(cases: list, args) -> list:
    """(query, candidates, labels) with real retrieval scores, for fitting/scoring.

    Reranks exactly as `recall.py --auto` does — calibrating on un-reranked
    candidates and then serving reranked ones would fit thresholds to a
    distribution production never sees.
    """
    from llm_wiki.retrieval._retrieve import search_with_confidence
    from llm_wiki.retrieval import retrieval_policy

    pool = args.rerank or AUTO_RERANK_POOL
    records = []
    for case in cases:
        result = search_with_confidence(case.q, k=args.k, mode=args.mode, rerank=pool)
        # Candidate.score is the CONFIDENCE (absolute cosine), not the RRF rank
        # score — see _retrieve.page_confidence for why the two must differ.
        candidates = tuple(retrieval_policy.Candidate(
                               id=m["id"], score=float(conf), meta=m,
                               rerank_score=float(rank) if result.reranked else None)
                           for rank, conf, m in result.hits)
        records.append(retrieval_policy.LabeledRecord(
            query=case.q,
            candidates=candidates,
            correct_ids=frozenset(case.expect),
            expect_none=case.expect_none,
            reranked=result.reranked,
        ))
    return records


def calibration_out_path(name: str) -> Path:
    """Where `--calibrate OUT` writes, mirroring `resolve_path`'s rule exactly.

    Absolute is used as given. Relative is anchored at the CONTENT root, with
    any directory component preserved — `--calibrate auto_thresholds.json`
    (the documented command) must land where `resolve_thresholds_path` looks
    for it, and `--calibrate out/x.json` must keep `out/`.

    An earlier version anchored only when `out.parent` did not exist, which got
    both cases backwards: `auto_thresholds.json` has parent `.`, which always
    exists, so it silently landed in the cwd and was never picked up again; and
    `out/x.json` was collapsed to `<root>/x.json`, discarding the directory the
    caller asked for.
    """
    out = Path(name)
    return out if out.is_absolute() else CONTENT_ROOT / out


def run_calibration(cases: list, args) -> int:
    from llm_wiki.retrieval import retrieval_policy

    calibration = [c for c in cases if c.split == "calibration"]
    if not calibration:
        print("❌ no calibration-split cases — nothing to fit")
        return 1
    records = _decision_records(calibration, args)
    thresholds = retrieval_policy.calibrate(records)
    out = calibration_out_path(args.calibrate)
    out.parent.mkdir(parents=True, exist_ok=True)
    retrieval_policy.write_thresholds(out, thresholds)
    fitted = retrieval_policy.score_records(records, thresholds)
    print(f"calibrated on {len(records)} calibration case(s) -> {out}")
    print(f"  score={thresholds.score:.4f}  margin={thresholds.margin:.4f}  "
          f"none={thresholds.none:.4f}  rerank={thresholds.rerank:.4f}")
    print(f"  answer coverage={fitted['answer_rate']:.1%}  false answers={fitted['false_answers']}")
    return 0


def run_auto(cases: list, args) -> dict:
    from llm_wiki.retrieval import retrieval_policy

    thresholds_path = retrieval_policy.resolve_thresholds_path(args.thresholds, CONTENT_ROOT)
    thresholds = retrieval_policy.load_thresholds(thresholds_path)
    records = _decision_records(cases, args)
    return retrieval_policy.score_records(records, thresholds)


def print_auto(auto: dict) -> None:
    print("\nAUTOMATIC CONSUMPTION")
    print(f"  answer  : {auto['answers']:>3}  ({auto['answer_rate']:.1%})  correct={auto['correct_answers']}")
    print(f"  review  : {auto['reviews']:>3}  ({auto['review_rate']:.1%})")
    print(f"  none    : {auto['nones']:>3}  ({auto['none_rate']:.1%})  correct={auto['correct_nones']}")
    print(f"  ❌ FALSE ANSWERS: {auto['false_answers']}")


if __name__ == "__main__":
    raise SystemExit(main())
