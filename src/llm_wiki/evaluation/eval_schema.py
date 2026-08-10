#!/usr/bin/env python3
"""Gold-file schema, ranking metrics, and coverage accounting for wiki recall eval.

Pure and dependency-light on purpose (stdlib + math only): the whole scoring
layer is unit-testable without embedding anything, so a metric regression can be
localised to *ranking* rather than to the model or the store.

v2 gold schema — every field explicit, nothing inferred:

    {"q": "...", "expect": ["page-id"], "expect_none": false,
     "split": "test", "category": "indirect", "difficulty": "hard",
     "layer": "domain", "domain": "research", "projects": ["project-a"]}

`split` exists so thresholds are calibrated on cases the reported numbers are
NOT computed from. Calibrating and reporting on the same cases would make the
automatic-answer thresholds look safe purely by memorising them.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from llm_wiki import config

SPLITS = ("calibration", "test")
CATEGORIES = ("direct", "indirect", "ambiguous", "negative")
DIFFICULTIES = ("easy", "hard")
LAYERS = ("domain", "pattern", "entity", "raw")
DEFAULT_KS = (1, 3, 8)

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class GoldCase:
    q: str
    expect: tuple = ()
    expect_none: bool = False
    split: str = "test"
    category: str = "direct"
    difficulty: str = "easy"
    layer: str = ""
    domain: str = ""
    projects: tuple = ()

    def as_json(self) -> dict:
        out = {"q": self.q, "expect": list(self.expect), "expect_none": self.expect_none,
               "split": self.split, "category": self.category, "difficulty": self.difficulty}
        if self.layer:
            out["layer"] = self.layer
        if self.domain:
            out["domain"] = self.domain
        if self.projects:
            out["projects"] = list(self.projects)
        return out


@dataclass(frozen=True)
class CaseMetrics:
    hit_at_1: int
    hit_at_3: int
    hit_at_8: int
    reciprocal_rank: float
    ndcg_at_3: float
    rank: int | None


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def _case_from_raw(raw: dict, where: str) -> GoldCase:
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: case must be an object, got {type(raw).__name__}")
    q = raw.get("q")
    if not isinstance(q, str) or not q.strip():
        raise ValueError(f"{where}: 'q' must be a non-empty string")
    expect = raw.get("expect") or []
    if not isinstance(expect, list) or not all(isinstance(x, str) for x in expect):
        raise ValueError(f"{where}: 'expect' must be a list of page ids")
    expect_none = bool(raw.get("expect_none", False))
    if expect_none and expect:
        raise ValueError(f"{where}: expect_none cannot include expect ({q!r})")
    if not expect_none and not expect:
        raise ValueError(f"{where}: a case must have 'expect' or expect_none ({q!r})")

    # v1 compatibility: the two original gold files carry only q/expect/note.
    split = raw.get("split", "test")
    category = raw.get("category", "negative" if expect_none else "direct")
    difficulty = raw.get("difficulty", "easy")
    for name, value, allowed in (("split", split, SPLITS),
                                 ("category", category, CATEGORIES),
                                 ("difficulty", difficulty, DIFFICULTIES)):
        if value not in allowed:
            raise ValueError(f"{where}: {name} {value!r} not in {list(allowed)} ({q!r})")
    layer = raw.get("layer", "") or ""
    if layer and layer not in LAYERS:
        raise ValueError(f"{where}: layer {layer!r} not in {list(LAYERS)} ({q!r})")
    projects = raw.get("projects") or []
    if not isinstance(projects, list) or not all(isinstance(x, str) for x in projects):
        raise ValueError(f"{where}: 'projects' must be a list of strings ({q!r})")

    return GoldCase(q=q, expect=tuple(expect), expect_none=expect_none, split=split,
                    category=category, difficulty=difficulty, layer=layer,
                    domain=raw.get("domain", "") or "", projects=tuple(projects))


def load_gold(path: Path) -> list:
    """Parse and structurally validate a gold file. Raises ValueError on any
    violation — a silently-dropped malformed case would quietly inflate scores."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path}: gold file must be a JSON array")
    cases = [_case_from_raw(r, f"{Path(path).name}[{i}]") for i, r in enumerate(raw)]
    seen: dict = {}
    for i, c in enumerate(cases):
        key = c.q.strip()
        if key in seen:
            raise ValueError(f"{Path(path).name}[{i}]: duplicate query {c.q!r} (also [{seen[key]}])")
        seen[key] = i
    return cases


def validate_gold(cases: list, index_entries: list) -> list:
    """Semantic validation against the live index. Returns error strings."""
    by_id = {e["id"]: e for e in index_entries}
    errors: list = []
    for c in cases:
        for _id in c.expect:
            if _id not in by_id:
                errors.append(f"{c.q!r}: expected id {_id!r} does not exist in index.yaml")
                continue
            entry = by_id[_id]
            if entry.get("status") == "superseded":
                errors.append(f"{c.q!r}: expected id {_id!r} is superseded — default recall excludes it")
            if c.layer and entry.get("layer") != c.layer:
                errors.append(f"{c.q!r}: layer {c.layer!r} != {_id!r} layer {entry.get('layer')!r}")
            if c.domain and entry.get("domain") != c.domain:
                errors.append(f"{c.q!r}: domain {c.domain!r} != {_id!r} domain {entry.get('domain')!r}")
        if c.category == "negative" and not c.expect_none:
            errors.append(f"{c.q!r}: category 'negative' requires expect_none: true")
        if c.expect_none and c.category != "negative":
            errors.append(f"{c.q!r}: expect_none requires category 'negative'")
        if not c.expect_none and not c.layer:
            errors.append(f"{c.q!r}: missing coverage label 'layer'")
        if not c.expect_none and c.layer == "domain" and not c.domain:
            errors.append(f"{c.q!r}: layer 'domain' requires a 'domain' coverage label")
    return errors


def leaked_ids(cases: list) -> list:
    """Conceptual queries that literally spell out their own expected page id.

    A query containing its answer's id measures string matching, not recall.
    """
    out = []
    for c in cases:
        if c.category not in ("indirect", "ambiguous"):
            continue
        lowered = c.q.lower()
        for _id in c.expect:
            if _id.lower() in lowered:
                out.append(f"{c.q!r}: leaks expected id {_id!r}")
    return out


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def evaluate_case(case: GoldCase, ranked_ids: list, ks: tuple = DEFAULT_KS) -> CaseMetrics:
    """Score one case against a ranked id list.

    A negative (`expect_none`) case is correct exactly when retrieval returned
    nothing — it is scored on abstention, so it uses the same fields rather than
    a parallel metric nobody would look at.
    """
    if case.expect_none:
        correct = 1 if not ranked_ids else 0
        return CaseMetrics(hit_at_1=correct, hit_at_3=correct, hit_at_8=correct,
                           reciprocal_rank=float(correct), ndcg_at_3=float(correct),
                           rank=None)

    expect = set(case.expect)
    rank = next((i + 1 for i, x in enumerate(ranked_ids) if x in expect), None)
    hits = {k: (1 if rank and rank <= k else 0) for k in ks}
    rr = 1.0 / rank if rank else 0.0
    # Single graded-relevance level, so IDCG == 1 and nDCG@3 collapses to the
    # positional discount of the first relevant hit within the cutoff.
    ndcg3 = (1.0 / math.log2(rank + 1)) if (rank and rank <= 3) else 0.0
    return CaseMetrics(hit_at_1=hits.get(1, 0), hit_at_3=hits.get(3, 0),
                       hit_at_8=hits.get(8, 0), reciprocal_rank=rr,
                       ndcg_at_3=ndcg3, rank=rank)


def aggregate(results: list) -> dict:
    """Mean of each metric over (case, CaseMetrics) pairs."""
    n = len(results)
    if not n:
        return {"n": 0, "hit@1": 0.0, "hit@3": 0.0, "hit@8": 0.0, "mrr": 0.0, "ndcg@3": 0.0}
    return {
        "n": n,
        "hit@1": sum(m.hit_at_1 for _, m in results) / n,
        "hit@3": sum(m.hit_at_3 for _, m in results) / n,
        "hit@8": sum(m.hit_at_8 for _, m in results) / n,
        "mrr": sum(m.reciprocal_rank for _, m in results) / n,
        "ndcg@3": sum(m.ndcg_at_3 for _, m in results) / n,
    }


def slice_by(results: list, key) -> dict:
    """Aggregate per slice, e.g. slice_by(results, lambda c: c.difficulty)."""
    buckets: dict = {}
    for case, metrics in results:
        buckets.setdefault(key(case) or "-", []).append((case, metrics))
    return {name: aggregate(rows) for name, rows in sorted(buckets.items())}


# --------------------------------------------------------------------------
# coverage
# --------------------------------------------------------------------------


def _count(cases: list, key) -> dict:
    out: dict = {}
    for c in cases:
        out[key(c) or "-"] = out.get(key(c) or "-", 0) + 1
    return dict(sorted(out.items()))


def coverage_report(cases: list, index_entries: list, recent_days: int = 7) -> dict:
    """How the corpus is distributed, and how much of the vault it touches.

    `recent` counts cases whose expected page was updated in the newest
    `recent_days`-day window *of the vault itself* — the newest knowledge is the
    most likely to be wrong-but-confident, so it has to be represented.
    """
    by_id = {e["id"]: e for e in index_entries}
    dates = sorted(
        (str(e.get("updated")) for e in index_entries if _ISO_DATE_RE.match(str(e.get("updated") or ""))),
        reverse=True,
    )
    newest = dates[0] if dates else ""
    window = _window_start(newest, recent_days)

    projects: dict = {}
    for c in cases:
        for p in c.projects:
            projects[p] = projects.get(p, 0) + 1

    referenced = {i for c in cases for i in c.expect}
    recent = sum(
        1 for c in cases
        if any(str(by_id.get(i, {}).get("updated") or "") >= window for i in c.expect)
    )
    return {
        "total": len(cases),
        "by_layer": _count(cases, lambda c: c.layer),
        "by_domain": _count(cases, lambda c: c.domain),
        "by_category": _count(cases, lambda c: c.category),
        "by_difficulty": _count(cases, lambda c: c.difficulty),
        "by_split": _count(cases, lambda c: c.split),
        "by_project": dict(sorted(projects.items())),
        "pages_referenced": len(referenced),
        "pages_total": len(by_id),
        "recent_window_start": window,
        "recent_cases": recent,
    }


def _window_start(newest: str, days: int) -> str:
    import datetime as dt

    if not _ISO_DATE_RE.match(newest or ""):
        return ""
    d = dt.date.fromisoformat(newest) - dt.timedelta(days=days - 1)
    return d.isoformat()


# The corpus-shaped defaults live in `llm_wiki.config` (a `[eval.minimums]`
# table in wiki.toml can override them per vault) rather than as a second
# copy here — a vault's own domain names are unknowable to this package, so
# the default `domain` axis is deliberately empty rather than carrying one
# author's own domain quotas.
DEFAULT_MINIMUMS = config.DEFAULT_MINIMUMS


def coverage_shortfalls(report: dict, minimums: dict = None) -> list:
    """Which curation minimums the corpus does not yet meet."""
    mins = minimums or DEFAULT_MINIMUMS
    out = []
    if report["total"] < mins["total"]:
        out.append(f"total {report['total']} < {mins['total']}")
    if report["recent_cases"] < mins["recent_cases"]:
        out.append(f"recent-window cases {report['recent_cases']} < {mins['recent_cases']}")
    for axis in ("layer", "domain", "category"):
        for name, need in mins[axis].items():
            have = report[f"by_{axis}"].get(name, 0)
            if have < need:
                out.append(f"{axis}/{name} {have} < {need}")
    return out
