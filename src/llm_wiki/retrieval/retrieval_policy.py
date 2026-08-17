#!/usr/bin/env python3
"""Decide whether a recall result may be consumed automatically.

An agent that reads the wiki without a human in the loop needs a third option
besides "here are 8 results" and "nothing found": it needs to know when the top
hit is trustworthy enough to act on. This module is that decision, kept pure so
the boundary is testable with literal scores.

Three outcomes:

  answer  — one candidate clears both the absolute score floor and the margin
            over the runner-up. Safe to consume without asking.
  review  — plausible but not decisive. Hand the candidates to a human/agent.
  none    — nothing clears the floor. Say so; do not serve the best of a bad set.

**Fails closed.** Every uncertainty resolves toward `review`, never toward
`answer`: an unavailable reranker, an uncalibrated threshold file, a tie at the
top. A wrong automatic answer is far more expensive than an unnecessary review,
because it is indistinguishable from a right one at the point of use.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_REVIEW_K = 5

# A cross-encoder relevance score below this means the model itself does not
# assert that the passage answers the query. Answering automatically on that is
# unjustified no matter what a small calibration sample happened to contain — so
# this is a floor on the FIT, chosen from the model's semantics rather than from
# the data. (Our calibration split holds only 4 negatives; a floor fitted from
# 4 points does not generalise, and the test split must not be used to pick it.)
MIN_RERANK_FLOOR = 0.5

# Deliberately unreachable: with no calibration file on disk, `score` cannot be
# cleared by any cosine/RRF score, so the policy degrades to review/none rather
# than answering on a guess.
UNCALIBRATED = None  # set below, after Thresholds is defined


@dataclass(frozen=True)
class Candidate:
    id: str
    score: float
    meta: dict = field(default_factory=dict, compare=False)
    # Cross-encoder relevance for this candidate, when a reranker ran. `None`
    # means "not reranked" — not "scored zero".
    rerank_score: float = None


@dataclass(frozen=True)
class Thresholds:
    score: float   # absolute cosine floor the top candidate must clear to answer
    margin: float  # how far the top must beat the runner-up to answer
    none: float    # below this, return nothing at all
    rerank: float = 0.0  # cross-encoder floor; the second, independent signal

    def as_json(self) -> dict:
        return {"score": self.score, "margin": self.margin, "none": self.none,
                "rerank": self.rerank}


UNCALIBRATED = Thresholds(score=float("inf"), margin=float("inf"), none=0.0,
                          rerank=float("inf"))


@dataclass(frozen=True)
class Decision:
    kind: str      # "answer" | "review" | "none"
    candidates: tuple = ()
    reason: str = ""

    def as_json(self) -> dict:
        return {
            "decision": self.kind,
            "reason": self.reason,
            "results": [{"id": c.id, "score": round(c.score, 6)} for c in self.candidates],
        }


@dataclass(frozen=True)
class LabeledRecord:
    """One evaluated query: what retrieval returned, and what was actually right."""

    query: str
    candidates: tuple
    correct_ids: frozenset
    expect_none: bool
    reranked: bool = True  # did the cross-encoder actually run for this record?


def decide(candidates, thresholds: Thresholds, reranker_available: bool,
           review_k: int = DEFAULT_REVIEW_K) -> Decision:
    """`candidates` MUST already be in final ranked order (best first).

    This function does not re-sort. Sorting here by `score` would silently
    discard the cross-encoder's ordering and re-rank by cosine — the very signal
    the reranker exists to override. Order comes from retrieval; `score` is the
    absolute confidence used only for the thresholds.
    """
    ranked = list(candidates)
    if not ranked:
        return Decision("none", (), "no-candidates")

    top = ranked[0]
    runner_up = ranked[1].score if len(ranked) > 1 else float("-inf")

    if top.score < thresholds.none:
        return Decision("none", (), "below-none-threshold")

    shortlist = tuple(ranked[:review_k])
    if not reranker_available:
        # The reranker is what turns "the retriever liked this" into "this
        # actually answers the question". Without it we have no basis for the
        # stronger claim, regardless of how confident the raw score looks.
        return Decision("review", shortlist, "reranker-unavailable")
    # When a cross-encoder score IS present, both signals must agree: cosine
    # says "this embeds close to the query", the cross-encoder says "this
    # actually answers it", and requiring only the first let a page the reranker
    # scored ~0 be answered automatically. The `is not None` guard means the
    # gate is skipped when the top candidate carries no rerank score at all —
    # a combination production never builds, because every caller passes
    # `reranker_available=result.reranked`, so a non-reranked record has already
    # returned `review` above. Do not restate this as an unconditional
    # two-signal requirement of `decide()`; see docs/RETRIEVAL.md.
    if top.rerank_score is not None and top.rerank_score < thresholds.rerank:
        return Decision("review", shortlist, "below-rerank-threshold")
    if top.score < thresholds.score:
        return Decision("review", shortlist, "below-answer-threshold")
    if (top.score - runner_up) < thresholds.margin:
        return Decision("review", shortlist, "top-two-margin")
    return Decision("answer", (top,), "clear-margin")


# --------------------------------------------------------------------------
# threshold persistence
# --------------------------------------------------------------------------

# The packaged calibration file ships INSIDE the distribution (populated by
# the evaluation pipeline), not inside any particular vault — a vault has no
# privileged place of its own to keep one. This is the fallback of last
# resort in `resolve_thresholds_path` below.
PACKAGED_THRESHOLDS_PATH = Path(__file__).resolve().parent.parent / "evaluation" / "auto_thresholds.json"


def resolve_thresholds_path(explicit, content_root: Path) -> Path:
    """Pick which `auto_thresholds.json` a caller should read, in one place.

    There used to be three call sites (`wiki-recall --auto`, `wiki-eval
    --auto`, `wiki-gate`) that each resolved this independently, and they
    disagreed: one read only the packaged file, one read only a vault-root
    file with no packaged fallback (so it silently measured `UNCALIBRATED`
    out of the box), and one read only the packaged file with no override at
    all. An evaluation harness that does not resolve the same file the same
    way as production measures something other than production.

    Precedence:
      1. `explicit` (e.g. a `--thresholds` flag), used exactly as given —
         the caller asked for a specific file, so no further guessing.
      2. `<content_root>/auto_thresholds.json`, if it exists — this is what
         makes a user's own `wiki-eval --calibrate` output actually take
         effect without having to pass `--thresholds` every time.
      3. `PACKAGED_THRESHOLDS_PATH` — the default shipped with the
         distribution, so an uncalibrated vault still gets a real (if
         generic) calibration instead of silently falling back to
         `UNCALIBRATED`.
    """
    if explicit is not None:
        return Path(explicit)
    vault_candidate = Path(content_root) / "auto_thresholds.json"
    if vault_candidate.exists():
        return vault_candidate
    return PACKAGED_THRESHOLDS_PATH


def write_thresholds(path: Path, thresholds: Thresholds) -> None:
    Path(path).write_text(json.dumps(thresholds.as_json(), indent=2) + "\n", encoding="utf-8")


def load_thresholds(path: Path) -> Thresholds:
    """Read a thresholds file, failing closed on anything short of a complete,
    well-formed set.

    A missing file was already handled this way; `--thresholds` now lets a
    caller point this at an arbitrary path, so the same guarantee must extend
    to a file that exists but is truncated, malformed, missing a required key,
    or not actually a file (e.g. a directory) — none of those may raise into
    `wiki-recall --auto` and none of them may fall back to a permissive
    default. `rerank` is the one optional key (defaults to 0.0); the other
    three are required, not invented, so a partial file is treated the same as
    an absent one rather than silently getting a lenient score/margin/none.
    """
    p = Path(path)
    if not p.exists():
        return UNCALIBRATED
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return Thresholds(score=float(data["score"]), margin=float(data["margin"]),
                          none=float(data["none"]), rerank=float(data.get("rerank", 0.0)))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"retrieval_policy: could not read thresholds from {p} ({exc}); "
              "failing closed to UNCALIBRATED", file=sys.stderr)
        return UNCALIBRATED


# --------------------------------------------------------------------------
# scoring and calibration
# --------------------------------------------------------------------------


def _is_false_answer(record: LabeledRecord, decision: Decision) -> bool:
    if decision.kind != "answer":
        return False
    if record.expect_none:
        return True  # answered a question the corpus cannot answer
    return decision.candidates[0].id not in record.correct_ids


def score_records(records, thresholds: Thresholds, reranker_available: bool = None) -> dict:
    """`reranker_available=None` (default) uses each record's own flag, which is
    what production does. Passing an explicit bool overrides it for tests."""
    n = len(records) or 1
    answers = reviews = nones = 0
    correct_answers = correct_nones = false_answers = 0
    for r in records:
        available = r.reranked if reranker_available is None else reranker_available
        d = decide(r.candidates, thresholds, available)
        if d.kind == "answer":
            answers += 1
            if _is_false_answer(r, d):
                false_answers += 1
            else:
                correct_answers += 1
        elif d.kind == "review":
            reviews += 1
        else:
            nones += 1
            if r.expect_none:
                correct_nones += 1
    return {
        "n": len(records),
        "answers": answers, "correct_answers": correct_answers,
        "false_answers": false_answers,
        "reviews": reviews, "nones": nones, "correct_nones": correct_nones,
        "answer_rate": answers / n, "review_rate": reviews / n, "none_rate": nones / n,
    }


def _observed_values(records) -> tuple:
    """Candidate threshold values = the score/margin/gap values actually seen.

    Enumerating observed values (rather than a fixed grid) means the fit lands
    exactly on a real decision boundary — no interpolation between points that
    never occur.
    """
    scores, margins = {0.0}, {0.0}
    for r in records:
        ranked = sorted(r.candidates, key=lambda c: -c.score)
        if not ranked:
            continue
        top = ranked[0].score
        scores.add(top)
        # nextafter-style epsilon: a threshold must be able to sit *just above*
        # an observed value to exclude it.
        scores.add(top + 1e-9)
        runner_up = ranked[1].score if len(ranked) > 1 else float("-inf")
        gap = top - runner_up
        if gap != float("inf"):
            margins.add(gap)
            margins.add(gap + 1e-9)
    return sorted(scores), sorted(margins)


def calibrate(records) -> Thresholds:
    """Fit (score, margin, none) so that NO record produces a false answer.

    Search order matters: among all safe pairs we take the one that answers the
    most cases, and break ties toward the *higher* thresholds. Ties broken
    upward keep the fit from sitting exactly on the boundary of the riskiest
    record it just barely tolerated.
    """
    if not records:
        raise ValueError("no records to calibrate on")

    # `none` must sit above the loudest confidently-wrong negative, so those
    # queries abstain instead of reviewing noise.
    negative_tops = [max((c.score for c in r.candidates), default=0.0)
                     for r in records if r.expect_none]
    none_threshold = (max(negative_tops) + 1e-9) if negative_tops else 0.0

    # Same rule for the cross-encoder floor: sit just above the loudest thing a
    # negative query could produce, so an unanswerable question cannot clear it.
    negative_rr = [c.rerank_score for r in records if r.expect_none
                   for c in r.candidates if c.rerank_score is not None]
    rerank_threshold = max((max(negative_rr) + 1e-9) if negative_rr else 0.0,
                           MIN_RERANK_FLOOR)

    score_values, margin_values = _observed_values(records)
    best = None
    for s in score_values:
        for m in margin_values:
            thresholds = Thresholds(score=s, margin=m, none=none_threshold,
                                    rerank=rerank_threshold)
            stats = score_records(records, thresholds)
            if stats["false_answers"]:
                continue
            key = (stats["correct_answers"], s, m)
            if best is None or key > best[0]:
                best = (key, thresholds)

    if best is None:  # pragma: no cover - score_records always admits the max
        return UNCALIBRATED
    return best[1]
