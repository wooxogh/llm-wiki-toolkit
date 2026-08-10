"""Answer / review / none decisions, reranker fallback, and threshold calibration.

Pure policy: every score here is a literal, so a decision boundary can never
shift silently under a retrieval change without this suite noticing.
"""
from __future__ import annotations

import json

import pytest

from llm_wiki.retrieval.retrieval_policy import (
    PACKAGED_THRESHOLDS_PATH,
    UNCALIBRATED,
    Candidate,
    LabeledRecord,
    Thresholds,
    calibrate,
    decide,
    load_thresholds,
    resolve_thresholds_path,
    score_records,
    write_thresholds,
)


def candidate(_id, score, **meta):
    return Candidate(id=_id, score=score, meta={"id": _id, **meta})


def thresholds(score=0.80, margin=0.15, none=0.30):
    return Thresholds(score=score, margin=margin, none=none)


def ids(decision):
    return [c.id for c in decision.candidates]


# --------------------------------------------------------------------------
# decisions
# --------------------------------------------------------------------------


def test_auto_answers_only_when_score_and_margin_clear():
    decision = decide(
        [candidate("right", 0.91), candidate("other", 0.62)],
        thresholds(score=0.80, margin=0.15, none=0.30),
        reranker_available=True,
    )
    assert decision.kind == "answer"
    assert ids(decision) == ["right"]


def test_auto_reviews_ambiguous_candidates():
    decision = decide(
        [candidate("one", 0.84), candidate("two", 0.80), candidate("three", 0.55)],
        thresholds(score=0.80, margin=0.15, none=0.30),
        reranker_available=True,
    )
    assert decision.kind == "review"
    assert ids(decision) == ["one", "two", "three"]
    assert decision.reason == "top-two-margin"


def test_decide_honours_the_caller_ranking_and_does_not_resort_by_score():
    """Candidates arrive in FINAL ranked order (reranker order when one ran).

    Re-sorting by score here silently threw the cross-encoder's ordering away and
    re-ranked by cosine — which is exactly the signal the reranker exists to
    override.
    """
    thr = thresholds(score=0.50, margin=0.15, none=0.30)
    pair = [candidate("a", 0.91), candidate("b", 0.62)]

    in_order = decide(pair, thr, reranker_available=True)
    reversed_order = decide(list(reversed(pair)), thr, reranker_available=True)

    # Same two candidates, opposite caller order -> different outcome. If decide
    # re-sorted by score, both calls would answer "a".
    assert in_order.kind == "answer" and ids(in_order) == ["a"]
    assert reversed_order.kind == "review"


def test_margin_is_measured_against_the_next_ranked_candidate_not_the_global_max():
    decision = decide(
        [candidate("reranker-top", 0.62), candidate("cosine-top", 0.91)],
        thresholds(score=0.50, margin=0.15, none=0.30),
        reranker_available=True,
    )

    # top(0.62) - next(0.91) is negative, so the margin cannot be cleared.
    assert decision.kind == "review"
    assert decision.reason == "top-two-margin"


def test_review_shortlist_preserves_caller_order(tmp_path):
    decision = decide(
        [candidate("c", 0.60), candidate("a", 0.84), candidate("b", 0.80)],
        thresholds(score=0.80, margin=0.15, none=0.30),
        reranker_available=True,
        review_k=3,
    )

    assert ids(decision) == ["c", "a", "b"]


def test_a_high_cosine_candidate_the_reranker_rejects_is_not_answered():
    """Two independent signals must agree. Ordering by the cross-encoder while
    thresholding only on cosine let a page the reranker scored ~0 through, purely
    because it embedded close to the query."""
    decision = decide(
        [Candidate(id="looks-close", score=0.95, meta={}, rerank_score=0.02),
         Candidate(id="other", score=0.40, meta={}, rerank_score=0.01)],
        Thresholds(score=0.80, margin=0.15, none=0.30, rerank=0.20),
        reranker_available=True,
    )

    assert decision.kind == "review"
    assert decision.reason == "below-rerank-threshold"


def test_both_signals_clear_means_answer():
    decision = decide(
        [Candidate(id="right", score=0.91, meta={}, rerank_score=0.98),
         Candidate(id="other", score=0.40, meta={}, rerank_score=0.01)],
        Thresholds(score=0.80, margin=0.15, none=0.30, rerank=0.20),
        reranker_available=True,
    )

    assert decision.kind == "answer"
    assert ids(decision) == ["right"]


def test_the_rerank_threshold_is_ignored_when_no_rerank_score_is_present():
    """Records from a non-reranked path carry no cross-encoder score; they are
    already forced to `review` by reranker_available=False, so the threshold
    must not double-fire on a missing value."""
    decision = decide(
        [Candidate(id="right", score=0.91, meta={}), candidate("other", 0.10)],
        Thresholds(score=0.80, margin=0.15, none=0.30, rerank=0.20),
        reranker_available=True,
    )

    assert decision.kind == "answer"


def test_calibration_fits_a_rerank_floor_above_the_loudest_negative():
    records = [
        LabeledRecord("neg", (Candidate("noise", 0.90, {}, rerank_score=0.11),),
                      frozenset(), True),
        LabeledRecord("pos", (Candidate("right", 0.95, {}, rerank_score=0.98),
                              Candidate("x", 0.10, {}, rerank_score=0.01)),
                      frozenset({"right"}), False),
    ]

    fitted = calibrate(records)
    scored = score_records(records, fitted)

    assert fitted.rerank > 0.11
    assert scored["false_answers"] == 0
    assert scored["answers"] == 1


def test_auto_fails_closed_when_reranker_is_unavailable():
    decision = decide(
        [candidate("one", 0.99), candidate("two", 0.10)],
        thresholds(score=0.80, margin=0.15, none=0.30),
        reranker_available=False,
    )
    assert decision.kind == "review"
    assert decision.reason == "reranker-unavailable"


def test_weak_top_score_returns_none():
    decision = decide(
        [candidate("weak", 0.21), candidate("weaker", 0.05)],
        thresholds(score=0.80, margin=0.15, none=0.30),
        reranker_available=True,
    )
    assert decision.kind == "none"
    assert decision.reason == "below-none-threshold"
    assert ids(decision) == []


def test_no_candidates_at_all_returns_none():
    decision = decide([], thresholds(), reranker_available=True)

    assert decision.kind == "none"
    assert decision.reason == "no-candidates"


def test_score_between_none_and_answer_thresholds_is_review():
    decision = decide(
        [candidate("mid", 0.55), candidate("low", 0.10)],
        thresholds(score=0.80, margin=0.15, none=0.30),
        reranker_available=True,
    )
    assert decision.kind == "review"
    assert decision.reason == "below-answer-threshold"


def test_a_single_clear_candidate_can_answer_without_a_runner_up():
    decision = decide([candidate("only", 0.95)], thresholds(), reranker_available=True)

    assert decision.kind == "answer"
    assert ids(decision) == ["only"]


def test_review_is_capped_and_ordered_by_score():
    cands = [candidate(f"c{i}", 0.8 - i * 0.01) for i in range(10)]

    decision = decide(cands, thresholds(), reranker_available=True, review_k=3)

    assert decision.kind == "review"
    assert ids(decision) == ["c0", "c1", "c2"]


def test_reranker_unavailable_still_returns_none_when_nothing_is_plausible():
    """Failing closed means never *answering* — it must not manufacture
    candidates for review out of a genuinely empty result."""
    decision = decide([candidate("weak", 0.01)], thresholds(), reranker_available=False)

    assert decision.kind == "none"


def test_decision_is_serializable_for_json_output():
    decision = decide([candidate("right", 0.91), candidate("other", 0.10)],
                      thresholds(), reranker_available=True)

    payload = decision.as_json()

    assert payload == {"decision": "answer", "reason": "clear-margin",
                       "results": [{"id": "right", "score": 0.91}]}
    json.dumps(payload)  # must not raise


# --------------------------------------------------------------------------
# thresholds io
# --------------------------------------------------------------------------


def test_thresholds_round_trip_through_disk(tmp_path):
    path = tmp_path / "auto_thresholds.json"
    original = Thresholds(score=0.77, margin=0.11, none=0.42)

    write_thresholds(path, original)

    assert load_thresholds(path) == original


def test_thresholds_round_trip_through_a_file(tmp_path):
    path = tmp_path / "t.json"
    original = Thresholds(score=0.63, margin=0.02, none=0.48, rerank=0.5)

    write_thresholds(path, original)

    assert load_thresholds(path) == original


def test_load_thresholds_ignores_annotation_keys(tmp_path):
    """A leading-underscore key (e.g. `_note`, documenting where the numbers
    came from) must not be treated as a required field or otherwise break
    loading — see the packaged `auto_thresholds.json`."""
    path = tmp_path / "t.json"
    path.write_text('{"_note": "x", "score": 0.6, "margin": 0.01, '
                    '"none": 0.4, "rerank": 0.5}', encoding="utf-8")
    assert load_thresholds(path).score == 0.6


def test_missing_threshold_file_falls_back_to_a_conservative_default(tmp_path):
    loaded = load_thresholds(tmp_path / "absent.json")

    # Unreachable by construction: an absent calibration must never let the
    # system answer automatically.
    assert loaded.score > 1.0


def test_missing_thresholds_file_fails_closed(tmp_path):
    loaded = load_thresholds(tmp_path / "absent.json")

    assert loaded == UNCALIBRATED


def test_malformed_json_thresholds_file_fails_closed(tmp_path, capsys):
    path = tmp_path / "auto_thresholds.json"
    path.write_text("{not json", encoding="utf-8")

    loaded = load_thresholds(path)

    assert loaded == UNCALIBRATED
    assert str(path) in capsys.readouterr().err


def test_thresholds_file_missing_a_required_key_fails_closed(tmp_path, capsys):
    path = tmp_path / "auto_thresholds.json"
    path.write_text(json.dumps({"score": 0.8, "margin": 0.1}), encoding="utf-8")  # no "none"

    loaded = load_thresholds(path)

    assert loaded == UNCALIBRATED
    assert str(path) in capsys.readouterr().err


def test_thresholds_path_that_is_a_directory_fails_closed(tmp_path, capsys):
    path = tmp_path / "auto_thresholds.json"
    path.mkdir()

    loaded = load_thresholds(path)

    assert loaded == UNCALIBRATED
    assert str(path) in capsys.readouterr().err


# --------------------------------------------------------------------------
# threshold *resolution*: one precedence rule shared by wiki-recall --auto,
# wiki-eval --auto, and wiki-gate. Before this, each of the three resolved
# the path differently — this is the ladder that keeps them agreeing.
# --------------------------------------------------------------------------


def test_an_explicit_path_wins_over_both_vault_root_and_packaged(tmp_path):
    explicit = tmp_path / "explicit.json"
    (tmp_path / "auto_thresholds.json").write_text("{}", encoding="utf-8")  # vault-root file present too

    resolved = resolve_thresholds_path(explicit, tmp_path)

    assert resolved == explicit
    assert resolved != PACKAGED_THRESHOLDS_PATH


def test_with_no_explicit_path_a_vault_root_file_wins_over_the_packaged_one(tmp_path):
    vault_file = tmp_path / "auto_thresholds.json"
    vault_file.write_text("{}", encoding="utf-8")

    resolved = resolve_thresholds_path(None, tmp_path)

    assert resolved == vault_file
    assert resolved != PACKAGED_THRESHOLDS_PATH


def test_with_neither_explicit_nor_vault_root_the_packaged_file_is_chosen_and_loads(tmp_path):
    """The test that would have caught `wiki-eval --auto`'s silent-UNCALIBRATED bug:
    an empty vault (no auto_thresholds.json of its own) must still resolve to a
    real, loadable calibration — not to a path that happens to not exist."""
    resolved = resolve_thresholds_path(None, tmp_path)

    assert resolved == PACKAGED_THRESHOLDS_PATH
    assert resolved.exists()
    assert load_thresholds(resolved) != UNCALIBRATED


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------


def record(correct, cands, expect_none=False):
    return LabeledRecord(
        query="q" + str(len(cands)) + str(correct),
        candidates=tuple(candidate(i, s) for i, s in cands),
        correct_ids=frozenset(correct),
        expect_none=expect_none,
    )


def test_calibration_admits_no_false_answers_on_its_own_records():
    records = [
        LabeledRecord("a", (candidate("right", 0.95), candidate("x", 0.40)),
                      frozenset({"right"}), False),
        LabeledRecord("b", (candidate("right2", 0.92), candidate("y", 0.30)),
                      frozenset({"right2"}), False),
        # a confident-but-WRONG top hit: no threshold pair may answer here
        LabeledRecord("c", (candidate("wrong", 0.90), candidate("z", 0.20)),
                      frozenset({"actual"}), False),
    ]

    fitted = calibrate(records)
    scored = score_records(records, fitted, reranker_available=True)

    assert scored["false_answers"] == 0


def test_calibration_prefers_more_coverage_among_safe_thresholds():
    records = [
        LabeledRecord("a", (candidate("right", 0.95), candidate("x", 0.10)),
                      frozenset({"right"}), False),
        LabeledRecord("b", (candidate("right2", 0.90), candidate("y", 0.10)),
                      frozenset({"right2"}), False),
        LabeledRecord("c", (candidate("right3", 0.85), candidate("z", 0.10)),
                      frozenset({"right3"}), False),
    ]

    fitted = calibrate(records)
    scored = score_records(records, fitted, reranker_available=True)

    assert scored["false_answers"] == 0
    assert scored["answers"] == 3


def test_a_negative_record_answered_confidently_forces_a_stricter_none_threshold():
    records = [
        LabeledRecord("neg", (candidate("anything", 0.70),), frozenset(), True),
        LabeledRecord("pos", (candidate("right", 0.95), candidate("x", 0.10)),
                      frozenset({"right"}), False),
    ]

    fitted = calibrate(records)
    scored = score_records(records, fitted, reranker_available=True)

    assert scored["false_answers"] == 0
    assert fitted.none > 0.70


def test_calibration_with_no_safe_answering_threshold_refuses_to_answer():
    records = [
        LabeledRecord("c", (candidate("wrong", 0.99), candidate("z", 0.01)),
                      frozenset({"actual"}), False),
    ]

    fitted = calibrate(records)
    scored = score_records(records, fitted, reranker_available=True)

    assert scored["answers"] == 0
    assert scored["false_answers"] == 0


def test_score_records_counts_correct_abstentions():
    thr = thresholds(score=0.80, margin=0.15, none=0.30)
    records = [
        LabeledRecord("neg", (candidate("noise", 0.10),), frozenset(), True),
        LabeledRecord("pos", (candidate("right", 0.95), candidate("x", 0.10)),
                      frozenset({"right"}), False),
    ]

    scored = score_records(records, thr, reranker_available=True)

    assert scored["nones"] == 1
    assert scored["correct_nones"] == 1
    assert scored["answers"] == 1
    assert scored["correct_answers"] == 1
    assert scored["false_answers"] == 0


def test_answering_a_negative_case_counts_as_a_false_answer():
    thr = thresholds(score=0.80, margin=0.15, none=0.30)
    records = [LabeledRecord("neg", (candidate("noise", 0.99), candidate("x", 0.01)),
                             frozenset(), True)]

    scored = score_records(records, thr, reranker_available=True)

    assert scored["false_answers"] == 1


def test_calibration_requires_records():
    with pytest.raises(ValueError, match="no records"):
        calibrate([])
