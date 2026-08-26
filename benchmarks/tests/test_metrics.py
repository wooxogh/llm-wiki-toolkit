from math import log2

import pytest

from llm_wiki_bench.metrics import (
    abstention_summary,
    aggregate,
    score_abstention,
    score_answer,
    score_citations,
    score_distractor_rejection,
    score_multi_slot_answer,
    score_retrieval,
    score_sub_claims,
)


def test_retrieval_scores_the_first_relevant_document_at_rank_two() -> None:
    scores = score_retrieval(("relevant",), ("distractor", "relevant", "other"))

    assert scores["recall@1"] == 0.0
    assert scores["recall@3"] == 1.0
    assert scores["mrr"] == 0.5
    assert scores["ndcg@3"] == pytest.approx(1 / log2(3))


def test_retrieval_distinguishes_any_evidence_recall_from_full_set_coverage() -> None:
    scores = score_retrieval(("first", "second"), ("first", "distractor"))

    assert scores["recall@1"] == 1.0
    assert scores["evidence_set_coverage"] == 0.5


def test_answer_normalizes_unicode_and_case_for_exact_match() -> None:
    scores = score_answer(("CAFÉ",), "cafe\u0301")

    assert scores == {"exact_match": 1.0, "token_f1": 1.0}


def test_citation_precision_is_undefined_without_citations() -> None:
    scores = score_citations(("evidence-1",), ())

    assert scores == {"citation_precision": None, "citation_recall": 0.0, "citation_f1": 0.0}


def test_citations_without_expected_evidence_leave_recall_and_f1_undefined() -> None:
    scores = score_citations((), ("unsupported",))

    assert scores == {"citation_precision": 0.0, "citation_recall": None, "citation_f1": None}


def test_aggregate_ignores_none_and_non_finite_metric_values() -> None:
    scores = aggregate([
        {"recall@1": 1.0, "citation_precision": None},
        {"recall@1": 0.0, "citation_precision": float("nan")},
    ])

    assert scores == {
        "n": 2,
        "citation_precision": None,
        "recall@1": 0.5,
        "counts": {"citation_precision": 0, "recall@1": 2},
    }


def test_aggregate_reports_per_key_contributing_count_for_differing_key_sets() -> None:
    # Simulates LongMemEval's per-record profile split: most rows carry
    # fine_* keys, a minority (abstention cases) do not.
    rows = [
        {"exact_match": 1.0, "fine_recall@1": 1.0},
        {"exact_match": 0.0, "fine_recall@1": 0.0},
        {"exact_match": 1.0},
    ]

    scores = aggregate(rows)

    assert scores["n"] == 3
    assert scores["exact_match"] == pytest.approx(2 / 3)
    assert scores["fine_recall@1"] == 0.5
    assert scores["counts"] == {"exact_match": 3, "fine_recall@1": 2}


def test_aggregate_reports_zero_count_for_key_with_no_contributing_rows() -> None:
    scores = aggregate([{"citation_precision": None}, {"citation_precision": float("nan")}])

    assert scores["citation_precision"] is None
    assert scores["counts"] == {"citation_precision": 0}


def test_abstention_true_positive():
    assert score_abstention(True, True) == {
        "abstention_tp": 1.0,
        "abstention_fp": 0.0,
        "abstention_fn": 0.0,
    }


def test_abstention_false_positive_when_answering_was_expected():
    assert score_abstention(False, True)["abstention_fp"] == 1.0


def test_abstention_false_negative_when_abstaining_was_expected():
    assert score_abstention(True, False)["abstention_fn"] == 1.0


def test_abstention_summary_computes_corpus_level_rates():
    rows = [
        score_abstention(True, True),
        score_abstention(True, False),
        score_abstention(False, True),
    ]
    summary = abstention_summary(rows)
    assert summary["abstention_precision"] == 0.5
    assert summary["abstention_recall"] == 0.5
    assert summary["abstention_f1"] == 0.5


def test_abstention_summary_is_undefined_without_any_abstention():
    summary = abstention_summary([score_abstention(False, False)])
    assert summary["abstention_precision"] is None
    assert summary["abstention_recall"] is None
    assert summary["abstention_f1"] is None


def test_distractor_rejection_rewards_avoiding_the_outdated_answer():
    assert score_distractor_rejection(("Saccharomyces bulderi",), "Maudiozyma bulderi") == {
        "distractor_rejection": 1.0
    }


def test_distractor_rejection_penalizes_reproducing_it():
    assert score_distractor_rejection(("Saccharomyces bulderi",), "saccharomyces  BULDERI") == {
        "distractor_rejection": 0.0
    }


def test_distractor_rejection_of_no_answer_is_a_rejection():
    assert score_distractor_rejection(("x",), None) == {"distractor_rejection": 1.0}


def test_multi_slot_answer_counts_matched_slots():
    scores = score_multi_slot_answer(
        (("January 2 2022", "Jan 2, 2022"), ("Ada Lovelace",)),
        "On Jan 2, 2022 Ada Lovelace opened it",
    )
    assert scores == {"slot_coverage": 1.0, "all_slots_matched": 1.0}


def test_multi_slot_answer_reports_partial_coverage():
    scores = score_multi_slot_answer(
        (("January 2 2022",), ("Ada Lovelace",)), "Ada Lovelace opened it"
    )
    assert scores == {"slot_coverage": 0.5, "all_slots_matched": 0.0}


def test_multi_slot_answer_handles_more_than_two_slots_fully_matched():
    # RGB's integration variant asks up to six sub-questions in one query
    # (real distribution: 2x94, 3x3, 4x1, 6x2), so nothing may assume two slots.
    scores = score_multi_slot_answer(
        (
            ("January 2 2022", "Jan 2, 2022"),
            ("Ada Lovelace",),
            ("Paris",),
            ("United Nations", "UN"),
        ),
        "On Jan 2, 2022 Ada Lovelace addressed the UN in Paris",
    )
    assert scores == {"slot_coverage": 1.0, "all_slots_matched": 1.0}


def test_multi_slot_answer_handles_more_than_two_slots_partially_matched():
    scores = score_multi_slot_answer(
        (
            ("January 2 2022",),
            ("Ada Lovelace",),
            ("Paris",),
            ("United Nations", "UN"),
        ),
        "Ada Lovelace addressed the UN in Paris",
    )
    assert scores == {"slot_coverage": 0.75, "all_slots_matched": 0.0}


def test_sub_claim_accuracy_and_macro_f1():
    scores = score_sub_claims(("true", "false", "true"), ("true", "true", "true"))
    assert scores["sub_claim_accuracy"] == pytest.approx(2 / 3)
    assert scores["sub_claim_macro_f1"] == pytest.approx(0.4)


def test_sub_claims_of_differing_length_are_undefined():
    scores = score_sub_claims(("true", "false"), ("true",))
    assert scores["sub_claim_accuracy"] is None
    assert scores["sub_claim_macro_f1"] is None
