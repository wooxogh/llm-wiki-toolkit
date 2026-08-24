from math import log2

import pytest

from llm_wiki_bench.metrics import (
    aggregate,
    score_answer,
    score_citations,
    score_retrieval,
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

    assert scores == {"n": 2, "citation_precision": None, "recall@1": 0.5}
