"""MERGE-candidate detection must measure redundancy, not topical similarity.

The vault's writing voice is uniform, so embedding similarity alone rates
unrelated pages as near-duplicates. sentence_overlap is the second signal that
keeps MERGE candidates honest.

Measured 2026-07-31: all 19 cosine>=0.85 pairs shared at most 2 identical
sentences (out of 48-116), and 12 of them already
linked each other as complements. Cosine alone had **0% precision** there —
acting on it would have collapsed several distinct artifacts (verdicts, a
postmortem, a design doc, a runbook) that were already correctly organised
hub-and-spoke.
"""
from __future__ import annotations

from llm_wiki.hygiene import compact


def test_sentence_overlap_ignores_short_lines():
    a = "짧다.\n" + "이 문장은 비교 대상에 들어갈 만큼 충분히 길게 작성된 한국어 문장이며 마흔 자를 넘긴다.\n"
    b = "다르다.\n" + "이 문장은 비교 대상에 들어갈 만큼 충분히 길게 작성된 한국어 문장이며 마흔 자를 넘긴다.\n"

    assert compact.sentence_overlap(a, b) == 1.0


def test_disjoint_documents_have_zero_overlap():
    a = "완전히 다른 내용을 담은 충분히 긴 첫 번째 문서의 유일한 문장이다 정말로.\n"
    b = "이쪽은 전혀 관계없는 두 번째 문서의 문장이며 길이는 비슷하게 맞춰두었다.\n"

    assert compact.sentence_overlap(a, b) == 0.0


def test_overlap_is_normalised_by_the_shorter_document():
    shared = "두 문서가 공유하는 충분히 긴 문장 하나이며 이것만 겹치도록 구성했고 마흔 자를 넘긴다.\n"
    a = shared
    b = shared + "여기에만 있는 또 다른 충분히 긴 문장을 하나 더 붙여서 전체 길이를 늘려 둔다 정말로.\n"

    # 1 shared / min(1, 2) == 1.0 — a is fully contained in b
    assert compact.sentence_overlap(a, b) == 1.0


def test_empty_documents_do_not_divide_by_zero():
    assert compact.sentence_overlap("", "") == 0.0
    assert compact.sentence_overlap("", "긴 문장 하나만 있는 문서이고 비교 대상은 비어 있다.\n") == 0.0


def test_merge_candidate_requires_both_cosine_and_real_overlap():
    """A high-cosine pair that shares no wording is a *related* pair, not a
    duplicate. Flagging it as MERGE sends a human to delete real knowledge."""
    topical_only = "하이브리드 랭킹에서 희소 점수와 밀집 점수를 융합하는 이유를 정리한 문서이다.\n"
    other = "재랭커 모델을 내려받고 임계값을 다시 맞추는 배포 절차를 적어 둔 런북이다.\n"

    assert compact.is_merge_candidate(0.95, topical_only, other) is False


def test_a_genuine_duplicate_is_still_flagged():
    dup = ("두 문서가 통째로 공유하는 충분히 긴 첫 문장이다 그러므로 중복으로 잡혀야 한다.\n"
           "역시 공유되는 두 번째 문장이며 길이 조건을 만족하도록 충분히 길게 적는다.\n")

    assert compact.is_merge_candidate(0.90, dup, dup) is True


def test_low_cosine_is_rejected_regardless_of_overlap():
    dup = "동일한 문장을 양쪽에 두었지만 코사인이 낮으면 후보가 아니다 충분히 길게.\n"

    assert compact.is_merge_candidate(0.10, dup, dup) is False
