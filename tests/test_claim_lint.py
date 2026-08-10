"""The vault's core rule — *measured facts only* — has had zero enforcement.

Every other invariant here is checked by a gate (schema, index drift, embedding
staleness, community synthesis, retrieval quality). The one rule the vault is
actually *about* was left to human and LLM discipline. This lints for the shape
of an unmeasured claim: hedging language with no measurement anywhere near it.

Deliberately a WARNING, never an error. Hedging is legitimate when a page is
explicitly recording uncertainty ("the retry ceiling under real load is
**unverified**"), and a gate that blocks on prose would get worked around
rather than obeyed.

The hedge vocabulary is language-specific, so most fixtures below pin an
explicit pack (`KO` or `EN`) rather than relying on whatever `[lint] packs`
this checkout happens to default to. Tests that only exercise the
language-neutral evidence anchors (dates, ratios, PR refs, code paths) or the
plain absence of any hedge call the functions with no `patterns` argument.
"""
from __future__ import annotations

import pytest

from llm_wiki.hygiene import claim_lint
from conftest import page, write_page

KO = claim_lint.load_packs(("ko",))
EN = claim_lint.load_packs(("en",))


# --------------------------------------------------------------------------
# pack loading
# --------------------------------------------------------------------------


def test_english_pack_flags_a_hedge_with_no_measurement():
    pats = claim_lint.load_packs(("en",))
    assert claim_lint.is_hedged("This probably fixes the timeout.", pats)
    assert not claim_lint.has_evidence("This probably fixes the timeout.", pats)


def test_english_pack_accepts_a_hedge_that_carries_evidence():
    pats = claim_lint.load_packs(("en",))
    assert claim_lint.has_evidence("Likely the same cause: 12/40 runs failed.", pats)


def test_korean_pack_still_distinguishes_seeming_from_sight():
    pats = claim_lint.load_packs(("ko",))
    assert claim_lint.is_hedged("괜찮아 보인다", pats)
    assert not claim_lint.is_hedged("지표가 보인다", pats)


def test_packs_compose():
    pats = claim_lint.load_packs(("en", "ko"))
    assert claim_lint.is_hedged("probably", pats)
    assert claim_lint.is_hedged("아마", pats)


def test_unknown_pack_is_an_error():
    with pytest.raises(claim_lint.UnknownPack, match="fr"):
        claim_lint.load_packs(("fr",))


def test_lint_vault_selects_packs_from_the_target_vaults_own_config(vault):
    """lint_vault(vault) must read `vault`'s own wiki.toml — content_paths(vault)
    and relative(path, vault) already do this correctly; the packs used to
    check each page must not come from the calling process's ambient config
    instead. A Korean-only vault linted under the process default (`en`)
    would silently find nothing, which is exactly what UnknownPack elsewhere
    exists to prevent for a misspelled name."""
    (vault / "wiki.toml").write_text('[lint]\npacks = ["ko"]\n', encoding="utf-8")
    write_page(vault, "domain/research/one.md",
               page(body="청크 크기를 1400까지 올려도 괜찮아 보인다.\n", id="one"))

    found = claim_lint.lint_vault(vault)

    assert "one" in found


def test_load_packs_with_no_argument_reads_the_vault_config(tmp_path, monkeypatch):
    """Callers that don't pin a pack get whatever the *effective* vault's
    wiki.toml declares. Pinned to a synthetic vault via WIKI_VAULT rather than
    relying on this checkout's ambient config (which has no wiki.toml today,
    so defaults to `("en",)`) — the point is the config lookup, not today's
    default, and this must not break if a wiki.toml is ever added at the repo
    root."""
    (tmp_path / "wiki.toml").write_text('[lint]\npacks = ["ko"]\n', encoding="utf-8")
    monkeypatch.setenv("WIKI_VAULT", str(tmp_path))

    pats = claim_lint.load_packs()

    assert claim_lint.is_hedged("아마", pats)
    assert not claim_lint.is_hedged("probably", pats)


# --------------------------------------------------------------------------
# the two signals
# --------------------------------------------------------------------------


def test_hedging_words_are_detected():
    assert claim_lint.is_hedged("청크 크기를 1400까지 올려도 괜찮아 보인다", KO) is True
    assert claim_lint.is_hedged("이 값이 맞을 것 같다", KO) is True
    assert claim_lint.is_hedged("아마 토크나이저 설정 문제일 듯", KO) is True


def test_visible_is_not_seems():
    """보이다 is a homonym. "~로 보인다" hedges; "X 이 보인다" means X is visible.
    Matching the bare verb flagged real assertions."""
    assert claim_lint.is_hedged("경합으로만 보인다", KO) is True
    assert claim_lint.is_hedged("색인이 갱신되어 값이 보인다", KO) is False
    assert claim_lint.is_hedged("실패해도 로그로도 안 보인다", KO) is False


def test_a_flat_assertion_is_not_hedged():
    assert claim_lint.is_hedged("인덱스 재생성 후 검증 오류 3건 발생") is False


def test_explicitly_labelled_uncertainty_is_not_hedging():
    """The vault *wants* pages to record what is unverified — that is a measured
    statement about the absence of measurement, not a guess."""
    assert claim_lint.is_hedged("재랭커 임계값의 일반화 **미검증**(단일 장비 기준)", KO) is False
    assert claim_lint.is_hedged("사유는 미실측", KO) is False


def test_uncertainty_marker_overrides_a_genuine_hedge_word_too():
    """The two fixtures above happen to contain no hedge word at all, so they
    pass whether or not the deliberate_uncertainty override actually fires.
    This one co-occurs a real hedge ("아마") with an uncertainty marker
    ("미검증") to exercise the override branch itself."""
    assert claim_lint.is_hedged("정정 사유는 미검증이다 아마", KO) is False


def test_evidence_is_a_number_date_pr_or_path():
    assert claim_lint.has_evidence("실측 29/31 성공") is True
    assert claim_lint.has_evidence("2026-03-04 확인") is True
    assert claim_lint.has_evidence("PR #527 으로 수정") is True
    assert claim_lint.has_evidence("`llm_wiki/retrieval/_retrieve.py` 참조") is True


def test_prose_without_any_anchor_has_no_evidence():
    assert claim_lint.has_evidence("구조상 괜찮아 보인다") is False


def test_a_bare_proposed_number_is_not_evidence():
    """"청크 크기를 1400 까지 올려도" — the 1400 is the claim being made, not a
    measurement supporting it. Counting any digit made the lint useless."""
    assert claim_lint.has_evidence("청크 크기를 700 에서 1400 까지 올려도") is False


# --------------------------------------------------------------------------
# the lint
# --------------------------------------------------------------------------


def test_a_hedged_line_with_no_evidence_is_flagged():
    body = "이 파이프라인은 구조상 청크 크기를 1400까지 올려도 괜찮아 보인다.\n"

    found = claim_lint.find_unmeasured_claims(body, patterns=KO)

    assert len(found) == 1
    assert "괜찮아 보인다" in found[0].line


def test_a_hedged_line_with_a_measurement_is_not_flagged():
    body = "청크 700 에서 임베딩 3회가 실패해서 400 으로 낮춘 것으로 보인다(PR #527).\n"

    assert claim_lint.find_unmeasured_claims(body, patterns=KO) == []


def test_evidence_on_an_adjacent_line_counts():
    """Measurements are usually on the next line in this vault's house style."""
    body = ("이 단계가 병목으로 보인다.\n"
            "실측: 인덱스 재생성 3.7s → 0.41s, 처리량 40 → 260건/분.\n")

    assert claim_lint.find_unmeasured_claims(body, patterns=KO) == []


def test_struck_through_text_is_ignored():
    """A retracted claim is *supposed* to read like a guess — that is the record
    of what was believed, not a live assertion."""
    body = "~~구조상 괜찮아 보인다~~ → 틀렸다(2026-03-04 실측).\n"

    assert claim_lint.find_unmeasured_claims(body, patterns=KO) == []


def test_code_blocks_are_ignored():
    body = "```\n# 아마 이 값이 맞을 듯\n```\n"

    assert claim_lint.find_unmeasured_claims(body, patterns=KO) == []


def test_quoted_user_speech_is_ignored():
    """Pages quote what someone said; that is data, not the page's own claim."""
    body = '> 사용자: "이거 재랭커 켠 상태였나?? 아마 설정이 반영 안 된 듯"\n'

    assert claim_lint.find_unmeasured_claims(body, patterns=KO) == []


def test_line_numbers_are_reported_for_navigation():
    body = "첫 줄은 멀쩡하다.\n\n구조상 문제 없어 보인다.\n"

    found = claim_lint.find_unmeasured_claims(body, patterns=KO)

    assert found[0].lineno == 3


def test_multiple_claims_are_all_reported():
    body = ("아마 이쪽이 원인일 것이다.\n"
            "\n"
            "\n"
            "저쪽도 비슷하게 동작할 듯하다.\n")

    assert [c.lineno for c in claim_lint.find_unmeasured_claims(body, patterns=KO)] == [1, 4]


# --------------------------------------------------------------------------
# English fixtures for the same lint behaviour
# --------------------------------------------------------------------------


def test_an_english_hedged_line_with_no_evidence_is_flagged():
    body = "This probably fixes the timeout under load.\n"

    found = claim_lint.find_unmeasured_claims(body, patterns=EN)

    assert len(found) == 1
    assert "probably" in found[0].line


def test_an_english_hedge_with_adjacent_evidence_is_not_flagged():
    body = ("This seems to be the bottleneck.\n"
            "measured: 3.7s -> 0.41s, 29/31 runs improved.\n")

    assert claim_lint.find_unmeasured_claims(body, patterns=EN) == []
