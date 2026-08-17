"""Gold validation, coverage accounting, and hand-derived ranking metrics."""
from __future__ import annotations

import json

import pytest

from llm_wiki import config
from llm_wiki.evaluation import eval_schema
from llm_wiki.evaluation.eval_schema import (
    GoldCase,
    aggregate,
    coverage_report,
    coverage_shortfalls,
    evaluate_case,
    leaked_ids,
    load_gold,
    slice_by,
    validate_gold,
)


def write_json(tmp_path, data):
    path = tmp_path / "gold.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def case(**over):
    base = dict(q="질문", expect=("canonical",), expect_none=False, split="test",
                category="direct", difficulty="easy", layer="domain",
                domain="research", projects=())
    base.update(over)
    return GoldCase(**base)


def entry(_id="canonical", **over):
    e = {"id": _id, "layer": "domain", "domain": "research", "status": "active",
         "updated": "2026-07-31", "projects": ["project-a"]}
    e.update(over)
    return e


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------


def test_none_case_rejects_expected_ids(tmp_path):
    path = write_json(tmp_path, [{
        "q": "동네 도서관 휴관일",
        "expect": ["vault-schema-contract"],
        "expect_none": True,
        "split": "test",
        "category": "negative",
    }])
    with pytest.raises(ValueError, match="expect_none cannot include expect"):
        load_gold(path)


def test_case_without_expect_or_expect_none_is_rejected(tmp_path):
    path = write_json(tmp_path, [{"q": "무엇", "expect": [], "split": "test"}])
    with pytest.raises(ValueError, match="must have 'expect' or expect_none"):
        load_gold(path)


def test_duplicate_queries_are_rejected(tmp_path):
    path = write_json(tmp_path, [
        {"q": "같은 질문", "expect": ["a"], "split": "test"},
        {"q": "같은 질문", "expect": ["b"], "split": "test"},
    ])
    with pytest.raises(ValueError, match="duplicate query"):
        load_gold(path)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("split", "holdout", "split"),
        ("category", "tricky", "category"),
        ("difficulty", "medium", "difficulty"),
        ("layer", "concept", "layer"),
    ],
)
def test_invalid_enum_values_are_rejected(tmp_path, field, value, match):
    raw = {"q": "질문", "expect": ["a"], "split": "test"}
    raw[field] = value
    with pytest.raises(ValueError, match=match):
        load_gold(write_json(tmp_path, raw and [raw]))


def test_empty_query_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="'q' must be a non-empty string"):
        load_gold(write_json(tmp_path, [{"q": "  ", "expect": ["a"]}]))


def test_v1_gold_files_load_by_adapting_absent_v2_fields(tmp_path):
    path = write_json(tmp_path, [{"q": "hybrid ranking", "expect": ["hybrid-ranking-tradeoffs"]}])

    cases = load_gold(path)

    assert cases[0].split == "test"
    assert cases[0].category == "direct"
    assert cases[0].difficulty == "easy"
    assert cases[0].expect == ("hybrid-ranking-tradeoffs",)


def test_negative_case_defaults_to_the_negative_category(tmp_path):
    path = write_json(tmp_path, [{"q": "동네 도서관 휴관일", "expect_none": True}])

    assert load_gold(path)[0].category == "negative"


def test_round_trip_through_as_json_is_stable(tmp_path):
    original = load_gold(write_json(tmp_path, [case().as_json()]))

    assert load_gold(write_json(tmp_path, [c.as_json() for c in original])) == original


# --------------------------------------------------------------------------
# semantic validation against the index
# --------------------------------------------------------------------------


def test_unknown_expected_id_is_an_error():
    errors = validate_gold([case(expect=("ghost",))], [entry()])

    assert any("does not exist in index.yaml" in e for e in errors)


def test_superseded_expected_id_is_an_error():
    errors = validate_gold([case()], [entry(status="superseded")])

    assert any("superseded" in e for e in errors)


def test_mismatched_coverage_labels_are_errors():
    errors = validate_gold([case(layer="pattern", domain="tooling")], [entry()])

    assert any("layer 'pattern'" in e for e in errors)
    assert any("domain 'tooling'" in e for e in errors)


def test_missing_coverage_label_is_an_error():
    errors = validate_gold([case(layer="")], [entry()])

    assert any("missing coverage label 'layer'" in e for e in errors)


def test_domain_layer_requires_a_domain_label():
    errors = validate_gold([case(domain="")], [entry()])

    assert any("requires a 'domain' coverage label" in e for e in errors)


def test_negative_category_and_expect_none_must_agree():
    mismatched = GoldCase(q="q", expect=(), expect_none=True, category="indirect")
    assert any("requires category 'negative'" in e for e in validate_gold([mismatched], []))

    wrong_way = GoldCase(q="q2", expect=("canonical",), expect_none=False,
                         category="negative", layer="domain", domain="research")
    assert any("requires expect_none" in e for e in validate_gold([wrong_way], [entry()]))


def test_a_well_formed_corpus_validates_clean():
    assert validate_gold([case()], [entry()]) == []


def test_leaked_ids_flags_conceptual_queries_naming_their_answer():
    leaky = case(q="canonical 페이지 알려줘", category="indirect")
    clean = case(q="임베딩 저장소가 오래됐는지 어떻게 아나", category="indirect")

    assert leaked_ids([leaky])
    assert leaked_ids([clean]) == []


def test_direct_queries_may_name_the_page():
    assert leaked_ids([case(q="canonical", category="direct")]) == []


# --------------------------------------------------------------------------
# metrics (hand-derived)
# --------------------------------------------------------------------------


def test_metrics_for_second_rank_relevant_result():
    result = evaluate_case(case(q="간접 질문", category="indirect"),
                           ["distractor", "canonical", "other"])

    assert result.hit_at_1 == 0
    assert result.hit_at_3 == 1
    assert result.reciprocal_rank == 0.5
    assert result.ndcg_at_3 == pytest.approx(0.6309297536)
    assert result.rank == 2


def test_metrics_for_first_rank_relevant_result():
    result = evaluate_case(case(), ["canonical", "other"])

    assert (result.hit_at_1, result.hit_at_3, result.hit_at_8) == (1, 1, 1)
    assert result.reciprocal_rank == 1.0
    assert result.ndcg_at_3 == 1.0


def test_ndcg_at_3_is_zero_beyond_the_cutoff_but_hit_at_8_still_counts():
    result = evaluate_case(case(), ["a", "b", "c", "canonical"])

    assert result.hit_at_3 == 0
    assert result.hit_at_8 == 1
    assert result.ndcg_at_3 == 0.0
    assert result.reciprocal_rank == 0.25


def test_complete_miss_scores_zero_everywhere():
    result = evaluate_case(case(), ["a", "b"])

    assert (result.hit_at_1, result.hit_at_8, result.reciprocal_rank, result.rank) == (0, 0, 0.0, None)


def test_any_of_several_expected_ids_counts_as_a_hit():
    result = evaluate_case(case(expect=("one", "two")), ["x", "two"])

    assert result.hit_at_3 == 1
    assert result.rank == 2


def test_negative_case_is_correct_only_when_nothing_is_returned():
    negative = GoldCase(q="동네 도서관 휴관일", expect=(), expect_none=True, category="negative")

    assert evaluate_case(negative, []).hit_at_1 == 1
    assert evaluate_case(negative, ["anything"]).hit_at_1 == 0


def test_aggregate_averages_over_cases():
    results = [
        (case(), evaluate_case(case(), ["canonical"])),
        (case(q="b"), evaluate_case(case(q="b"), ["x", "y", "canonical"])),
    ]

    agg = aggregate(results)

    assert agg["n"] == 2
    assert agg["hit@1"] == 0.5
    assert agg["hit@3"] == 1.0
    assert agg["mrr"] == pytest.approx((1.0 + 1 / 3) / 2)


def test_aggregate_of_nothing_is_zero_not_a_crash():
    assert aggregate([])["n"] == 0


def test_slice_by_partitions_results():
    easy, hard = case(q="e", difficulty="easy"), case(q="h", difficulty="hard")
    results = [(easy, evaluate_case(easy, ["canonical"])),
               (hard, evaluate_case(hard, ["x"]))]

    sliced = slice_by(results, lambda c: c.difficulty)

    assert sliced["easy"]["hit@1"] == 1.0
    assert sliced["hard"]["hit@1"] == 0.0


# --------------------------------------------------------------------------
# coverage
# --------------------------------------------------------------------------


def test_coverage_report_counts_every_axis_with_literal_values():
    cases = [
        case(q="a", layer="domain", domain="research", difficulty="easy",
             category="direct", split="test", projects=("project-a",)),
        case(q="b", layer="domain", domain="tooling", difficulty="hard",
             category="indirect", split="calibration", expect=("crawl",),
             projects=("project-b",)),
        GoldCase(q="c", expect=(), expect_none=True, category="negative", split="test"),
    ]
    index = [entry("canonical", updated="2026-07-31"),
             entry("crawl", domain="tooling", updated="2026-07-01"),
             entry("unused", updated="2026-06-01")]

    report = coverage_report(cases, index)

    assert report["total"] == 3
    assert report["by_layer"] == {"-": 1, "domain": 2}
    assert report["by_domain"] == {"-": 1, "research": 1, "tooling": 1}
    assert report["by_category"] == {"direct": 1, "indirect": 1, "negative": 1}
    assert report["by_difficulty"] == {"easy": 2, "hard": 1}
    assert report["by_split"] == {"calibration": 1, "test": 2}
    assert report["by_project"] == {"project-a": 1, "project-b": 1}
    assert report["pages_referenced"] == 2
    assert report["pages_total"] == 3


def test_recent_window_is_measured_from_the_newest_vault_page():
    cases = [case(q="fresh", expect=("new",)), case(q="old", expect=("ancient",))]
    index = [entry("new", updated="2026-07-31"), entry("ancient", updated="2026-01-01")]

    report = coverage_report(cases, index, recent_days=7)

    assert report["recent_window_start"] == "2026-07-25"
    assert report["recent_cases"] == 1


def test_coverage_shortfalls_names_every_unmet_minimum():
    report = coverage_report([case()], [entry()])

    shortfalls = coverage_shortfalls(report)

    assert any("total 1 < 150" in s for s in shortfalls)
    assert any("layer/domain 1 < 100" in s for s in shortfalls)
    assert any("category/negative 0 < 8" in s for s in shortfalls)


def test_a_corpus_meeting_every_minimum_has_no_shortfalls():
    report = {
        "total": 150, "recent_cases": 30,
        "by_layer": {"domain": 100, "pattern": 25, "entity": 5, "raw": 2},
        "by_domain": {},
        "by_category": {"ambiguous": 10, "negative": 8},
    }

    assert coverage_shortfalls(report) == []


def test_coverage_shortfalls_uses_the_vaults_minimums(tmp_path):
    (tmp_path / "wiki.toml").write_text(
        "[eval.minimums]\ntotal = 2\nrecent_cases = 0\n"
        "layer = { domain = 1 }\ncategory = {}\n", encoding="utf-8")
    report = {"total": 2, "recent_cases": 0,
              "by_layer": {"domain": 1}, "by_domain": {}, "by_category": {}}
    mins = config.load(tmp_path).minimums
    assert eval_schema.coverage_shortfalls(report, mins) == []


def test_default_minimums_reject_a_tiny_corpus():
    report = {"total": 2, "recent_cases": 0,
              "by_layer": {"domain": 1}, "by_domain": {}, "by_category": {}}
    assert eval_schema.coverage_shortfalls(report) != []
