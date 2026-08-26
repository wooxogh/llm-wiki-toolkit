import json
from pathlib import Path

import pytest

from llm_wiki_bench.adapters.rgb_base import RGBBaseAdapter, require_documents

RECORD = {
    "id": 0,
    "query": "Super Bowl 2021 location",
    "answer": [["Tampa, Florida", "Tampa", "Raymond James Stadium"]],
    "positive": ["The game was played in Tampa, Florida.", "Held at Raymond James Stadium."],
    "negative": ["Super Bowl LVIII will be in Las Vegas.", "Ticket packages now available."],
}


def _write(tmp_path, records, name="en.json"):
    path = tmp_path / name
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def test_normalizes_the_released_record(tmp_path):
    case = RGBBaseAdapter().load(_write(tmp_path, [RECORD]), split="en").cases[0]
    assert case.id == "0"
    assert case.dataset == "rgb_base"
    assert case.profile == "retrieval_qa"
    assert case.prompt == "Super Bowl 2021 location"


def test_answer_slot_aliases_flatten_into_accepted_answers(tmp_path):
    case = RGBBaseAdapter().load(_write(tmp_path, [RECORD]), split="en").cases[0]
    assert case.labels["answers"] == ("Tampa, Florida", "Tampa", "Raymond James Stadium")


def test_a_bare_string_answer_is_also_accepted(tmp_path):
    record = dict(RECORD, answer="Tampa, Florida")
    case = RGBBaseAdapter().load(_write(tmp_path, [record]), split="en").cases[0]
    assert case.labels["answers"] == ("Tampa, Florida",)


def test_a_single_element_list_answer_is_also_accepted(tmp_path):
    """A one-element outer list is one slot regardless of nesting depth."""
    record = dict(RECORD, answer=["Tampa, Florida"])
    case = RGBBaseAdapter().load(_write(tmp_path, [record]), split="en").cases[0]
    assert case.profile == "retrieval_qa"
    assert case.labels["answers"] == ("Tampa, Florida",)


def test_document_ids_are_synthesized_and_context_matches_their_order(tmp_path):
    case = RGBBaseAdapter().load(_write(tmp_path, [RECORD]), split="en").cases[0]
    assert case.evidence_ids == ("0:positive:0", "0:positive:1")
    assert case.metadata["candidate_ids"] == (
        "0:positive:0",
        "0:positive:1",
        "0:negative:0",
        "0:negative:1",
    )
    assert case.context == (
        "The game was played in Tampa, Florida.",
        "Held at Raymond James Stadium.",
        "Super Bowl LVIII will be in Las Vegas.",
        "Ticket packages now available.",
    )


def test_a_json_extension_is_read_as_json_lines(tmp_path):
    result = RGBBaseAdapter().load(_write(tmp_path, [RECORD, dict(RECORD, id=1)]), split="en")
    assert result.record_count == 2


def test_empty_positive_pool_is_rejected(tmp_path):
    record = dict(RECORD, positive=[])
    with pytest.raises(ValueError, match="record 1: positive must be a non-empty list"):
        RGBBaseAdapter().load(_write(tmp_path, [record]), split="en")


def test_require_documents_honors_the_required_flag_for_any_key(tmp_path):
    path = tmp_path / "en.json"
    record = {"positive_wrong": []}
    with pytest.raises(ValueError, match="record 3: positive_wrong must be a non-empty list"):
        require_documents(record, "positive_wrong", path, 3, required=True)
    assert require_documents(record, "positive_wrong", path, 3, required=False) == []


def test_two_distinct_answer_slots_use_the_multi_slot_profile_and_do_not_flatten(tmp_path):
    """[["A"],["B"]] is two genuinely distinct required answers, not one
    two-alias answer. Measured against the real en_refine.json (300 records):
    outer answer-list lengths are {1: 288, 2: 9, 3: 1, 4: 1, 6: 1} -- 12 of
    300 records are multi-slot and must not be silently flattened into one
    accepted-answer tuple, which would let a prediction of just "A" score
    exact_match 1.0 for answering only half the question."""
    record = dict(RECORD, answer=[["A"], ["B"]])
    case = RGBBaseAdapter().load(_write(tmp_path, [record]), split="en").cases[0]
    assert case.profile == "multi_slot_retrieval_qa"
    assert case.labels["answer_slots"] == (("A",), ("B",))
    assert "answers" not in case.labels


def test_two_answer_slots_with_bare_string_entries_still_use_the_multi_slot_profile(tmp_path):
    """The outer list length IS the slot count regardless of the inner
    entries' type -- an inner string is a slot with a single alias, not a
    second alias for the same slot. Measured directly against the real
    en_refine.json: 11 of the 12 real multi-slot records have STRING inner
    entries, not nested lists (e.g. id=39, "Who stars in The Lost City?",
    answer=['Sandra Bullock', 'Channing Tatum'] -- two stars, not one
    two-alias answer). A routing rule keyed on "are the inner entries lists"
    misses exactly this shape and still flattens 11 of 12 real multi-slot
    records; this is the case that must not silently over-score a
    single-star prediction as exact_match 1.0."""
    record = dict(RECORD, answer=["Sandra Bullock", "Channing Tatum"])
    case = RGBBaseAdapter().load(_write(tmp_path, [record]), split="en").cases[0]
    assert case.profile == "multi_slot_retrieval_qa"
    assert case.labels["answer_slots"] == (("Sandra Bullock",), ("Channing Tatum",))
    assert "answers" not in case.labels


def test_a_single_slot_with_multiple_aliases_still_uses_the_retrieval_profile(tmp_path):
    """A one-element outer list (288 of 300 real records) is one slot with
    several accepted aliases, not multi-slot, and must keep flattening."""
    record = dict(RECORD, answer=[["Tampa, Florida", "Tampa"]])
    case = RGBBaseAdapter().load(_write(tmp_path, [record]), split="en").cases[0]
    assert case.profile == "retrieval_qa"
    assert case.labels["answers"] == ("Tampa, Florida", "Tampa")
    assert "answer_slots" not in case.labels


def test_three_answer_slots_are_each_kept_separate(tmp_path):
    record = dict(RECORD, answer=[["A"], ["B"], ["C"]])
    case = RGBBaseAdapter().load(_write(tmp_path, [record]), split="en").cases[0]
    assert case.profile == "multi_slot_retrieval_qa"
    assert case.labels["answer_slots"] == (("A",), ("B",), ("C",))


def test_four_answer_slots_with_bare_string_entries_are_each_kept_separate(tmp_path):
    """Mirrors the real id=15 record (four recipients of a named prize),
    all four slots given as bare strings, not nested lists."""
    record = dict(
        RECORD,
        answer=["Lawrence Williams", "Ralph Long Jr.", "Ford Greene", "Ronald Yancey"],
    )
    case = RGBBaseAdapter().load(_write(tmp_path, [record]), split="en").cases[0]
    assert case.profile == "multi_slot_retrieval_qa"
    assert case.labels["answer_slots"] == (
        ("Lawrence Williams",),
        ("Ralph Long Jr.",),
        ("Ford Greene",),
        ("Ronald Yancey",),
    )


def test_the_committed_fixture_matches_the_released_shape():
    fixture = Path(__file__).parent.parent / "fixtures" / "rgb_base.jsonl"
    result = RGBBaseAdapter().load(fixture, split="en")
    assert result.record_count == 2
    assert all(case.evidence_ids for case in result.cases)
