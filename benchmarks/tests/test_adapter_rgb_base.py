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


def test_a_flat_list_answer_is_also_accepted(tmp_path):
    record = dict(RECORD, answer=["Tampa, Florida", "Tampa"])
    case = RGBBaseAdapter().load(_write(tmp_path, [record]), split="en").cases[0]
    assert case.labels["answers"] == ("Tampa, Florida", "Tampa")


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
    assert require_documents(record, "positive_wrong", path, 3) == []


def test_the_committed_fixture_matches_the_released_shape():
    fixture = Path(__file__).parent.parent / "fixtures" / "rgb_base.jsonl"
    result = RGBBaseAdapter().load(fixture, split="en")
    assert result.record_count == 2
    assert all(case.evidence_ids for case in result.cases)
