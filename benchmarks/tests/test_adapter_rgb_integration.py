import json
from pathlib import Path

import pytest

from llm_wiki_bench.adapters.rgb_integration import RGBIntegrationAdapter

RECORD = {
    "id": 0,
    "query": "When was the summit and who chaired it?",
    "answer": [["January 2 2022", "Jan 2, 2022"], ["Ada Lovelace", "A. Lovelace"]],
    "asnwer1": "January 2 2022",
    "answer2": "Ada Lovelace",
    "positive": [
        ["The summit opened on January 2 2022."],
        ["Ada Lovelace chaired the summit."],
    ],
    "negative": ["Unrelated conference news.", "Venue rumours."],
}


def _write(tmp_path, records):
    path = tmp_path / "en_int.json"
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def test_each_answer_slot_is_kept_separate(tmp_path):
    case = RGBIntegrationAdapter().load(_write(tmp_path, [RECORD]), split="en_int").cases[0]
    assert case.profile == "multi_slot_retrieval_qa"
    assert case.labels["answer_slots"] == (
        ("January 2 2022", "Jan 2, 2022"),
        ("Ada Lovelace", "A. Lovelace"),
    )


def test_three_answer_slots_are_each_kept_separate(tmp_path):
    # Real en_int.json records go beyond two slots (2x94, 3x3, 4x1, 6x2 in the
    # released file), so a fixture that only ever exercises two slots would
    # not catch a two-slot assumption in `_answer_slots`.
    record = dict(
        RECORD,
        id=1,
        answer=[["January 2 2022"], ["Ada Lovelace"], ["Paris"]],
        positive=[["The summit opened."], ["Ada Lovelace chaired it."], ["It was held in Paris."]],
    )
    case = RGBIntegrationAdapter().load(_write(tmp_path, [record]), split="en_int").cases[0]
    assert case.labels["answer_slots"] == (
        ("January 2 2022",),
        ("Ada Lovelace",),
        ("Paris",),
    )


def test_grouped_positive_documents_flatten_in_order(tmp_path):
    case = RGBIntegrationAdapter().load(_write(tmp_path, [RECORD]), split="en_int").cases[0]
    assert case.context[:2] == (
        "The summit opened on January 2 2022.",
        "Ada Lovelace chaired the summit.",
    )
    assert case.evidence_ids == ("0:positive:0", "0:positive:1")


def test_the_upstream_asnwer1_typo_is_preserved_as_metadata(tmp_path):
    case = RGBIntegrationAdapter().load(_write(tmp_path, [RECORD]), split="en_int").cases[0]
    assert case.metadata["source_fields"]["asnwer1"] == "January 2 2022"
    assert case.metadata["source_fields"]["answer2"] == "Ada Lovelace"


def test_a_flat_positive_list_is_also_accepted(tmp_path):
    record = dict(RECORD, positive=["One.", "Two."])
    case = RGBIntegrationAdapter().load(_write(tmp_path, [record]), split="en_int").cases[0]
    assert case.context[:2] == ("One.", "Two.")


def test_a_single_slot_answer_is_rejected(tmp_path):
    record = dict(RECORD, answer=[["only one slot"]])
    with pytest.raises(ValueError, match="record 1: answer must declare at least two slots"):
        RGBIntegrationAdapter().load(_write(tmp_path, [record]), split="en_int")


def test_the_committed_fixture_matches_the_released_shape():
    fixture = Path(__file__).parent.parent / "fixtures" / "rgb_integration.jsonl"
    result = RGBIntegrationAdapter().load(fixture, split="en_int")
    assert result.record_count == 1
    assert len(result.cases[0].labels["answer_slots"]) == 2
