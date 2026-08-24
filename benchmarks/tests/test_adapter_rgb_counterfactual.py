import json
from pathlib import Path

import pytest

from llm_wiki_bench.adapters.rgb_counterfactual import RGBCounterfactualAdapter

RECORD = {
    "id": 0,
    "query": "Super Bowl 2021 location",
    "answer": "Tampa, Florida",
    "fakeanswer": "Glendale, Arizona",
    "positive": ["The game was played in Tampa, Florida."],
    "positive_wrong": ["The game was played in Glendale, Arizona."],
    "negative": ["Ticket packages now available."],
}


def _write(tmp_path, records):
    path = tmp_path / "en_fact.json"
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def test_a_bare_string_answer_is_normalized(tmp_path):
    case = RGBCounterfactualAdapter().load(_write(tmp_path, [RECORD]), split="en_fact").cases[0]
    assert case.profile == "counterfactual_qa"
    assert case.labels["answers"] == ("Tampa, Florida",)


def test_a_list_shaped_answer_is_normalized(tmp_path):
    record = dict(RECORD, id=1, answer=["Tampa", "Tampa, Florida"])
    case = RGBCounterfactualAdapter().load(_write(tmp_path, [record]), split="en_fact").cases[0]
    assert case.labels["answers"] == ("Tampa", "Tampa, Florida")


def test_the_fake_answer_becomes_the_distractor(tmp_path):
    case = RGBCounterfactualAdapter().load(_write(tmp_path, [RECORD]), split="en_fact").cases[0]
    assert case.labels["distractor_answers"] == ("Glendale, Arizona",)


def test_positive_wrong_documents_are_candidates_but_not_expected_evidence(tmp_path):
    case = RGBCounterfactualAdapter().load(_write(tmp_path, [RECORD]), split="en_fact").cases[0]
    assert case.evidence_ids == ("0:positive:0",)
    assert case.metadata["candidate_ids"] == (
        "0:positive:0",
        "0:negative:0",
        "0:positive_wrong:0",
    )
    assert case.context == (
        "The game was played in Tampa, Florida.",
        "Ticket packages now available.",
        "The game was played in Glendale, Arizona.",
    )


def test_a_missing_fakeanswer_is_rejected(tmp_path):
    record = {key: value for key, value in RECORD.items() if key != "fakeanswer"}
    with pytest.raises(ValueError, match="record 1: missing fakeanswer"):
        RGBCounterfactualAdapter().load(_write(tmp_path, [record]), split="en_fact")


def test_the_committed_fixture_matches_the_released_shape():
    fixture = Path(__file__).parent.parent / "fixtures" / "rgb_counterfactual.jsonl"
    result = RGBCounterfactualAdapter().load(fixture, split="en_fact")
    assert result.record_count == 1
    assert result.cases[0].labels["distractor_answers"]
