import json

import pytest

from llm_wiki_bench.adapters.vitaminc import VitaminCAdapter

RECORD = {
    "unique_id": "5ed4de07c9e77c000848a180_1",
    "case_id": "5ed4de07c9e77c000848a180",
    "wiki_revision_id": "927477259",
    "label": "NOT ENOUGH INFO",
    "claim": "Westlife made under 23.5 million sales in the UK .",
    "evidence": "According to the British Phonographic Industry , Westlife ...",
    "page": "Westlife",
    "revision_type": "real",
    "FEVER_id": "",
    "big_bench_canary": "26b5c67b",
}


def _write(tmp_path, records):
    path = tmp_path / "test.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "released,expected",
    [
        ("SUPPORTS", "entailment"),
        ("REFUTES", "contradiction"),
        ("NOT ENOUGH INFO", "neutral"),
    ],
)
def test_released_label_spellings_map(tmp_path, released, expected):
    record = dict(RECORD, label=released)
    case = VitaminCAdapter().load(_write(tmp_path, [record]), split="test").cases[0]
    assert case.labels["label"] == expected


def test_underscored_label_is_not_accepted(tmp_path):
    """The released spelling has spaces; the old adapter only took underscores."""
    record = dict(RECORD, label="NOT_ENOUGH_INFO")
    with pytest.raises(ValueError, match="unsupported VitaminC label 'NOT_ENOUGH_INFO'"):
        VitaminCAdapter().load(_write(tmp_path, [record]), split="test")


def test_evidence_is_context_and_no_retrieval_ids_are_invented(tmp_path):
    case = VitaminCAdapter().load(_write(tmp_path, [RECORD]), split="test").cases[0]
    assert case.id == "5ed4de07c9e77c000848a180_1"
    assert case.prompt == RECORD["claim"]
    assert case.context == (RECORD["evidence"],)
    assert case.evidence_ids == ()


def test_page_and_revision_are_kept_as_metadata(tmp_path):
    case = VitaminCAdapter().load(_write(tmp_path, [RECORD]), split="test").cases[0]
    assert case.metadata["page"] == "Westlife"
    assert case.metadata["wiki_revision_id"] == "927477259"
    assert case.metadata["revision_type"] == "real"


def test_the_committed_fixture_matches_the_released_shape():
    from pathlib import Path

    fixture = Path(__file__).parent.parent / "fixtures" / "vitaminc.jsonl"
    result = VitaminCAdapter().load(fixture, split="test")
    assert result.record_count == 3
    assert {case.labels["label"] for case in result.cases} == {
        "entailment",
        "contradiction",
        "neutral",
    }
