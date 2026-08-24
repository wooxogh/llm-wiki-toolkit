import json

import pytest

from llm_wiki_bench.adapters.longmemeval import LongMemEvalAdapter

RECORD = {
    "question_id": "gpt4_2655b836",
    "question_type": "temporal-reasoning",
    "question": "What was the first issue with my new car?",
    "answer": "GPS system not functioning correctly",
    "question_date": "2023/05/20 (Sat) 14:31",
    "haystack_session_ids": ["s1", "s2"],
    "haystack_dates": ["2023/05/01", "2023/05/10"],
    "haystack_sessions": [
        [
            {"role": "user", "content": "picked up the car"},
            {"role": "assistant", "content": "the GPS was wrong", "has_answer": True},
        ],
        [{"role": "user", "content": "unrelated chat"}],
    ],
    "answer_session_ids": ["s1"],
}


def _write(tmp_path, records):
    path = tmp_path / "longmemeval_oracle.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def test_normalizes_the_released_record(tmp_path):
    case = LongMemEvalAdapter().load(_write(tmp_path, [RECORD]), split="oracle").cases[0]
    assert case.id == "gpt4_2655b836"
    assert case.dataset == "longmemeval"
    assert case.profile == "memory_qa"
    assert case.prompt == "What was the first issue with my new car?"
    assert case.labels["answers"] == ("GPS system not functioning correctly",)


def test_session_and_turn_evidence_are_both_retained(tmp_path):
    case = LongMemEvalAdapter().load(_write(tmp_path, [RECORD]), split="oracle").cases[0]
    assert case.evidence_ids == ("s1",)
    assert case.fine_evidence_ids == ("s1:1",)


def test_context_flattens_sessions_in_order(tmp_path):
    case = LongMemEvalAdapter().load(_write(tmp_path, [RECORD]), split="oracle").cases[0]
    assert case.context == (
        "user: picked up the car",
        "assistant: the GPS was wrong",
        "user: unrelated chat",
    )


def test_question_type_and_date_are_kept_as_metadata(tmp_path):
    case = LongMemEvalAdapter().load(_write(tmp_path, [RECORD]), split="oracle").cases[0]
    assert case.metadata["question_type"] == "temporal-reasoning"
    assert case.metadata["question_date"] == "2023/05/20 (Sat) 14:31"


def test_abs_suffix_marks_an_abstention_question(tmp_path):
    record = dict(RECORD, question_id="gpt4_2655b836_abs")
    case = LongMemEvalAdapter().load(_write(tmp_path, [record]), split="oracle").cases[0]
    assert case.expects_abstention is True


def test_a_non_abs_question_does_not_expect_abstention(tmp_path):
    case = LongMemEvalAdapter().load(_write(tmp_path, [RECORD]), split="oracle").cases[0]
    assert case.expects_abstention is False


def test_session_id_and_session_count_must_agree(tmp_path):
    record = dict(RECORD, haystack_session_ids=["s1"])
    with pytest.raises(ValueError, match="record 1: haystack_session_ids and haystack_sessions differ in length"):
        LongMemEvalAdapter().load(_write(tmp_path, [record]), split="oracle")


def test_missing_released_field_names_the_record(tmp_path):
    record = {key: value for key, value in RECORD.items() if key != "answer"}
    with pytest.raises(ValueError, match="record 1: missing answer"):
        LongMemEvalAdapter().load(_write(tmp_path, [record]), split="oracle")


def test_the_committed_fixture_matches_the_released_shape(tmp_path):
    from pathlib import Path

    fixture = Path(__file__).parent.parent / "fixtures" / "longmemeval.json"
    result = LongMemEvalAdapter().load(fixture, split="oracle")
    assert result.record_count == len(json.loads(fixture.read_text(encoding="utf-8")))
    assert result.cases
