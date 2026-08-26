import pytest

from llm_wiki_bench.adapters.hoh import HoHAdapter

pytest.importorskip("pyarrow", reason="HoH needs the optional 'hoh' extra")

RECORD = {
    "question": "Which yeast ferments gluconolactone?",
    "answer": "Maudiozyma bulderi",
    "last_modified_time": "2024-07-01T00:00:00",
    "evidence": 'The yeast "Maudiozyma bulderi" ferments gluconolactone.',
    "outdated_infos": [
        {"answer": "Saccharomyces bulderi", "evidence": 'The yeast "Saccharomyces bulderi" ferments gluconolactone.'}
    ],
    "document": {"id": "1000005", "title": "Glucono delta-lactone"},
}


def _write(tmp_path, records):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "hoh.parquet"
    pq.write_table(pa.Table.from_pylist(records), str(path))
    return path


def test_normalizes_the_released_record(tmp_path):
    case = HoHAdapter().load(_write(tmp_path, [RECORD]), split="240601_241201").cases[0]
    assert case.dataset == "hoh"
    assert case.profile == "temporal_discrimination"
    assert case.prompt == "Which yeast ferments gluconolactone?"
    assert case.labels["answers"] == ("Maudiozyma bulderi",)


def test_the_case_id_is_synthesized_from_document_and_record_number(tmp_path):
    """HoH has no record identifier of its own."""
    cases = HoHAdapter().load(_write(tmp_path, [RECORD, RECORD]), split="s").cases
    assert [case.id for case in cases] == ["1000005:1", "1000005:2"]


def test_outdated_answers_become_distractors(tmp_path):
    case = HoHAdapter().load(_write(tmp_path, [RECORD]), split="s").cases[0]
    assert case.labels["distractor_answers"] == ("Saccharomyces bulderi",)


def test_current_evidence_precedes_outdated_evidence_in_context(tmp_path):
    case = HoHAdapter().load(_write(tmp_path, [RECORD]), split="s").cases[0]
    assert case.context == (
        'The yeast "Maudiozyma bulderi" ferments gluconolactone.',
        'The yeast "Saccharomyces bulderi" ferments gluconolactone.',
    )


def test_no_retrieval_metrics_are_claimed(tmp_path):
    """temporal_discrimination does not declare `retrieval`."""
    case = HoHAdapter().load(_write(tmp_path, [RECORD]), split="s").cases[0]
    assert case.evidence_ids == ()


def test_last_modified_time_is_stringified_for_json_artifacts(tmp_path):
    case = HoHAdapter().load(_write(tmp_path, [RECORD]), split="s").cases[0]
    assert isinstance(case.metadata["last_modified_time"], str)


def test_an_empty_outdated_infos_list_is_rejected(tmp_path):
    record = dict(RECORD, outdated_infos=[])
    with pytest.raises(ValueError, match="record 1: outdated_infos must be a non-empty list"):
        HoHAdapter().load(_write(tmp_path, [record]), split="s")


def test_a_missing_document_id_is_rejected(tmp_path):
    record = dict(RECORD, document={"title": "no id"})
    with pytest.raises(ValueError, match="record 1: document.id is required"):
        HoHAdapter().load(_write(tmp_path, [record]), split="s")


def test_a_null_outdated_answer_is_rejected_rather_than_stringified(tmp_path):
    """A `None` answer must not silently become the literal string "None"."""
    record = dict(
        RECORD,
        outdated_infos=[{"answer": None, "evidence": RECORD["outdated_infos"][0]["evidence"]}],
    )
    with pytest.raises(ValueError, match=r"record 1: outdated_infos\[0\]\.answer is required"):
        HoHAdapter().load(_write(tmp_path, [record]), split="s")
