import json
from pathlib import Path

import pytest

from llm_wiki_bench.adapters.base import BenchmarkAdapter, LoadResult


class _Stub(BenchmarkAdapter):
    name = "vitaminc"
    profile = "grounded_verification"
    container = "jsonl"
    evidence_id_origin = "upstream"

    def normalize(self, record, path, record_number, split):
        return {
            "id": str(record["unique_id"]),
            "prompt": record["claim"],
            "labels": {"label": "entailment"},
            "metadata": {},
        }


def _source(tmp_path: Path) -> Path:
    path = tmp_path / "test.jsonl"
    path.write_text(
        json.dumps({"unique_id": "u1", "claim": "c1"}) + "\n"
        + json.dumps({"unique_id": "u2", "claim": "c2"}) + "\n",
        encoding="utf-8",
    )
    return path


def test_load_returns_cases_with_the_configured_split(tmp_path):
    result = _Stub().load(_source(tmp_path), split="test")
    assert isinstance(result, LoadResult)
    assert [case.id for case in result.cases] == ["u1", "u2"]
    assert {case.split for case in result.cases} == {"test"}
    assert {case.profile for case in result.cases} == {"grounded_verification"}


def test_load_reports_digest_and_record_count(tmp_path):
    result = _Stub().load(_source(tmp_path), split="test")
    assert result.record_count == 2
    assert result.content_digest.startswith("sha256:")


def test_load_records_provenance_in_case_metadata(tmp_path):
    source = _source(tmp_path)
    case = _Stub().load(source, split="test").cases[0]
    assert case.metadata["source_path"] == str(source)
    assert case.metadata["source_record"] == 1
    assert "source_version" not in case.metadata


def test_load_rejects_a_blank_split(tmp_path):
    with pytest.raises(ValueError, match="split must be a non-blank string"):
        _Stub().load(_source(tmp_path), split="  ")


def test_load_rejects_an_unknown_container():
    class _Bad(_Stub):
        container = "xml"

    with pytest.raises(ValueError, match="unknown container: xml"):
        _Bad().load(Path("unused.jsonl"), split="test")
