import json

import pytest

from llm_wiki_bench.readers import (
    READERS,
    file_digest,
    read_csv,
    read_json_array,
    read_jsonl,
    read_parquet,
)


def test_read_json_array_yields_numbered_records(tmp_path):
    path = tmp_path / "a.json"
    path.write_text(json.dumps([{"a": 1}, {"a": 2}]), encoding="utf-8")
    assert list(read_json_array(path)) == [(1, {"a": 1}), (2, {"a": 2})]


def test_read_json_array_rejects_a_top_level_object(tmp_path):
    path = tmp_path / "a.json"
    path.write_text(json.dumps({"a": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="expected a JSON array"):
        list(read_json_array(path))


def test_read_jsonl_skips_blank_lines_without_consuming_a_record_number(tmp_path):
    path = tmp_path / "a.jsonl"
    path.write_text('{"a":1}\n\n{"a":2}\n', encoding="utf-8")
    assert list(read_jsonl(path)) == [(1, {"a": 1}), (2, {"a": 2})]


def test_read_jsonl_reports_the_offending_record_number(tmp_path):
    path = tmp_path / "a.jsonl"
    path.write_text('{"a":1}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="record 2: invalid JSON"):
        list(read_jsonl(path))


def test_read_jsonl_accepts_a_json_extension(tmp_path):
    """RGB ships JSON Lines under a .json extension."""
    path = tmp_path / "en.json"
    path.write_text('{"id":0}\n{"id":1}\n', encoding="utf-8")
    assert [record for _, record in read_jsonl(path)] == [{"id": 0}, {"id": 1}]


def test_read_csv_yields_dict_rows(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("ind,claim\n0,hello\n", encoding="utf-8")
    assert list(read_csv(path)) == [(1, {"ind": "0", "claim": "hello"})]


def test_read_csv_reports_malformed_row(tmp_path):
    """Malformed CSV (field larger than limit) should report record number and source."""
    path = tmp_path / "a.csv"
    # Write a CSV with a field larger than the csv module's default limit (131072 bytes),
    # which triggers csv.Error: field larger than field limit
    large_field = "x" * 200000
    path.write_text(f"a,b\n1,2\n3,{large_field}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="record 2:"):
        list(read_csv(path))


def test_file_digest_is_stable_and_prefixed(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"abc")
    digest = file_digest(path)
    assert digest.startswith("sha256:")
    assert digest == file_digest(path)
    assert digest == (
        "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_readers_registry_exposes_every_container():
    assert set(READERS) == {"json_array", "jsonl", "csv", "parquet"}


def test_read_parquet_without_pyarrow_explains_the_extra(tmp_path, monkeypatch):
    # A None entry in sys.modules makes `import pyarrow.parquet` raise
    # ImportError ("halted; None in sys.modules"), which is how this simulates
    # the extra being absent without uninstalling anything.
    monkeypatch.setitem(__import__("sys").modules, "pyarrow.parquet", None)
    path = tmp_path / "a.parquet"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match=r"benchmarks\[hoh\]"):
        list(read_parquet(path))
