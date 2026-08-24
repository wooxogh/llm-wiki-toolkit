import json
from pathlib import Path

from llm_wiki_bench.reports import write_report


def test_report_serialization_is_byte_identical_for_same_input(tmp_path: Path) -> None:
    manifest = {"suite": "hoh", "source": "fixture.jsonl"}
    per_case = [{"case_id": "case-2", "scores": {"mrr": 0.5}}, {"case_id": "case-1", "scores": {"mrr": 1.0}}]
    metrics = {"mrr": 0.75, "n": 2, "citation_precision": None}
    first, second = tmp_path / "first", tmp_path / "second"

    write_report(first, manifest, per_case, metrics)
    write_report(second, manifest, per_case, metrics)

    names = ("manifest.json", "per_case.jsonl", "metrics.json", "report.md")
    assert all((first / name).read_bytes() == (second / name).read_bytes() for name in names)
    assert json.loads((first / "manifest.json").read_text(encoding="utf-8")) == manifest
    assert (first / "per_case.jsonl").read_text(encoding="utf-8").splitlines()[0].startswith('{"case_id":"case-2"')
    assert (first / "report.md").read_text(encoding="utf-8") == (
        "# Benchmark report\n\n"
        "## Metrics\n\n"
        "- citation_precision: null\n"
        "- mrr: 0.75\n"
        "- n: 2\n"
    )
