import json
from pathlib import Path

from llm_wiki_bench.reports import write_report


def test_report_serialization_is_byte_identical_for_same_input(tmp_path: Path) -> None:
    manifest = {
        "suite": "hoh",
        "split": "240601_241201",
        "case_counts": {"errors": 0, "evaluated": 2, "skipped": 0, "total": 2},
        "capabilities_scored": ["answer", "distractor_rejection"],
        "dataset": {"version": "hoh-fixture", "content_digest": "sha256:" + "ab" * 32},
    }
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
        "## Run\n\n"
        "- suite: hoh\n"
        "- split: 240601_241201\n"
        "- version: hoh-fixture\n"
        "- content_digest: sha256:" + "ab" * 32 + "\n"
        '- case_counts: {"errors":0,"evaluated":2,"skipped":0,"total":2}\n'
        '- capabilities_scored: ["answer","distractor_rejection"]\n'
        "\n"
        "## Metrics\n\n"
        "- citation_precision: null\n"
        "- mrr: 0.75\n"
        "- n: 2\n"
    )


def test_report_header_tolerates_a_manifest_missing_the_new_fields(tmp_path: Path) -> None:
    """A manifest without the newer fields (e.g. hand-built in a test, or a
    format this code has not been told about yet) must render `null` for the
    missing header fields rather than crashing report generation."""
    manifest = {"suite": "hoh"}
    write_report(tmp_path, manifest, [], {})
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "- suite: hoh\n" in report
    assert "- split: null\n" in report
    assert "- version: null\n" in report
    assert "- content_digest: null\n" in report
