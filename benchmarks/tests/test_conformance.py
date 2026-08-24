import json
from pathlib import Path

import pytest
import yaml

from llm_wiki_bench.__main__ import main
from llm_wiki_bench.registry import get_adapter
from llm_wiki_bench.runner import check_conformance

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_conformance_reports_digest_and_no_failures_for_a_good_source():
    dataset = {
        "path": str(FIXTURES / "vitaminc.jsonl"),
        "version": "fixture",
        "split": "test",
    }
    report = check_conformance(get_adapter("vitaminc"), dataset, limit=None)
    assert report["failures"] == ()
    assert report["record_count"] == 3
    assert report["checked"] == 3
    assert report["content_digest"].startswith("sha256:")


def test_conformance_reports_the_failing_record_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps({"unique_id": "u1", "claim": "c", "evidence": "e", "label": "SUPPORTS"}) + "\n"
        + json.dumps({"unique_id": "u2", "claim": "c", "evidence": "e", "label": "NOT_ENOUGH_INFO"}) + "\n",
        encoding="utf-8",
    )
    dataset = {"path": str(path), "version": "fixture", "split": "test"}
    report = check_conformance(get_adapter("vitaminc"), dataset, limit=None)
    assert len(report["failures"]) == 1
    assert "record 2" in report["failures"][0]


def test_validate_rejects_a_nonexistent_data_path(tmp_path, capsys):
    config = {
        "output_root": str(tmp_path / "out"),
        "top_k": 8,
        "datasets": {
            name: {"path": str(tmp_path / "absent"), "version": "v", "split": "s"}
            for name in (
                "longmemeval",
                "hoh",
                "vitaminc",
                "rgb_base",
                "rgb_integration",
                "rgb_counterfactual",
            )
        },
    }
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["validate", "--config", str(path)])
    assert "does not exist" in capsys.readouterr().err


def test_conformance_subcommand_accepts_a_single_suite(tmp_path, capsys):
    config = {
        "output_root": str(tmp_path / "out"),
        "top_k": 8,
        "datasets": {
            "vitaminc": {
                "path": str(FIXTURES / "vitaminc.jsonl"),
                "version": "fixture",
                "split": "test",
            }
        },
    }
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert main(["conformance", "--config", str(path), "--suite", "vitaminc"]) == 0
    assert "vitaminc" in capsys.readouterr().out


def test_conformance_rejects_a_non_string_split_as_a_config_error_before_reading(tmp_path, capsys):
    """A YAML 1.1 quirk (unquoted `240601_241201` parses as an int, not a
    split name) must fail as a single config error, not as one identical
    failure per record in the source."""
    config = {
        "output_root": str(tmp_path / "out"),
        "top_k": 8,
        "datasets": {
            "vitaminc": {
                "path": str(FIXTURES / "vitaminc.jsonl"),
                "version": "fixture",
                "split": 240601_241201,
            }
        },
    }
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(SystemExit):
        main(["conformance", "--config", str(path), "--suite", "vitaminc"])

    captured = capsys.readouterr()
    assert "datasets.vitaminc.split" in captured.err
    assert captured.out == ""


def test_conformance_cli_exits_nonzero_on_a_malformed_record(tmp_path, capsys):
    """The CLI wiring, not just `check_conformance` directly: a config whose
    source exists but contains a malformed record must make `conformance`
    exit non-zero and print the record-level message."""
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps({"unique_id": "u1", "claim": "c", "evidence": "e", "label": "NOT_ENOUGH_INFO"}) + "\n",
        encoding="utf-8",
    )
    config = {
        "output_root": str(tmp_path / "out"),
        "top_k": 8,
        "datasets": {
            "vitaminc": {"path": str(path), "version": "fixture", "split": "test"},
        },
    }
    config_path = tmp_path / "suite.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    exit_code = main(["conformance", "--config", str(config_path), "--suite", "vitaminc"])

    assert exit_code == 1
    out = capsys.readouterr().out
    assert "1 failures" in out
    assert "record 1" in out


def test_conformance_bounds_repeated_per_record_failures(tmp_path, capsys):
    """A systemic problem must not print one line per record: report a
    bounded number of examples plus a count of the rest."""
    path = tmp_path / "all_bad.jsonl"
    bad_records = [
        {"unique_id": f"u{i}", "claim": "c", "evidence": "e", "label": "NOT_ENOUGH_INFO"} for i in range(1, 51)
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in bad_records), encoding="utf-8")
    config = {
        "output_root": str(tmp_path / "out"),
        "top_k": 8,
        "datasets": {
            "vitaminc": {"path": str(path), "version": "fixture", "split": "test"},
        },
    }
    config_path = tmp_path / "suite.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    report = check_conformance(get_adapter("vitaminc"), config["datasets"]["vitaminc"], limit=None)
    assert report["failure_count"] == 50
    assert len(report["failures"]) == 20
    assert report["content_digest"].startswith("sha256:")

    exit_code = main(["conformance", "--config", str(config_path), "--suite", "vitaminc"])
    assert exit_code == 1
    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line.strip()]
    record_lines = [line for line in lines if line.startswith("  record ")]
    assert len(record_lines) == 20
    assert "... and 30 more" in captured.out
    assert "sha256:" in captured.out
