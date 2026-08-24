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
