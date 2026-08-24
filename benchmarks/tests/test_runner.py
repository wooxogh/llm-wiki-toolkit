import json
from pathlib import Path

import pytest

from llm_wiki_bench.registry import get_adapter
from llm_wiki_bench.runner import load_config, run_suite, validate_config
from llm_wiki_bench.schema import Prediction

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _config(tmp_path, **overrides):
    dataset = {
        "path": str(FIXTURES / "vitaminc.jsonl"),
        "version": "vitaminc-fixture",
        "split": "test",
    }
    dataset.update(overrides)
    return {
        "output_root": str(tmp_path / "results"),
        "top_k": 8,
        "datasets": {"vitaminc": dataset},
    }


def _predictions():
    return {
        "fixture000_1": Prediction(case_id="fixture000_1", label="entailment"),
        "fixture000_2": Prediction(case_id="fixture000_2", label="contradiction"),
        "fixture001_1": Prediction(case_id="fixture001_1", label="neutral"),
    }


def test_validate_config_requires_a_split(tmp_path):
    config = _config(tmp_path)
    del config["datasets"]["vitaminc"]["split"]
    assert "datasets.vitaminc.split is required" in validate_config(config)


def test_validate_config_rejects_a_missing_data_file(tmp_path):
    config = _config(tmp_path, path=str(tmp_path / "absent.jsonl"))
    errors = validate_config(config)
    assert any("does not exist" in error for error in errors)


def test_manifest_records_the_digest_and_record_count(tmp_path):
    run_dir = tmp_path / "run"
    manifest = run_suite(get_adapter("vitaminc"), _config(tmp_path), _predictions(), run_dir)
    assert manifest["dataset"]["content_digest"].startswith("sha256:")
    assert manifest["dataset"]["record_count"] == 3
    assert manifest["evidence_id_origin"] == "upstream"
    assert manifest["profiles"] == {"grounded_verification": 3}
    assert manifest["capabilities_scored"] == ["label"]


def test_a_pinned_digest_mismatch_fails_the_run(tmp_path):
    config = _config(tmp_path, expected_digest="sha256:" + "0" * 64)
    with pytest.raises(ValueError, match="content digest mismatch"):
        run_suite(get_adapter("vitaminc"), config, _predictions(), tmp_path / "run")


def test_a_matching_pinned_digest_passes(tmp_path):
    first = run_suite(get_adapter("vitaminc"), _config(tmp_path), _predictions(), tmp_path / "a")
    config = _config(tmp_path, expected_digest=first["dataset"]["content_digest"])
    manifest = run_suite(get_adapter("vitaminc"), config, _predictions(), tmp_path / "b")
    assert manifest["dataset"]["content_digest"] == first["dataset"]["content_digest"]


def test_declared_run_parameters_are_recorded_verbatim(tmp_path):
    config = _config(tmp_path, run_parameters={"noise_rate": 0.6, "passage_num": 5})
    manifest = run_suite(get_adapter("vitaminc"), config, _predictions(), tmp_path / "run")
    assert manifest["run_parameters"] == {"noise_rate": 0.6, "passage_num": 5}


def test_only_declared_capabilities_are_scored(tmp_path):
    """grounded_verification declares `label`; retrieval must not appear."""
    run_dir = tmp_path / "run"
    run_suite(get_adapter("vitaminc"), _config(tmp_path), _predictions(), run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "label_accuracy" in metrics["aggregate"]
    assert not [key for key in metrics["aggregate"] if key.startswith("recall@")]
    assert "mrr" not in metrics["aggregate"]


def test_abstention_summary_appears_only_for_profiles_declaring_it(tmp_path):
    run_dir = tmp_path / "run"
    run_suite(get_adapter("vitaminc"), _config(tmp_path), _predictions(), run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "abstention_precision" not in metrics


def test_a_missing_prediction_still_fails_without_allow_skips(tmp_path):
    predictions = _predictions()
    del predictions["fixture001_1"]
    with pytest.raises(ValueError, match="missing prediction for case fixture001_1"):
        run_suite(get_adapter("vitaminc"), _config(tmp_path), predictions, tmp_path / "run")
