import json
from pathlib import Path

import pytest

from llm_wiki_bench.registry import get_adapter
from llm_wiki_bench.runner import load_config, load_predictions, run_suite, timestamped_run_dir, validate_config
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


def test_run_suite_rejects_a_missing_top_k(tmp_path):
    config = _config(tmp_path)
    del config["top_k"]
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        run_suite(get_adapter("vitaminc"), config, _predictions(), tmp_path / "run")


def test_run_suite_rejects_a_non_positive_top_k(tmp_path):
    config = _config(tmp_path)
    config["top_k"] = 0
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        run_suite(get_adapter("vitaminc"), config, _predictions(), tmp_path / "run")


def _write_predictions(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_load_config_rejects_a_non_mapping_yaml_document(tmp_path):
    config_path = tmp_path / "suite.yaml"
    config_path.write_text("- not-a-mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(config_path)


def test_load_predictions_rejects_a_duplicate_case_id(tmp_path):
    predictions_path = _write_predictions(
        tmp_path / "predictions.jsonl",
        [{"case_id": "fixture000_1", "label": "entailment"}, {"case_id": "fixture000_1", "label": "neutral"}],
    )
    with pytest.raises(ValueError, match="duplicate prediction case_id"):
        load_predictions(predictions_path)


def test_timestamped_run_dir_atomically_reserves_a_unique_directory(tmp_path, monkeypatch):
    class FixedDateTime:
        @classmethod
        def now(cls, _timezone):
            return cls()

        def strftime(self, _format):
            return "20260824T010203Z"

    monkeypatch.setattr("llm_wiki_bench.runner.datetime", FixedDateTime)

    first = timestamped_run_dir(tmp_path, "vitaminc")
    second = timestamped_run_dir(tmp_path, "vitaminc")

    assert first.is_dir()
    assert second.is_dir()
    assert first != second


def test_optional_suite_absent_from_config_is_not_configured(tmp_path):
    config = _config(tmp_path)
    with pytest.raises(ValueError, match="factlens: not_configured"):
        run_suite(get_adapter("factlens"), config, {}, tmp_path / "run")


def test_run_suite_writes_all_five_artifacts(tmp_path):
    run_dir = tmp_path / "run"
    manifest = run_suite(get_adapter("vitaminc"), _config(tmp_path), _predictions(), run_dir)

    assert {path.name for path in run_dir.iterdir()} == {
        "manifest.json",
        "per_case.jsonl",
        "metrics.json",
        "report.md",
        "skips.jsonl",
    }
    assert manifest["dataset"]["path"] == str(FIXTURES / "vitaminc.jsonl")
    assert manifest["dataset"]["version"] == "vitaminc-fixture"
    assert manifest["top_k"] == 8
    assert manifest["case_counts"] == {"errors": 0, "evaluated": 3, "skipped": 0, "total": 3}


def test_allow_skips_records_a_skipped_row_in_per_case(tmp_path):
    predictions = _predictions()
    del predictions["fixture001_1"]
    run_dir = tmp_path / "run"

    run_suite(get_adapter("vitaminc"), _config(tmp_path), predictions, run_dir, allow_skips=True)

    rows = [json.loads(line) for line in (run_dir / "per_case.jsonl").read_text(encoding="utf-8").splitlines()]
    skipped = [row for row in rows if row["case_id"] == "fixture001_1"]
    assert len(skipped) == 1
    assert skipped[0]["status"] == "skipped"
    assert skipped[0]["reason"] == "missing_prediction"


def test_run_suite_resolves_a_profile_per_case_over_a_mixed_profile_dataset(tmp_path):
    """LongMemEval's real release mixes memory_qa and memory_qa_abstention per record.

    The fixture already carries both shapes: an ordinary answerable question
    (memory_qa), an abstention question with no answering turn
    (memory_qa_abstention, since fine_evidence_ids must be empty there), and
    an abstention-labelled question that still has an answering turn
    (memory_qa, per LongMemEvalAdapter's own per-record profile selection).
    """
    config = {
        "output_root": str(tmp_path / "results"),
        "top_k": 8,
        "datasets": {
            "longmemeval": {
                "path": str(FIXTURES / "longmemeval.json"),
                "version": "longmemeval-fixture",
                "split": "test",
            }
        },
    }
    predictions = {
        "fixture_single_session": Prediction(
            case_id="fixture_single_session",
            answer="answer",
            ranked_evidence_ids=("fixture-s1",),
        ),
        "fixture_unanswerable_abs": Prediction(
            case_id="fixture_unanswerable_abs",
            abstained=True,
        ),
        "fixture_answerable_abs": Prediction(
            case_id="fixture_answerable_abs",
            answer="answer",
            ranked_evidence_ids=("fixture-s4",),
        ),
    }

    manifest = run_suite(get_adapter("longmemeval"), config, predictions, tmp_path / "run")

    assert manifest["profiles"] == {"memory_qa": 2, "memory_qa_abstention": 1}
    memory_qa_capabilities = {"retrieval", "fine_retrieval", "answer", "abstention"}
    memory_qa_abstention_capabilities = {"retrieval", "answer", "abstention"}
    assert set(manifest["capabilities_scored"]) == memory_qa_capabilities | memory_qa_abstention_capabilities

    metrics = json.loads((tmp_path / "run" / "metrics.json").read_text(encoding="utf-8"))
    assert "abstention_precision" in metrics
    aggregate = metrics["aggregate"]
    # fine_retrieval is only declared by memory_qa (2 of the 3 rows); recall@1
    # (declared by both profiles) is contributed by all 3.
    assert aggregate["counts"]["fine_recall@1"] == 2
    assert aggregate["counts"]["recall@1"] == 3
    assert aggregate["counts"]["fine_recall@1"] < aggregate["n"]


def test_a_stray_label_key_under_a_non_label_profile_produces_no_confusion_matrix(tmp_path):
    """The confusion matrix must be gated on the profile declaring `label`,
    never on `labels` happening to carry a `"label"` key.

    claim_decomposition declares only `sub_claim_labels`. A case whose
    `labels` dict carries a stray `"label"` entry anyway (e.g. copy-pasted
    from a sibling adapter) must not grow a `label_confusion_matrix` with no
    matching `capabilities_scored` entry.
    """
    from llm_wiki_bench.adapters.base import BenchmarkAdapter

    class _StrayLabelAdapter(BenchmarkAdapter):
        name = "factlens"
        profile = "claim_decomposition"
        container = "jsonl"
        evidence_id_origin = "upstream"
        required = False

        def normalize(self, record, path, record_number, split):
            return {
                "id": str(record["id"]),
                "prompt": "claim",
                "labels": {
                    "sub_claims": ("c1",),
                    "sub_claim_labels": ("true",),
                    "label": "entailment",
                },
                "metadata": {},
            }

    source = tmp_path / "records.jsonl"
    source.write_text(json.dumps({"id": "1"}) + "\n", encoding="utf-8")
    config = {
        "output_root": str(tmp_path / "results"),
        "top_k": 8,
        "datasets": {
            "factlens": {"path": str(source), "version": "fixture", "split": "test"}
        },
    }
    predictions = {"1": Prediction(case_id="1", sub_claim_labels=("true",))}

    run_dir = tmp_path / "run"
    run_suite(_StrayLabelAdapter(), config, predictions, run_dir)

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "label_confusion_matrix" not in metrics
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "label" not in manifest["capabilities_scored"]


def test_run_suite_fails_loudly_on_a_non_serializable_run_parameter(tmp_path):
    """A NaN in config-supplied run_parameters must not crash inside json.dumps unattributed."""
    config = _config(tmp_path, run_parameters={"noise_rate": float("nan")})
    with pytest.raises(ValueError, match="manifest.*not JSON-serializable"):
        run_suite(get_adapter("vitaminc"), config, _predictions(), tmp_path / "run")
