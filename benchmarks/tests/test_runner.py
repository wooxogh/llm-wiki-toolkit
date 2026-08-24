import json
from pathlib import Path

import pytest
import yaml

from llm_wiki_bench.registry import get_adapter
from llm_wiki_bench.runner import load_config, load_predictions, run_suite, timestamped_run_dir, validate_config


FIXTURES = Path(__file__).parents[1] / "fixtures"


def _config(tmp_path: Path, *, factlens: bool = False) -> dict:
    datasets = {
        name: {"path": str(FIXTURES / f"{name}.jsonl"), "version": f"{name}-v1"}
        for name in ("longmemeval", "hoh", "vitaminc", "rgb")
    }
    if factlens:
        datasets["factlens"] = {"path": str(FIXTURES / "factlens.jsonl"), "version": "factlens-v1"}
    return {"output_root": str(tmp_path / "results"), "top_k": 3, "datasets": datasets}


def _write_predictions(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_validate_config_requires_paths_for_required_suites(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert validate_config(config) == []

    del config["datasets"]["rgb"]["path"]
    assert "datasets.rgb.path is required" in validate_config(config)


def test_load_config_and_predictions_reject_invalid_yaml_shape_and_duplicate_case_ids(tmp_path: Path) -> None:
    config_path = tmp_path / "suite.yaml"
    config_path.write_text("- not-a-mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(config_path)

    predictions_path = _write_predictions(
        tmp_path / "predictions.jsonl",
        [{"case_id": "hoh-1", "answer": "Larkspur"}, {"case_id": "hoh-1", "answer": "Elsewhere"}],
    )
    with pytest.raises(ValueError, match="duplicate prediction case_id"):
        load_predictions(predictions_path)


def test_run_suite_scores_recorded_prediction_and_writes_five_artifacts(tmp_path: Path) -> None:
    config = _config(tmp_path)
    predictions = load_predictions(_write_predictions(tmp_path / "predictions.jsonl", [{"case_id": "hoh-1", "answer": "Larkspur", "ranked_evidence_ids": ["hop-1", "hop-2"], "cited_evidence_ids": ["hop-1", "hop-2"]}]))
    run_dir = tmp_path / "run"

    manifest = run_suite(get_adapter("hoh"), config, predictions, run_dir)

    assert {path.name for path in run_dir.iterdir()} == {"manifest.json", "per_case.jsonl", "metrics.json", "report.md", "skips.jsonl"}
    assert manifest["dataset"] == {"path": config["datasets"]["hoh"]["path"], "version": "hoh-v1"}
    assert manifest["top_k"] == 3
    assert manifest["case_counts"] == {"errors": 0, "evaluated": 1, "skipped": 0, "total": 1}


def test_run_suite_rejects_missing_prediction_unless_skips_allowed(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run_dir = tmp_path / "run"

    with pytest.raises(ValueError, match="missing prediction for case hoh-1"):
        run_suite(get_adapter("hoh"), config, {}, run_dir)

    run_suite(get_adapter("hoh"), config, {}, run_dir, allow_skips=True)
    row = json.loads((run_dir / "per_case.jsonl").read_text(encoding="utf-8"))
    assert row["case_id"] == "hoh-1"
    assert row["reason"] == "missing_prediction"


def test_factlens_absent_from_config_is_not_configured(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert validate_config(config) == []
    with pytest.raises(ValueError, match="factlens: not_configured"):
        run_suite(get_adapter("factlens"), config, {}, tmp_path / "run")


def test_present_factlens_config_requires_mapping_path_and_version(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["datasets"]["factlens"] = None
    assert "datasets.factlens must be a mapping" in validate_config(config)

    config["datasets"]["factlens"] = {"path": ""}
    errors = validate_config(config)
    assert "datasets.factlens.path is required" in errors
    assert "datasets.factlens.version is required" in errors


def test_timestamped_run_dir_atomically_reserves_a_unique_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FixedDateTime:
        @classmethod
        def now(cls, _timezone: object) -> "FixedDateTime":
            return cls()

        def strftime(self, _format: str) -> str:
            return "20260824T010203Z"

    monkeypatch.setattr("llm_wiki_bench.runner.datetime", FixedDateTime)

    first = timestamped_run_dir(tmp_path, "hoh")
    second = timestamped_run_dir(tmp_path, "hoh")

    assert first.is_dir()
    assert second.is_dir()
    assert first != second
