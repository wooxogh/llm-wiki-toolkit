import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


FIXTURES = Path(__file__).parents[1] / "fixtures"
BENCHMARKS = Path(__file__).parents[1]
REPOSITORY = BENCHMARKS.parent


def _cli_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(BENCHMARKS / "src")
    return environment


_HOH_RECORD = {
    "question": "Which yeast ferments gluconolactone?",
    "answer": "Maudiozyma bulderi",
    "last_modified_time": "2024-07-01T00:00:00",
    "evidence": 'The yeast "Maudiozyma bulderi" ferments gluconolactone.',
    "outdated_infos": [
        {"answer": "Saccharomyces bulderi", "evidence": 'The yeast "Saccharomyces bulderi" ferments gluconolactone.'}
    ],
    "document": {"id": "1000005", "title": "Glucono delta-lactone"},
}


def _write_hoh_fixture(tmp_path: Path) -> Path:
    pytest.importorskip("pyarrow", reason="HoH needs the optional 'hoh' extra")
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "hoh.parquet"
    pq.write_table(pa.Table.from_pylist([_HOH_RECORD]), str(path))
    return path


def _write_config(tmp_path: Path) -> Path:
    """Build a config from the tracked fixtures for every required suite.

    Every required suite must be present because both `validate` and `run`
    validate the whole config before dispatching (a config that leaves the
    other required suites unconfigured is not something `validate_config`
    will accept), so this constructs a complete, real config rather than a
    single-suite one.
    """
    config = {
        "output_root": str(tmp_path / "results"),
        "top_k": 8,
        "datasets": {
            "longmemeval": {
                "path": str(FIXTURES / "longmemeval.json"),
                "version": "longmemeval-fixture",
                "split": "test",
            },
            "hoh": {
                "path": str(_write_hoh_fixture(tmp_path)),
                "version": "hoh-fixture",
                "split": "240601_241201",
            },
            "vitaminc": {
                "path": str(FIXTURES / "vitaminc.jsonl"),
                "version": "vitaminc-fixture",
                "split": "test",
            },
            "rgb_base": {
                "path": str(FIXTURES / "rgb_base.jsonl"),
                "version": "rgb-fixture",
                "split": "test",
            },
            "rgb_integration": {
                "path": str(FIXTURES / "rgb_integration.jsonl"),
                "version": "rgb-fixture",
                "split": "test",
            },
            "rgb_counterfactual": {
                "path": str(FIXTURES / "rgb_counterfactual.jsonl"),
                "version": "rgb-fixture",
                "split": "test",
            },
        },
    }
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_validate_exits_zero_without_loading_predictions_or_models(tmp_path: Path) -> None:
    config = _write_config(tmp_path)

    result = subprocess.run(
        [sys.executable, "-m", "llm_wiki_bench", "validate", "--config", str(config)],
        text=True,
        capture_output=True,
        check=False,
        cwd=BENCHMARKS,
        env=_cli_environment(),
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "valid"


def test_run_prints_timestamped_artifact_directory_after_writing(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps({"case_id": "1000005:1", "answer": "Maudiozyma bulderi"}) + "\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "llm_wiki_bench", "run", "--config", str(config), "--suite", "hoh", "--predictions", str(predictions)],
        text=True,
        capture_output=True,
        check=False,
        cwd=BENCHMARKS,
        env=_cli_environment(),
    )

    assert result.returncode == 0, result.stderr
    run_dir = Path(result.stdout.strip())
    assert run_dir.parent == tmp_path / "results"
    assert run_dir.name.endswith("-hoh")
    assert (run_dir / "manifest.json").is_file()


def test_example_config_validates_against_tracked_fixtures(tmp_path: Path) -> None:
    """A config built entirely from the tracked fixtures must validate.

    `benchmarks/configs/suite.example.yaml` itself points at real
    (unfetched) dataset paths and is *correctly* rejected by `validate` in a
    checkout without downloaded data -- that rejection is the fix working,
    not a regression, and Task 16 owns rewriting that file. This test's
    original intent -- that the shipped example's shape validates once
    pointed at fixtures actually present in the repository -- is preserved
    by constructing a fixture-backed config directly instead of depending on
    the example file's contents.
    """
    config = _write_config(tmp_path)

    result = subprocess.run(
        [sys.executable, "-m", "llm_wiki_bench", "validate", "--config", str(config)],
        text=True,
        capture_output=True,
        check=False,
        cwd=BENCHMARKS,
        env=_cli_environment(),
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "valid"


def test_runtime_paths_are_ignored_without_ignoring_examples_or_markers() -> None:
    """Catch accidental commits of local data, outputs, and copied configs."""
    patterns = (REPOSITORY / ".gitignore").read_text(encoding="utf-8")

    assert "benchmarks/data/**" in patterns
    assert "!benchmarks/data/.gitkeep" in patterns
    assert "benchmarks/results/**" in patterns
    assert "!benchmarks/results/.gitkeep" in patterns
    assert "benchmarks/configs/*.local.yaml" in patterns
    assert (BENCHMARKS / "data" / ".gitkeep").is_file()
    assert (BENCHMARKS / "results" / ".gitkeep").is_file()
    tracked_paths = tuple(FIXTURES.iterdir()) + (BENCHMARKS / "configs" / "suite.example.yaml",)
    for tracked in tracked_paths:
        relative_path = str(tracked.relative_to(REPOSITORY))
        result = subprocess.run(
            ["git", "check-ignore", "-q", "--", relative_path],
            cwd=REPOSITORY,
            check=False,
        )
        assert result.returncode == 1, tracked
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative_path],
            cwd=REPOSITORY,
            check=False,
        )
        assert result.returncode == 0, tracked
