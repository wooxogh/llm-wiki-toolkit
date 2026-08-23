import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


FIXTURES = Path(__file__).parents[1] / "fixtures"
BENCHMARKS = Path(__file__).parents[1]


def _cli_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(BENCHMARKS / "src")
    return environment


def _write_config(tmp_path: Path) -> Path:
    config = {
        "output_root": str(tmp_path / "results"),
        "top_k": 8,
        "datasets": {
            name: {"path": str(FIXTURES / f"{name}.jsonl"), "version": f"{name}-v1"}
            for name in ("longmemeval", "hoh", "vitaminc", "rgb")
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
    assert result.stdout.strip() == "valid"


def test_run_prints_timestamped_artifact_directory_after_writing(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps({"case_id": "hoh-1", "answer": "Larkspur"}) + "\n", encoding="utf-8")

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
