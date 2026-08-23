"""Command line interface for offline benchmark validation and scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .registry import get_adapter
from .reports import write_report
from .runner import load_config, load_predictions, run_suite, timestamped_run_dir, validate_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m llm_wiki_bench")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--config", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--suite", required=True)
    run.add_argument("--predictions", type=Path, required=True)
    run.add_argument("--allow-skips", action="store_true")
    report = commands.add_parser("report")
    report.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            errors = validate_config(load_config(args.config))
            if errors:
                raise ValueError("; ".join(errors))
            print("valid")
            return 0
        if args.command == "run":
            config = load_config(args.config)
            errors = validate_config(config)
            if errors:
                raise ValueError("; ".join(errors))
            run_dir = timestamped_run_dir(Path(config["output_root"]), args.suite)
            run_suite(get_adapter(args.suite), config, load_predictions(args.predictions), run_dir, args.allow_skips)
            print(run_dir)
            return 0
        _rewrite_report(args.run_dir)
        print(args.run_dir / "report.md")
        return 0
    except ValueError as error:
        parser.error(str(error))
    return 2


def _rewrite_report(run_dir: Path) -> None:
    try:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in (run_dir / "per_case.jsonl").read_text(encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read run artifacts from {run_dir}: {error}") from error
    write_report(run_dir, manifest, rows, metrics)


if __name__ == "__main__":
    raise SystemExit(main())
