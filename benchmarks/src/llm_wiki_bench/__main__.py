"""Command line interface for offline benchmark validation and scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .registry import enabled_adapters, get_adapter
from .reports import write_report
from .runner import (
    check_conformance,
    dataset_entry_errors,
    load_config,
    load_predictions,
    run_suite,
    timestamped_run_dir,
    validate_config,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m llm_wiki_bench")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--config", type=Path, required=True)
    validate.add_argument("--sample", type=int, default=5)
    run = commands.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--suite", required=True)
    run.add_argument("--predictions", type=Path, required=True)
    run.add_argument("--allow-skips", action="store_true")
    report = commands.add_parser("report")
    report.add_argument("--run-dir", type=Path, required=True)
    conformance = commands.add_parser("conformance")
    conformance.add_argument("--config", type=Path, required=True)
    conformance.add_argument("--suite", default=None)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            config = load_config(args.config)
            errors = validate_config(config)
            if errors:
                raise ValueError("; ".join(errors))
            for adapter in enabled_adapters(config):
                conformance_report = check_conformance(adapter, config["datasets"][adapter.name], args.sample)
                if conformance_report["failure_count"]:
                    omitted = conformance_report["failure_count"] - len(conformance_report["failures"])
                    suffix = f"; ... and {omitted} more" if omitted > 0 else ""
                    raise ValueError(f"{adapter.name}: " + "; ".join(conformance_report["failures"]) + suffix)
                print(f"{adapter.name}: {conformance_report['checked']}/{conformance_report['record_count']} sampled ok")
            print("valid")
            return 0
        if args.command == "conformance":
            config = load_config(args.config)
            names = [args.suite] if args.suite else [adapter.name for adapter in enabled_adapters(config)]
            # A config-level error (a bad path, a non-string split, a missing
            # entry) must surface as a config error before any record is
            # read: otherwise a single misconfigured value fails every
            # record in the source with the same message, drowning the root
            # cause under however many records the release has. Validate
            # every dataset entry in scope up front, across all suites in
            # scope, before touching any of them.
            datasets = config.get("datasets")
            if not isinstance(datasets, dict):
                raise ValueError("datasets must be a mapping")
            errors: list[str] = []
            entries: dict[str, dict] = {}
            for name in names:
                details = datasets.get(name)
                if not isinstance(details, dict):
                    errors.append(f"datasets.{name} is required")
                    continue
                entry_errors = dataset_entry_errors(name, details)
                if entry_errors:
                    errors.extend(entry_errors)
                else:
                    entries[name] = details
            if errors:
                raise ValueError("; ".join(errors))
            failed = False
            for name in names:
                adapter = get_adapter(name)
                conformance_report = check_conformance(adapter, entries[name], None)
                failure_count = conformance_report["failure_count"]
                status = "ok" if not failure_count else f"{failure_count} failures"
                print(f"{name}: {conformance_report['record_count']} records, {conformance_report['content_digest']}, {status}")
                for failure in conformance_report["failures"]:
                    print(f"  {failure}")
                omitted = failure_count - len(conformance_report["failures"])
                if omitted > 0:
                    print(f"  ... and {omitted} more")
                failed = failed or bool(failure_count)
            return 1 if failed else 0
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
