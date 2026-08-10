#!/usr/bin/env python3
"""Deterministic daily ingest: fixed step order, fail-fast, stamp last.

A shell version of this pipeline once recorded a successful day whenever the
authoring agent exited 0, and only then ran the hygiene commands — so a day
where the index or the embedding store failed to rebuild still got stamped
"done" and was never retried. This orchestrator inverts that: the success stamp
is written only after every artifact has been rebuilt AND the health gate has
passed.

Order (each must succeed before the next runs):

  preflight refuse to start on a dirty working tree               → fatal if ANY
  llm       author/update pages from the configured repos' new commits (skippable;
            skipped entirely when [ingest] repos is empty — see below)
  build     index.yaml
  embed     .embeddings/
  graph     GRAPH_REPORT.md
  community COMMUNITIES.md
  stale     communities still awaiting a grounded synthesis      → fatal if ANY
  health    wiki_health --mode full                              → fatal on error
  commit    commit whatever the above regenerated
  ---- stamp written here ----
  push      backup to origin; failure does NOT clear the stamp

The `llm` step invokes an external agent CLI against the repositories listed
under `[ingest] repos` in wiki.toml. Running that step with no repos configured
would mean running an agent against nothing, so it is omitted from the step
list entirely whenever that list is empty — the default for anyone who has not
configured it.

  python -m integrations.ingest.ingest_pipeline                 # the daily run
  python -m integrations.ingest.ingest_pipeline --dry-run --skip-llm
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from llm_wiki import config as _config
from llm_wiki.paths import VAULT_ROOT

DEFAULT_STAMP = Path.home() / ".local" / "share" / "llm-wiki" / "last-run-date"
DEFAULT_PROMPT_FILE = Path(__file__).with_name("prompt.txt")


def _load_prompt(vault: Path) -> str:
    """Read the authoring prompt: `[ingest] prompt_file` if set, else the packaged default."""
    prompt_file = _config.load(vault).ingest_prompt_file
    if prompt_file:
        path = Path(prompt_file)
        path = path if path.is_absolute() else vault / path
    else:
        path = DEFAULT_PROMPT_FILE
    return path.read_text(encoding="utf-8").strip()


@dataclass(frozen=True)
class Step:
    name: str
    argv: tuple
    fatal_if_output: bool = False
    allow_failure: bool = False
    fatal_hint: str = ""  # shown when fatal_if_output trips, to say what to do


@dataclass(frozen=True)
class PipelineConfig:
    vault: Path
    steps: tuple
    stamp_path: Path
    today: str
    push: object = None
    dry_run: bool = False

    @classmethod
    def default(cls, vault: Path = VAULT_ROOT, today: str = "",
                stamp_path: Path = DEFAULT_STAMP, skip_llm: bool = False,
                dry_run: bool = False, repos: tuple | None = None) -> "PipelineConfig":
        py = sys.executable  # launchd/cron have no profile; a bare `python3` may resolve wrong
        if repos is None:
            repos = _config.load(vault).ingest_repos
        # Refuse to start on a dirty tree. The commit step stages whole content
        # directories, so from a dirty start it cannot distinguish its own output
        # from a human's work in progress — and would silently fold their
        # uncommitted edits into a "daily wiki refresh" commit. Starting clean is
        # what makes "everything dirty at the end is ours" true.
        steps = [Step("preflight", ("git", "-C", str(vault), "status", "--porcelain"),
                      fatal_if_output=True,
                      fatal_hint="uncommitted changes present — commit, stash, or revert them "
                                 "first; ingest will retry on the next tick")]
        if not skip_llm and repos:
            steps.append(Step("llm", (
                "claude", "-p", _load_prompt(vault),
                "--permission-mode", "acceptEdits",
                "--allowedTools", "Bash", "Read", "Grep", "Glob", "Edit", "Write",
            )))
        steps = list(steps) + [
            Step("build", (py, "-m", "llm_wiki.build_index")),
            Step("embed", (py, "-m", "llm_wiki.retrieval.embed_index")),
            Step("graph", (py, "-m", "llm_wiki.reports.graph_report", "--write")),
            Step("community", (py, "-m", "llm_wiki.reports.community_report", "--write")),
            Step("stale", (py, "-m", "llm_wiki.reports.community_report", "--stale"),
                 fatal_if_output=True),
            Step("health", (py, "-m", "llm_wiki.wiki_health", "--mode", "full")),
            Step("commit", (str(Path(__file__).with_name("commit_artifacts.sh")),)),
        ]
        return cls(vault=vault, steps=tuple(steps), stamp_path=stamp_path,
                   today=today or dt.date.today().isoformat(),
                   push=Step("push", ("git", "-C", str(vault), "push", "origin", "HEAD"),
                             allow_failure=True),
                   dry_run=dry_run)


def _run(step: Step, config: PipelineConfig, runner) -> tuple:
    # WIKI_VAULT is exported (not just cwd) so a shell-script step that derives
    # its own vault (e.g. commit_artifacts.sh, which may live outside the vault
    # once this package is installed separately from the content it manages)
    # agrees with the vault this pipeline was configured for.
    env = {**os.environ, "WIKI_VAULT": str(config.vault)}
    result = runner(step.argv, cwd=str(config.vault), env=env, capture_output=True, text=True)
    out = (result.stdout or "").strip()
    if out:
        print(out)
    err = (result.stderr or "").strip()
    if err:
        print(err, file=sys.stderr)
    return result.returncode, out


def run_pipeline(config: PipelineConfig, runner=subprocess.run) -> int:
    stamped = (config.stamp_path.exists()
               and config.stamp_path.read_text().strip() == config.today)

    # Dry-run is checked FIRST: its job is to show what would run. Letting the
    # once-a-day guard short-circuit it would print nothing on exactly the days
    # someone is most likely to be inspecting the pipeline.
    if config.dry_run:
        if stamped:
            print(f"[note] stamp already set to {config.today}; a real run would skip everything")
        for step in list(config.steps) + ([config.push] if config.push else []):
            print(f"[dry-run] {step.name}: {' '.join(step.argv)}")
        return 0

    if stamped:
        print(f"[skip] already succeeded on {config.today}")
        return 0

    for step in config.steps:
        print(f"== {step.name} ==")
        code, out = _run(step, config, runner)
        if code != 0:
            print(f"❌ {step.name} failed (rc={code}) — stamp NOT written, next tick retries",
                  file=sys.stderr)
            return code
        if step.fatal_if_output and out:
            hint = f" — {step.fatal_hint}" if step.fatal_hint else ""
            print(f"❌ {step.name} produced output that must be empty{hint}. "
                  f"Stamp NOT written.\n{out}", file=sys.stderr)
            return 1

    config.stamp_path.parent.mkdir(parents=True, exist_ok=True)
    config.stamp_path.write_text(config.today, encoding="utf-8")
    print(f"✓ ingest done (stamped {config.today})")

    if config.push:
        print(f"== {config.push.name} ==")
        code, _ = _run(config.push, config, runner)
        if code != 0:
            # The knowledge is safely committed locally; only the off-machine
            # backup is behind. Clearing the stamp would re-run a full ingest
            # tomorrow because of a network blip.
            print("⚠ push failed — local stamp preserved, next cycle retries", file=sys.stderr)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-llm", action="store_true",
                    help="run only the deterministic hygiene/health steps")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the exact ordered commands; change nothing")
    ap.add_argument("--vault", type=Path, default=VAULT_ROOT)
    ap.add_argument("--stamp", type=Path, default=DEFAULT_STAMP)
    args = ap.parse_args()

    config = PipelineConfig.default(vault=args.vault.resolve(), stamp_path=args.stamp,
                                    skip_llm=args.skip_llm, dry_run=args.dry_run)
    return run_pipeline(config)


if __name__ == "__main__":
    raise SystemExit(main())
