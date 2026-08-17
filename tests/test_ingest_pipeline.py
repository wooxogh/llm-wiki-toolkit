"""Fail-fast orchestration and success-stamp ordering.

Every step here is a REAL executable script that appends its own name to a trace
file. Asserting on a mock's call list would only prove the orchestrator called
what the test told it to call; asserting on a trace written by processes that
actually ran proves the ordering and the short-circuit are real.
"""
from __future__ import annotations

import stat
from pathlib import Path

import pytest

from integrations.ingest.ingest_pipeline import PipelineConfig, Step, authoring_argv, run_pipeline

TRACE = "trace.txt"
STEP_NAMES = ("build", "embed", "graph", "community", "stale", "health", "commit")


def make_script(tmp_path: Path, name: str, exit_code: int = 0, stdout: str = "") -> Path:
    """A tiny shell script that records that it ran, then exits as instructed."""
    path = tmp_path / f"{name}.sh"
    body = f'#!/bin/sh\nprintf "%s\\n" "{name}" >> "{tmp_path / TRACE}"\n'
    if stdout:
        body += f'printf "%s\\n" {stdout!r}\n'
    body += f"exit {exit_code}\n"
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def read_trace(tmp_path: Path) -> list:
    path = tmp_path / TRACE
    return path.read_text(encoding="utf-8").split() if path.exists() else []


def pipeline_fixture(tmp_path: Path, outcomes: dict, stdouts: dict = None,
                     push_code: int = 0, dry_run: bool = False) -> PipelineConfig:
    stdouts = stdouts or {}
    steps = tuple(
        Step(name=n,
             argv=(str(make_script(tmp_path, n, outcomes.get(n, 0), stdouts.get(n, ""))),),
             fatal_if_output=(n == "stale"))
        for n in STEP_NAMES
    )
    return PipelineConfig(
        vault=tmp_path,
        steps=steps,
        stamp_path=tmp_path / "last-run-date",
        today="2026-07-31",
        push=Step(name="push", argv=(str(make_script(tmp_path, "push", push_code)),)),
        dry_run=dry_run,
    )


# --------------------------------------------------------------------------
# ordering and stamping
# --------------------------------------------------------------------------


def test_pipeline_stamps_only_after_all_health_steps_pass(tmp_path):
    config = pipeline_fixture(tmp_path, outcomes={})

    result = run_pipeline(config)

    assert result == 0
    assert read_trace(tmp_path)[:7] == [
        "build", "embed", "graph", "community", "stale", "health", "commit"
    ]
    assert config.stamp_path.read_text() == config.today


def test_pipeline_failure_does_not_write_stamp_or_run_later_steps(tmp_path):
    config = pipeline_fixture(tmp_path, outcomes={"embed": 7})

    result = run_pipeline(config)

    assert result == 7
    assert read_trace(tmp_path) == ["build", "embed"]
    assert not config.stamp_path.exists()


def test_a_failed_health_gate_blocks_the_commit_and_the_stamp(tmp_path):
    config = pipeline_fixture(tmp_path, outcomes={"health": 1})

    result = run_pipeline(config)

    assert result == 1
    assert "commit" not in read_trace(tmp_path)
    assert not config.stamp_path.exists()


def test_stale_community_output_is_fatal_even_on_exit_zero(tmp_path):
    """`community_report --stale` exits 0 and prints the stale communities.
    Treating exit code alone as the signal would let a stale synthesis through."""
    config = pipeline_fixture(tmp_path, outcomes={},
                              stdouts={"stale": '{"sig": "abc", "label": "research"}'})

    result = run_pipeline(config)

    assert result != 0
    assert read_trace(tmp_path) == ["build", "embed", "graph", "community", "stale"]
    assert not config.stamp_path.exists()


def test_empty_stale_output_is_not_fatal(tmp_path):
    config = pipeline_fixture(tmp_path, outcomes={}, stdouts={"stale": ""})

    assert run_pipeline(config) == 0


# --------------------------------------------------------------------------
# once-a-day guard
# --------------------------------------------------------------------------


def test_an_existing_current_day_stamp_skips_every_step(tmp_path):
    config = pipeline_fixture(tmp_path, outcomes={})
    config.stamp_path.write_text("2026-07-31", encoding="utf-8")

    result = run_pipeline(config)

    assert result == 0
    assert read_trace(tmp_path) == []


def test_a_stale_stamp_does_not_skip(tmp_path):
    config = pipeline_fixture(tmp_path, outcomes={})
    config.stamp_path.write_text("2026-07-30", encoding="utf-8")

    run_pipeline(config)

    assert read_trace(tmp_path)[0] == "build"
    assert config.stamp_path.read_text() == "2026-07-31"


# --------------------------------------------------------------------------
# push
# --------------------------------------------------------------------------


def test_a_failed_push_preserves_the_local_success_stamp(tmp_path):
    """The local commits succeeded; the backup did not. Clearing the stamp would
    re-run the whole ingest tomorrow for a network problem."""
    config = pipeline_fixture(tmp_path, outcomes={}, push_code=1)

    result = run_pipeline(config)

    assert result == 0
    assert config.stamp_path.read_text() == config.today
    assert read_trace(tmp_path)[-1] == "push"


def test_push_runs_after_the_stamp_is_written(tmp_path):
    config = pipeline_fixture(tmp_path, outcomes={})

    run_pipeline(config)

    assert read_trace(tmp_path) == list(STEP_NAMES) + ["push"]


# --------------------------------------------------------------------------
# dry run
# --------------------------------------------------------------------------


def test_dry_run_executes_nothing_and_changes_no_state(tmp_path, capsys):
    config = pipeline_fixture(tmp_path, outcomes={}, dry_run=True)

    result = run_pipeline(config)

    assert result == 0
    assert read_trace(tmp_path) == []
    assert not config.stamp_path.exists()
    printed = capsys.readouterr().out
    for name in STEP_NAMES:
        assert name in printed


def test_dry_run_still_prints_the_plan_when_today_is_already_stamped(tmp_path, capsys):
    """Inspecting the pipeline must work on a day it has already succeeded —
    that is precisely when someone is checking what it would do."""
    config = pipeline_fixture(tmp_path, outcomes={}, dry_run=True)
    config.stamp_path.write_text(config.today, encoding="utf-8")

    assert run_pipeline(config) == 0

    printed = capsys.readouterr().out
    assert "build.sh" in printed
    assert read_trace(tmp_path) == []


def test_dry_run_prints_the_steps_in_execution_order(tmp_path, capsys):
    run_pipeline(pipeline_fixture(tmp_path, outcomes={}, dry_run=True))

    printed = capsys.readouterr().out
    positions = [printed.index(f"{n}.sh") for n in STEP_NAMES]
    assert positions == sorted(positions)


# --------------------------------------------------------------------------
# real command construction
# --------------------------------------------------------------------------


def test_default_steps_use_the_running_interpreter_not_bare_python3(tmp_path):
    """launchd/cron have no shell profile; a bare `python3` would resolve to
    whatever the system PATH picks, which may lack this project's dependencies."""
    import sys

    config = PipelineConfig.default(vault=tmp_path, today="2026-07-31",
                                    stamp_path=tmp_path / "stamp", skip_llm=True)

    python_steps = [s for s in config.steps if s.argv[0] == sys.executable]
    assert python_steps
    assert all("python3" != a for s in config.steps for a in s.argv[1:])


def test_default_steps_are_ordered_build_embed_then_health_then_commit(tmp_path):
    config = PipelineConfig.default(vault=tmp_path, today="2026-07-31",
                                    stamp_path=tmp_path / "stamp", skip_llm=True)
    names = [s.name for s in config.steps]

    assert names.index("build") < names.index("embed") < names.index("health")
    assert names.index("health") < names.index("commit")
    assert "llm" not in names


def test_skip_llm_false_adds_the_authoring_step_right_after_preflight(tmp_path):
    config = PipelineConfig.default(vault=tmp_path, today="2026-07-31",
                                    stamp_path=tmp_path / "stamp", skip_llm=False,
                                    repos=("/tmp/example-repo",))

    assert [s.name for s in config.steps[:3]] == ["preflight", "llm", "build"]


def test_codex_authoring_step_uses_exec_and_adds_repo_read_roots(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    config = PipelineConfig.default(vault=tmp_path, today="2026-07-31",
                                    stamp_path=tmp_path / "stamp", skip_llm=False,
                                    repos=(repo,), agent="codex")

    argv = config.steps[1].argv
    assert argv[:2] == ("codex", "exec")
    assert ("--cd", str(tmp_path)) in zip(argv, argv[1:])
    assert ("--sandbox", "workspace-write") in zip(argv, argv[1:])
    assert ("--ask-for-approval", "never") in zip(argv, argv[1:])
    assert ("--add-dir", str(repo.resolve())) in zip(argv, argv[1:])


def test_unknown_authoring_agent_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown agent"):
        authoring_argv("other", tmp_path, ())


def test_output_fatal_steps_are_exactly_preflight_and_stale(tmp_path):
    config = PipelineConfig.default(vault=tmp_path, today="2026-07-31",
                                    stamp_path=tmp_path / "stamp", skip_llm=True)

    fatal = [s.name for s in config.steps if s.fatal_if_output]
    assert fatal == ["preflight", "stale"]


def test_no_configured_repos_skips_the_authoring_step_even_when_skip_llm_is_false(tmp_path):
    """Running the authoring agent against an empty repo list would mean running
    it against nothing. With no wiki.toml (and so no [ingest] repos) at `vault`,
    the llm step must not appear even though skip_llm asks for it to run."""
    config = PipelineConfig.default(vault=tmp_path, today="2026-07-31",
                                    stamp_path=tmp_path / "stamp", skip_llm=False)

    assert "llm" not in [s.name for s in config.steps]


# --------------------------------------------------------------------------
# working-tree safety: the pipeline must never absorb someone else's edits
# --------------------------------------------------------------------------


def test_preflight_runs_before_everything_including_the_authoring_step(tmp_path):
    config = PipelineConfig.default(vault=tmp_path, today="2026-07-31",
                                    stamp_path=tmp_path / "stamp", skip_llm=False,
                                    repos=("/tmp/example-repo",))

    assert config.steps[0].name == "preflight"
    assert config.steps[1].name == "llm"


def test_a_dirty_working_tree_aborts_before_any_step_runs(tmp_path):
    """A human's uncommitted edit must never end up in a daily-ingest commit.

    The pipeline stages whole content directories, so if it started on a dirty
    tree it could not tell its own output from someone's work in progress.
    """
    config = pipeline_fixture_with_preflight(
        tmp_path, outcomes={}, preflight_output=" M domain/research/some-page.md")

    result = run_pipeline(config)

    assert result != 0
    assert read_trace(tmp_path) == ["preflight"]
    assert not config.stamp_path.exists()


def test_a_clean_working_tree_proceeds_through_every_step(tmp_path):
    config = pipeline_fixture_with_preflight(tmp_path, outcomes={}, preflight_output="")

    result = run_pipeline(config)

    assert result == 0
    assert read_trace(tmp_path)[:2] == ["preflight", "build"]
    assert config.stamp_path.read_text() == config.today


def test_a_dirty_tree_abort_names_the_offending_paths(tmp_path, capsys):
    config = pipeline_fixture_with_preflight(
        tmp_path, outcomes={}, preflight_output=" M domain/research/some-page.md")

    run_pipeline(config)

    printed = capsys.readouterr().out + capsys.readouterr().err
    assert "some-page.md" in printed


def pipeline_fixture_with_preflight(tmp_path: Path, outcomes: dict,
                                    preflight_output: str) -> PipelineConfig:
    names = ("preflight",) + STEP_NAMES
    steps = tuple(
        Step(name=n,
             argv=(str(make_script(tmp_path, n, outcomes.get(n, 0),
                                   preflight_output if n == "preflight" else "")),),
             fatal_if_output=(n in ("preflight", "stale")),
             fatal_hint="working tree is dirty" if n == "preflight" else "")
        for n in names
    )
    return PipelineConfig(
        vault=tmp_path,
        steps=steps,
        stamp_path=tmp_path / "last-run-date",
        today="2026-07-31",
        push=Step(name="push", argv=(str(make_script(tmp_path, "push", 0)),)),
    )
