"""`wiki-eval --validate-only` must honour a vault's own [eval.minimums].

Run as a real subprocess (not by importing llm_wiki.evaluation.eval directly)
because VAULT_ROOT / VAULT are module-level constants resolved once, at import
time, from WIKI_VAULT — the exact env var a fresh process picks up freshly.
Importing the module in-process would freeze it to whatever vault the test
session happened to import it against first, which is not what this test
needs to control. No torch/numpy import is on this path (see eval.py's own
docstring: `--validate-only` never embeds), so a subprocess here is as cheap
as any other lightweight CLI invocation in this suite.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from llm_wiki import build_index
from conftest import page, write_page

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_validate_only(vault: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "WIKI_VAULT": str(vault)}
    return subprocess.run(
        [sys.executable, "-m", "llm_wiki.evaluation.eval", "--validate-only"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=30,
    )


def _write_one_page_vault(vault: Path) -> None:
    write_page(vault, "domain/research/one.md", page(id="one", summary="measured summary"))
    build_index.write_index(vault, vault / "index.yaml")
    (vault / "eval_gold.json").write_text(
        '[{"q": "what is one", "expect": ["one"], "layer": "domain", "domain": "research", '
        '"category": "direct", "difficulty": "easy", "split": "test"}]',
        encoding="utf-8",
    )


def test_validate_only_passes_under_the_vaults_own_lowered_minimums(vault):
    """This is the exact scenario the bug broke: a vault whose wiki.toml lowers
    [eval.minimums] below the packaged defaults must be able to pass
    --validate-only on its own declared floors, not the packaged total=150 one."""
    _write_one_page_vault(vault)
    (vault / "wiki.toml").write_text(
        "[eval.minimums]\ntotal = 1\nrecent_cases = 0\n"
        "layer = { domain = 1 }\ncategory = {}\n",
        encoding="utf-8",
    )

    result = _run_validate_only(vault)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "gold set valid and complete" in result.stdout
    assert "coverage shortfall" not in result.stdout


def test_validate_only_still_enforces_default_minimums_with_no_override(vault):
    """The fix must not be mistaken for 'minimums are now optional': a vault
    that declares no [eval.minimums] at all (no wiki.toml here) still gets the
    packaged default floors (total=150, ...), so the same tiny gold set fails."""
    _write_one_page_vault(vault)
    # No wiki.toml at all: config.load() falls back to the packaged
    # DEFAULT_MINIMUMS (total=150, layer.domain=100, ...).

    result = _run_validate_only(vault)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "gold set not ready" in result.stdout
    assert "total 1 < 150" in result.stdout


# --------------------------------------------------------------------------
# --calibrate output anchoring (pure; no embedding involved)
# --------------------------------------------------------------------------


def _calibration_out_path(monkeypatch, root: Path, name: str) -> Path:
    from llm_wiki.evaluation import eval as eval_mod

    monkeypatch.setattr(eval_mod, "CONTENT_ROOT", root)
    return eval_mod.calibration_out_path(name)


def test_calibrate_anchors_a_bare_filename_at_the_vault_not_the_cwd(tmp_path, monkeypatch):
    """`wiki-eval --calibrate auto_thresholds.json` is the command the docs teach,
    and its output must land where `resolve_thresholds_path` looks for it. The
    previous rule anchored only when `out.parent` did not exist — but `.` always
    exists, so this exact command silently wrote into the cwd and was never read
    back."""
    assert _calibration_out_path(monkeypatch, tmp_path, "auto_thresholds.json") == \
        tmp_path / "auto_thresholds.json"


def test_calibrate_preserves_a_relative_directory_component(tmp_path, monkeypatch):
    """`--calibrate out/x.json` used to collapse to `<root>/x.json`, discarding
    the directory the caller asked for."""
    assert _calibration_out_path(monkeypatch, tmp_path, "out/x.json") == \
        tmp_path / "out" / "x.json"


def test_calibrate_uses_an_absolute_path_exactly_as_given(tmp_path, monkeypatch):
    absolute = tmp_path / "elsewhere" / "t.json"

    assert _calibration_out_path(monkeypatch, tmp_path / "vault", str(absolute)) == absolute


def test_calibrate_and_gold_resolve_relative_names_the_same_way(tmp_path, monkeypatch):
    """One rule, not two: a relative gold file and a relative calibration target
    both anchor at the content root. Two rules is how the two ends of the same
    round trip stopped meeting."""
    from llm_wiki.evaluation import eval as eval_mod

    monkeypatch.setattr(eval_mod, "CONTENT_ROOT", tmp_path)

    assert eval_mod.resolve_path("auto_thresholds.json") == \
        eval_mod.calibration_out_path("auto_thresholds.json")
